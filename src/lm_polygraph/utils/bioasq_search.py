
import os
import glob
import gzip
import xml.etree.ElementTree as ET
import torch.multiprocessing as mp
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import time
import queue
import json # Added for BioASQ verification

# --- CONFIGURATION ---
# Point this to where your .xml.gz files are
PUBMED_BASELINE_DIR = "/common/users/yl2310/pubmed_2025"
DATA_DIR = os.path.join(PUBMED_BASELINE_DIR, "pubmed_baseline_2025")  # The Corpus (Haystack)
BIOASQ_QA_FILE = os.path.join(PUBMED_BASELINE_DIR, "BioASQ-training12b/training12b_new.json")            # The Questions (Needles)
DB_PATH = os.path.join(PUBMED_BASELINE_DIR, "bioasq_pubmedbert_db") 
# Log file for checkpointing
PROCESSED_LOG_FILE = os.path.join(DB_PATH, "processed_files.log")

NUM_GPUS = 4
BATCH_SIZE = 256  # Batch size for Embedding generation
DB_WRITE_BATCH_SIZE = 1000  # Batch size for ChromaDB commits

# --- EMBEDDING CLASS (Per Process) ---
class GPU_PubMedBERTEmbedding:
    """
    Standalone class to load model on a specific GPU.
    """
    def __init__(self, device_id):
        self.device = f'cuda:{device_id}'
        print(f"Initializing model on {self.device}...")
        self.model = SentenceTransformer('NeuML/pubmedbert-base-embeddings', device=self.device)

    def encode(self, documents):
        # Encode locally on the specific GPU
        return self.model.encode(documents, convert_to_numpy=True, show_progress_bar=False).tolist()

# --- XML PARSER ---
def parse_pubmed_xml(filepath):
    """Yields (pmid, title, abstract) from a single gzipped XML file."""
    try:
        with gzip.open(filepath, 'rb') as f:
            context = ET.iterparse(f, events=("end",))
            for event, elem in context:
                if elem.tag == 'PubmedArticle':
                    try:
                        medline = elem.find('MedlineCitation')
                        if medline is None: continue
                        
                        pmid_elem = medline.find('PMID')
                        if pmid_elem is None: continue
                        pmid = pmid_elem.text

                        article = medline.find('Article')
                        if article is None: continue

                        title_elem = article.find('ArticleTitle')
                        title = title_elem.text if title_elem is not None else ""
                        
                        # Safe abstract extraction (handles multiple parts)
                        abstract_elem = article.find('Abstract')
                        abstract = ""
                        if abstract_elem is not None:
                            texts = [t.text for t in abstract_elem.findall('AbstractText') if t.text]
                            abstract = " ".join(texts)
                        
                        if pmid and (title or abstract):
                            yield pmid, str(title), str(abstract)
                    except Exception:
                        continue # Skip malformed articles
                    finally:
                        elem.clear() # Clear memory
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")

# --- WORKER PROCESS ---
def gpu_worker(gpu_id, file_queue, write_queue):
    """
    1. Pops a file path from file_queue.
    2. Parses XML.
    3. Batches text.
    4. Generates Embeddings on GPU.
    5. Pushes (ids, embeddings, metadatas, documents) to write_queue.
    6. Pushes a "file_done" message to write_queue.
    """
    embedder = GPU_PubMedBERTEmbedding(gpu_id)
    
    while True:
        try:
            filepath = file_queue.get(timeout=5) # Wait briefly for new files
        except queue.Empty:
            break # No more files to process

        print(f"[GPU {gpu_id}] Processing {os.path.basename(filepath)}")
        
        # Use a dictionary to handle intra-batch duplicates
        local_batch_data = {}
        
        for pmid, title, abstract in parse_pubmed_xml(filepath):
            text_blob = f"{title} [SEP] {abstract}"
            
            # Store by PMID, overwriting duplicates. Last one in file wins.
            local_batch_data[pmid] = {
                "document": text_blob,
                "metadata": {"pmid": pmid, "title": title[:200]}
            }

            # When batch is full, embed and push to writer
            if len(local_batch_data) >= BATCH_SIZE:
                # Unpack the dictionary
                ids_batch = list(local_batch_data.keys())
                docs_batch = [data["document"] for data in local_batch_data.values()]
                metas_batch = [data["metadata"] for data in local_batch_data.values()]

                # Generate embeddings
                embeddings = embedder.encode(docs_batch)
                
                write_queue.put({
                    "ids": ids_batch,
                    "embeddings": embeddings,
                    "metadatas": metas_batch,
                    "documents": docs_batch
                })
                
                # Reset for next batch
                local_batch_data = {}

        # Process remaining items in the file (end of file reached)
        if local_batch_data:
            ids_batch = list(local_batch_data.keys())
            docs_batch = [data["document"] for data in local_batch_data.values()]
            metas_batch = [data["metadata"] for data in local_batch_data.values()]

            embeddings = embedder.encode(docs_batch)
            
            write_queue.put({
                "ids": ids_batch,
                "embeddings": embeddings,
                "metadatas": metas_batch,
                "documents": docs_batch
            })
            local_batch_data = {}
        
        # --- NEW ---
        # Signal that this file is completely processed
        write_queue.put({"status": "file_done", "filepath": filepath})

    # Signal completion
    print(f"[GPU {gpu_id}] Finished.")
    write_queue.put(None) 

# --- MAIN CONTROLLER ---
def build_index_multigpu():
    """
    Main function to orchestrate the multi-GPU indexing.
    Returns the ChromaDB collection object for verification.
    """
    # 1. Setup Multiprocessing
    mp.set_start_method('spawn', force=True) # Required for CUDA in multiprocessing
    
    # --- NEW: Checkpoint System ---
    processed_files = set()
    if os.path.exists(PROCESSED_LOG_FILE):
        print(f"Loading checkpoint log: {PROCESSED_LOG_FILE}")
        try:
            with open(PROCESSED_LOG_FILE, 'r') as f:
                for line in f:
                    processed_files.add(line.strip())
        except Exception as e:
            print(f"Warning: Could not read log file. Retrying all files. Error: {e}")
            processed_files = set() # Reset on error
    print(f"Found {len(processed_files)} already processed files.")

    # 2. Collect Files
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.xml.gz")))
    if not all_files:
        print(f"No files found in {DATA_DIR}. Please check the path.")
        return None

    # Filter out processed files
    files_to_process = [f for f in all_files if f not in processed_files]
    
    if not files_to_process:
        print("All files are already processed. Running verification.")
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(name="pubmed_full")
        return collection # Return collection for verification
        
    print(f"Found {len(all_files)} total files. {len(files_to_process)} new files to process.")


    # 3. Queues
    file_queue = mp.Queue()
    write_queue = mp.Queue(maxsize=50) # Limit size to prevent RAM explosion if writer is slow
    
    # Put only the new files in the queue
    for f in files_to_process:
        file_queue.put(f)

    # 4. Start Workers
    processes = []
    for i in range(NUM_GPUS):
        p = mp.Process(target=gpu_worker, args=(i, file_queue, write_queue))
        p.start()
        processes.append(p)

    # 5. ChromaDB Consumer (Writer)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="pubmed_full")

    finished_workers = 0
    total_docs = 0
    start_time = time.time()

    print(">> Database Writer Started...")

    while finished_workers < NUM_GPUS:
        try:
            batch = write_queue.get(timeout=10)
            
            if batch is None:
                finished_workers += 1
                continue
            
            # --- NEW: Handle messages from worker ---
            if isinstance(batch, dict) and batch.get("status") == "file_done":
                filepath = batch.get("filepath")
                if filepath:
                    try:
                        with open(PROCESSED_LOG_FILE, 'a') as f:
                            f.write(f"{filepath}\n")
                    except Exception as e:
                        print(f"Warning: Could not write to log file. Error: {e}")
                continue # Go get next item from queue

            # --- This is a data batch ---
            collection.upsert(
                ids=batch['ids'],
                embeddings=batch['embeddings'],
                metadatas=batch['metadatas'],
                documents=batch['documents']
            )
            
            total_docs += len(batch['ids'])
            if total_docs % 5000 == 0:
                elapsed = time.time() - start_time
                print(f">> Total Indexed: {total_docs} | Speed: {total_docs/elapsed:.2f} docs/sec")

        except queue.Empty:
            if all(not p.is_alive() for p in processes) and finished_workers < NUM_GPUS:
                 break
            continue
        except Exception as e:
            print(f"[Writer Error] Failed to upsert batch: {e}")

    # 6. Cleanup
    for p in processes:
        p.join()
    
    print(f"Done! Indexed {total_docs} documents in {time.time() - start_time:.2f} seconds.")
    return collection

# --- VERIFICATION FUNCTION (from your draft) ---
def verify_bioasq_search(collection):
    """
    Loads real BioASQ questions and checks if our search finds the Golden Documents.
    """
    if not os.path.exists(BIOASQ_QA_FILE):
        print(f"\n[!] Warning: {BIOASQ_QA_FILE} not found.")
        print("    Cannot run verification. Please download it and update the path.")
        print("    (See README.md for download instructions)")
        return

    print("\n>> Loading BioASQ Training Data for verification...")
    try:
        with open(BIOASQ_QA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data.get('questions', [])
    except Exception as e:
        print(f"Error loading {BIOASQ_QA_FILE}: {e}")
        return

    if not questions:
        print("No questions found in BioASQ file.")
        return

    print(f">> Loaded {len(questions)} questions.")
    
    # We will test the first 5 questions
    print("\n--- RUNNING SEARCH TEST (Top 3 Questions) ---")
    
    hits = 0
    total_queries = 0
    
    for i, q in enumerate(questions[:3]): # Limit to 3 for demo
        total_queries += 1
        query_text = q['body']
        golden_urls = q.get('documents', [])
        # Extract just PMIDs from URLs (e.g., "http.../234234" -> "234234")
        golden_pmids = [url.split('/')[-1] for url in golden_urls if "pubmed" in url]
        
        if not golden_pmids:
            print(f"\nQ{i+1}: {query_text} (Skipping, no golden PMIDs)")
            continue

        print(f"\nQ{i+1}: {query_text}")
        print(f"   Golden PMIDs: {golden_pmids}")
        
        # Perform Search
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=5
            )
        except Exception as e:
            print(f"   [!] Error querying ChromaDB: {e}")
            continue
        
        # Display matches
        found_relevant = False
        if results['ids']:
            for rank, (doc_id, score) in enumerate(zip(results['ids'][0], results['distances'][0])):
                is_hit = doc_id in golden_pmids
                marker = "✅ MATCH!" if is_hit else ""
                if is_hit: 
                    found_relevant = True
                
                print(f"   Rank {rank+1}: PMID {doc_id} (Dist: {score:.4f}) {marker}")
        
        if found_relevant:
            hits += 1
        else:
            print(f"   [!] None of the top 5 were in the Golden Set for this question.")
            print(f"       (Note: This is expected if you only indexed a few files from the full dataset.)")

    print("\n--- TEST SUMMARY ---")
    if total_queries > 0:
        print(f"Found relevant results for {hits} out of {total_queries} questions.")
    else:
        print("No valid queries were run.")


if __name__ == "__main__":
    # Ensure directory exists
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    
    # 1. Build the index
    collection = build_index_multigpu()
    
    # 2. Verify the index
    if collection:
        verify_bioasq_search(collection)
    else:
        print("Index building failed. Skipping verification.")