"""
Tool calling infrastructure for LLM models.
Supports optional and mandatory tool usage.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

log = logging.getLogger("lm_polygraph")


@dataclass
class ToolResponse:
    """Response from a tool execution."""
    content: str
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """
    Abstract base class for tools that can be called by LLMs.
    """
    
    def __init__(self, name: str, description: str):
        """
        Parameters:
            name (str): Name of the tool.
            description (str): Description of what the tool does.
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def __call__(self, input_text: str, **kwargs) -> ToolResponse:
        """
        Execute the tool with the given input.
        
        Parameters:
            input_text (str): Input to the tool.
            **kwargs: Additional arguments for tool execution.
        
        Returns:
            ToolResponse: Response from the tool.
        """
        raise NotImplementedError
    
    def get_description(self) -> str:
        """Get a description of the tool for the LLM."""
        return self.description
    
    def get_name(self) -> str:
        """Get the name of the tool."""
        return self.name


class BM25RetrieverTool(Tool):
    """
    BM25-based retrieval tool for document search.
    """
    
    def __init__(
        self,
        documents: List[str],
        name: str = "bm25_retriever",
        description: str = "A retrieval tool that searches through documents using BM25 algorithm. Use this tool to find relevant information from a document collection.",
        top_k: int = 5,
    ):
        """
        Parameters:
            documents (List[str]): List of documents to search through.
            name (str): Name of the tool.
            description (str): Description of the tool.
            top_k (int): Number of top documents to retrieve.
        """
        super().__init__(name, description)
        self.documents = documents
        self.top_k = top_k
        self._index = None
        self._tokenizer = None
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            
            # Download required NLTK data if not present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            from nltk.tokenize import word_tokenize
            
            # Tokenize documents
            tokenized_docs = [word_tokenize(doc.lower()) for doc in self.documents]
            
            # Initialize BM25
            self._index = BM25Okapi(tokenized_docs)
            self._tokenizer = word_tokenize
            
            log.info(f"Initialized BM25 index with {len(self.documents)} documents")
        except ImportError:
            log.error("rank_bm25 not installed. Please install it with: pip install rank-bm25")
            raise ImportError(
                "rank_bm25 is required for BM25RetrieverTool. "
                "Install it with: pip install rank-bm25"
            )
    
    def __call__(self, query: str, **kwargs) -> ToolResponse:
        """
        Retrieve top-k documents matching the query.
        
        Parameters:
            query (str): Search query.
            **kwargs: Additional arguments. Can include 'top_k' to override default.
        
        Returns:
            ToolResponse: Retrieved documents and their scores.
        """
        if self._index is None:
            raise RuntimeError("BM25 index not initialized")
        
        top_k = kwargs.get("top_k", self.top_k)
        
        # Tokenize query
        query_tokens = self._tokenizer(query.lower())
        
        # Get scores
        scores = self._index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        # Retrieve documents
        retrieved_docs = [
            {
                "document": self.documents[idx],
                "score": float(scores[idx]),
                "rank": rank + 1
            }
            for rank, idx in enumerate(top_indices)
        ]
        
        # Format response
        response_text = f"Retrieved {len(retrieved_docs)} documents:\n\n"
        for doc_info in retrieved_docs:
            response_text += f"[Document {doc_info['rank']}] (score: {doc_info['score']:.4f})\n"
            response_text += f"{doc_info['document']}\n\n"
        
        metadata = {
            "num_retrieved": len(retrieved_docs),
            "top_k": top_k,
            "documents": retrieved_docs
        }
        
        return ToolResponse(content=response_text, metadata=metadata)


class WikipediaBM25RetrieverTool(Tool):
    """
    Wikipedia BM25-based retrieval tool for searching Wikipedia using BM25 algorithm.
    Uses pyserini for efficient Wikipedia search with pre-built BM25 indices.
    """
    
    def __init__(
        self,
        name: str = "wiki_bm25_retriever",
        description: str = "A retrieval tool that searches Wikipedia using BM25 algorithm (Robertson & Zaragoza, 2009). Use this tool to find relevant information from Wikipedia articles.",
        top_k: int = 5,
        index_name: str = "wikipedia-dpr-100w",
        cache_dir: Optional[str] = None,
    ):
        """
        Parameters:
            name (str): Name of the tool.
            description (str): Description of the tool.
            top_k (int): Number of top documents to retrieve.
            index_name (str): Name of the pyserini index to use. 
                Default: "wikipedia-dpr-100w" (uses BM25 with k1=0.9, b=0.4, tuned for QA tasks).
            cache_dir (str, optional): Cache directory for pyserini indices. 
                If None, uses PYSERINI_CACHE env var or default ~/.cache/pyserini
        """
        super().__init__(name, description)
        self.top_k = top_k
        self.index_name = index_name
        self.cache_dir = cache_dir
        self._searcher = None
        self._initialize_searcher()
    
    def _initialize_searcher(self):
        """Initialize the pyserini searcher for Wikipedia.
        
        Uses wikipedia-dpr-100w index which comes with BM25 parameters k1=0.9, b=0.4
        tuned specifically for standard QA tasks.
        """
        try:
            import os
            from pathlib import Path
            
            # Set cache directory for pyserini
            # IMPORTANT: Set environment variable BEFORE importing pyserini modules
            # Pyserini stores indices in PYSERINI_CACHE/indexes/ by default
            if self.cache_dir is not None:
                cache_path = Path(self.cache_dir)
                # Create cache directory if it doesn't exist
                cache_path.mkdir(parents=True, exist_ok=True)
                # Create indexes subdirectory that pyserini uses
                (cache_path / "indexes").mkdir(parents=True, exist_ok=True)
                # Set environment variable for pyserini (must be set before importing)
                os.environ["PYSERINI_CACHE"] = str(cache_path)
                log.info(f"Set pyserini cache directory to: {cache_path} (indices will be in {cache_path}/indexes/)")
            elif "PYSERINI_CACHE" not in os.environ:
                # Use default cache location if not set
                default_cache = Path.home() / ".cache" / "pyserini"
                default_cache.mkdir(parents=True, exist_ok=True)
                (default_cache / "indexes").mkdir(parents=True, exist_ok=True)
                os.environ["PYSERINI_CACHE"] = str(default_cache)
                log.info(f"Using default pyserini cache directory: {default_cache}")
            
            # Now import pyserini after setting environment variable
            from pyserini.search.lucene import LuceneSearcher
            
            # Initialize searcher with Wikipedia index
            # wikipedia-dpr-100w uses BM25 with k1=0.9, b=0.4 by default, tuned for QA tasks
            self._searcher = LuceneSearcher.from_prebuilt_index(self.index_name)
            
            # Set BM25 parameters (k1=0.9, b=0.4) - these are the defaults for wikipedia-dpr-100w
            # but we set them explicitly to ensure consistency
            self._searcher.set_bm25(k1=0.9, b=0.4)
            
            log.info(f"Initialized Wikipedia BM25 searcher with index: {self.index_name} (BM25: k1=0.9, b=0.4)")
        except ImportError as e:
            log.error("pyserini not installed. Please install it with: pip install pyserini")
            raise ImportError(
                "pyserini is required for WikipediaBM25RetrieverTool. "
                "Install it with: pip install pyserini"
            )
        except Exception as e:
            log.error(f"Failed to initialize Wikipedia BM25 searcher: {e}")
            raise ImportError(
                f"Failed to initialize Wikipedia BM25 searcher with index '{self.index_name}'. "
                f"Please ensure pyserini is installed and the index is available. "
                f"Install with: pip install pyserini"
            )
    
    def __call__(self, query: str, **kwargs) -> ToolResponse:
        """
        Search Wikipedia using BM25 and retrieve top-k documents.
        
        Parameters:
            query (str): Search query.
            **kwargs: Additional arguments. Can include 'top_k' to override default.
        
        Returns:
            ToolResponse: Retrieved Wikipedia passages and their scores.
        """
        if self._searcher is None:
            raise RuntimeError("Wikipedia BM25 searcher not initialized")
        
        top_k = kwargs.get("top_k", self.top_k)
        
        try:
            # Search using BM25
            hits = self._searcher.search(query, k=top_k)
            
            # Retrieve documents
            retrieved_docs = []
            for i, hit in enumerate(hits):
                doc_id = hit.docid
                try:
                    doc = self._searcher.doc(doc_id)
                    
                    # Extract content (format may vary by index)
                    content = ""
                    title = ""
                    
                    # Try different methods to get document content
                    if hasattr(doc, 'raw'):
                        content = doc.raw()
                    elif hasattr(doc, 'contents'):
                        content = doc.contents()
                    elif hasattr(doc, 'get'):
                        content = doc.get('contents', '')
                        title = doc.get('title', '')
                    elif isinstance(doc, dict):
                        content = doc.get('contents', doc.get('text', ''))
                        title = doc.get('title', '')
                    else:
                        # Try to get as string representation
                        content = str(doc)
                    
                    # Extract title if not already extracted
                    if not title:
                        if hasattr(doc, 'get') and isinstance(doc, dict):
                            title = doc.get('title', '')
                        elif hasattr(doc, 'title'):
                            try:
                                title = doc.title()
                            except:
                                title = ""
                    
                    # If content is still empty, try accessing by index
                    if not content:
                        try:
                            # Some pyserini indices store content differently
                            if hasattr(doc, 'lucene_document'):
                                lucene_doc = doc.lucene_document()
                                if lucene_doc:
                                    content = lucene_doc.get('contents') or lucene_doc.get('raw')
                        except:
                            pass
                    
                    # Fallback: use doc_id if content is empty
                    if not content:
                        content = f"Document {doc_id}"
                        log.warning(f"Could not extract content for document {doc_id}")
                    
                    retrieved_docs.append({
                        "document": content,
                        "title": title if title else f"Wikipedia Passage {i+1}",
                        "score": float(hit.score),
                        "rank": i + 1,
                        "doc_id": doc_id
                    })
                except Exception as e:
                    log.warning(f"Error retrieving document {doc_id}: {e}")
                    # Continue with other documents
                    continue
            
            # Format response
            response_text = f"Retrieved {len(retrieved_docs)} Wikipedia passages:\n\n"
            for doc_info in retrieved_docs:
                response_text += f"[{doc_info['title']}] (score: {doc_info['score']:.4f})\n"
                response_text += f"{doc_info['document']}\n\n"
            
            metadata = {
                "num_retrieved": len(retrieved_docs),
                "top_k": top_k,
                "documents": retrieved_docs,
                "index_name": self.index_name
            }
            
            return ToolResponse(content=response_text, metadata=metadata)
            
        except Exception as e:
            log.error(f"Error during Wikipedia BM25 search: {e}")
            return ToolResponse(
                content=f"Error searching Wikipedia: {str(e)}",
                metadata={"error": str(e)}
            )


class ToolManager:
    """
    Manages tools available to the LLM.
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None):
        """
        Parameters:
            tools (Optional[List[Tool]]): List of tools to manage.
        """
        self.tools = tools or []
        self.tool_dict = {tool.get_name(): tool for tool in self.tools}
    
    def add_tool(self, tool: Tool):
        """Add a tool to the manager."""
        self.tools.append(tool)
        self.tool_dict[tool.get_name()] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tool_dict.get(name)
    
    def get_tool_descriptions(self) -> str:
        """Get descriptions of all tools for the LLM."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.get_name()}: {tool.get_description()}")
        
        return "\n".join(descriptions)
    
    def has_tools(self) -> bool:
        """Check if any tools are available."""
        return len(self.tools) > 0
    
    def __bool__(self) -> bool:
        """Allow ToolManager to be used in boolean context."""
        return self.has_tools()

