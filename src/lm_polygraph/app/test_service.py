import requests
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3001)
    args = parser.parse_args()
    url = f"http://0.0.0.0:{args.port}/get-prompt-result"
    data = {
        "prompt": "Здарова",
        "model": "Llama 2 7b",
        "openai_key": "",
        "ensembles": "",
        "seq_ue": "Mean Token Entropy",
        "temperature": 1.0,
        "topp": 1.0,
        "topk": 1,
        "num_beams": 1,
        "repetition_penalty": 5.0,
        "presence_penalty": 0.0,
    }
    response = requests.post(url, json=data)
    print("Response:", response)
    result = json.loads(response.text)
    print(result)
