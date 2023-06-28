import requests
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5239)
    args = parser.parse_args()
    url = f'http://localhost:{args.port}/chat/completions'
    data = {
        'model': 'Bloomz 560M',
        'ue': 'Maximum Probability, token-level',
        'messages': [{'content': 'Здарова'}],
    }
    response = requests.post(url, json=data)
    print('Response:', response)
    result = json.loads(response.text)
    print(result)
