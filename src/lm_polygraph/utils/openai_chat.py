import openai
import json
import os

from filelock import FileLock
from typing import Optional


class OpenAIChat:
    """
    Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    """

    def __init__(
        self,
        openai_access_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        cache_path: str = os.path.expanduser("~") + "/.cache",
    ):
        """
        Parameters
        ----------
        openai_access_key : str
            the token key to access OpenAI
        openai_model: str
            the model to use in OpenAI to chat.
        """
        openai.api_key = openai_access_key
        self.openai_model = openai_model

        self.cache_path = os.path.join(cache_path, 'openai_chat_cache.json')
        self.cache_lock = FileLock(self.cache_path + '.lock')
        with self.cache_lock:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            with open(self.cache_path, 'w') as f:
                json.dump({}, f)

    def ask(self, message: str) -> str:
        # check if the message is cached
        with open(self.cache_path, 'r') as f:
            openai_responses = json.load(f)
        if self.openai_model in openai_responses.keys():
            if message in openai_responses[self.openai_model]:
                return openai_responses[self.openai_model][message]

        # Ask openai
        messages = [
            {"role": "system", "content": "You are a intelligent assistant."},
            {"role": "user", "content": message},
        ]
        chat = openai.ChatCompletion.create(
            model=self.openai_model, messages=messages
        )
        reply = chat.choices[0].message.content

        # add reply to cache
        with self.cache_lock:
            with open(self.cache_path, 'r') as f:
                openai_responses = json.load(f)
            if self.openai_model not in openai_responses.keys():
                openai_responses[self.openai_model] = {}
            openai_responses[self.openai_model][message] = reply
            with open(self.cache_path, 'w') as f:
                json.dump(openai_responses, f)

        return reply
