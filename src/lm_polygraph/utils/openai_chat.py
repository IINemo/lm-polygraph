import openai
import json
import os

from filelock import FileLock


class OpenAIChat:
    """
    Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    """

    def __init__(
        self,
        openai_model: str = "gpt-4",
        cache_path: str = os.path.expanduser("~") + "/.cache",
    ):
        """
        Parameters
        ----------
        openai_model: str
            the model to use in OpenAI to chat.
        """
        api_key = os.environ.get("OPENAI_KEY", None)
        if api_key is not None:
            openai.api_key = api_key
        self.openai_model = openai_model

        self.cache_path = os.path.join(cache_path, "openai_chat_cache.json")
        self.cache_lock = FileLock(self.cache_path + ".lock")
        with self.cache_lock:
            if not os.path.exists(self.cache_path):
                if not os.path.exists(cache_path):
                    os.makedirs(cache_path)
                with open(self.cache_path, "w") as f:
                    json.dump({}, f)

    def ask(self, message: str) -> str:
        # check if the message is cached
        with open(self.cache_path, "r") as f:
            openai_responses = json.load(f)

        if message in openai_responses.get(self.openai_model, {}).keys():
            reply = openai_responses[self.openai_model][message]
        else:
            # Ask openai

            if openai.api_key is None:
                raise Exception(
                    "Cant ask openAI without token. "
                    "Please specify OPENAI_KEY in environment parameters."
                )

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
                with open(self.cache_path, "r") as f:
                    openai_responses = json.load(f)
                if self.openai_model not in openai_responses.keys():
                    openai_responses[self.openai_model] = {}
                openai_responses[self.openai_model][message] = reply
                with open(self.cache_path, "w") as f:
                    json.dump(openai_responses, f)

        if "please provide" in reply.lower():
            return ""
        if "to assist you" in reply.lower():
            return ""
        if "as an ai language model" in reply.lower():
            return ""

        return reply
