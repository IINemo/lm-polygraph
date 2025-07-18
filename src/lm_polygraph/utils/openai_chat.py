import openai
import os
import time
import logging
import diskcache as dc


log = logging.getLogger()


class OpenAIChat:
    """
    Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o",
        base_url: str = None,
        cache_path: str = os.path.expanduser("~") + "/.cache",
        timeout: int = 600,
        max_tokens: int = None,
        rewrite_cache: bool = False,
    ):
        """
        Parameters
        ----------
        openai_model: str
            the model to use in OpenAI to chat.
        """
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is not None:
            openai.api_key = api_key
        self.openai_model = openai_model

        self.cache_path = os.path.join(cache_path, "openai_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.rewrite_cache = rewrite_cache

    def ask(self, message: str) -> str:
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        openai_responses = dc.Cache(self.cache_path, **cache_settings)

        if (self.openai_model, message) in openai_responses and not self.rewrite_cache:
            reply = openai_responses[(self.openai_model, message)]

        else:
            # Ask openai
            if openai.api_key is None:
                raise Exception(
                    "Cant ask openAI without token. "
                    "Please specify OPENAI_API_KEY in environment parameters."
                )
            messages = [
                {"role": "system", "content": "You are an intelligent assistant."},
                {"role": "user", "content": message},
            ]
            chat = self._send_request(messages)
            reply = chat.choices[0].message.content

            openai_responses[(self.openai_model, message)] = reply
            openai_responses.close()

        if "please provide" in reply.lower():
            return ""
        if "to assist you" in reply.lower():
            return ""
        if "as an ai language model" in reply.lower():
            return ""

        return reply

    def _send_request(self, messages):
        sleep_time_values = (5, 10, 30, 60, 120)
        for i in range(len(sleep_time_values)):
            try:
                return openai.OpenAI(
                    base_url=self.base_url, timeout=self.timeout
                ).chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0,  # for deterministic outputs
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                sleep_time = sleep_time_values[i]
                log.info(
                    f"Request to OpenAI failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)

        return openai.OpenAI(
            base_url=self.base_url, timeout=self.timeout
        ).chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=0,  # for deterministic outputs
            max_tokens=self.max_tokens,
        )
