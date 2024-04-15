import openai

from typing import Optional


class OpenAIChat:
    """
    Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    """

    def __init__(
        self,
        openai_access_key: Optional[str] = None,
        openai_model: str = "gpt-4",
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

    def ask(self, message: str) -> str:
        # TODO(rediska): cache responses
        messages = [
            {"role": "system", "content": "You are a intelligent assistant."},
            {"role": "user", "content": message},
        ]
        chat = openai.ChatCompletion.create(
            model=self.openai_model, messages=messages
        )
        return chat.choices[0].message.content