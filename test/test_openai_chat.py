from types import SimpleNamespace
from unittest.mock import MagicMock

import diskcache_community as dc
import openai

from lm_polygraph.utils.openai_chat import OpenAIChat


def _mock_client(reply):
    client = MagicMock()
    client.return_value.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
    )
    return client


def test_openai_chat_cache_refuses_pickled_entries(tmp_path, monkeypatch):
    client = _mock_client("Paris")
    create = client.return_value.chat.completions.create
    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.setattr(openai, "api_key", "test-key")
    marker = tmp_path / "marker"
    chat = OpenAIChat(cache_path=str(tmp_path / "cache"))

    assert chat.ask("What is the capital of France?") == "Paris"
    assert chat.ask("What is the capital of France?") == "Paris"
    assert create.call_count == 1

    with dc.Cache(chat.cache_path, disk=dc.Disk) as other_writer:
        key = (chat.openai_model, "What is the capital of France?")
        other_writer.set(key, _Planted(str(marker)))

    assert chat.ask("What is the capital of France?") == "Paris"
    assert create.call_count == 2
    assert not marker.exists()

    assert chat.ask("What is the capital of France?") == "Paris"
    assert create.call_count == 2


def _write_marker(path):
    with open(path, "w") as marker:
        marker.write("ran")
    return "planted reply"


class _Planted:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (_write_marker, (self.path,))
