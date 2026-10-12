"""Every unit test must run without opening network sockets."""

import pytest


@pytest.fixture(autouse=True)
def offline(socket_disabled):
    yield
