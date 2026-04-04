# -*- coding: utf-8 -*-
"""Мок клиента OpenAI Chat Completions для unit-тестов без сети."""
from unittest.mock import MagicMock


def openai_client_returning(content: str, *, with_usage: bool = True):
    """Клиент, у которого `chat.completions.create` возвращает ответ с заданным текстом."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    if with_usage:
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    else:
        resp.usage = None
    client.chat.completions.create.return_value = resp
    return client
