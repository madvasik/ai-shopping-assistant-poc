# -*- coding: utf-8 -*-
"""
Тесты `src.services.knowledge_base`: форматирование, язык, CatalogKB,
поиск по каталогу, ответы с LLM и без.
"""
import builtins
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.services.knowledge_base as knowledge_base

from src.services.knowledge_base import (
    CatalogKB,
    KBAnswer,
    _detect_lang,
    _fmt_price,
    _get_openai_client as kb_get_client,
    _slim_row,
)


# ---------------------------------------------------------------------------
# OpenAI client factory
# ---------------------------------------------------------------------------


def test_kb_get_openai_client_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c, err = kb_get_client()
    assert c is None and err and "OPENAI_API_KEY" in err


def test_kb_get_openai_client_openai_init_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    with patch("openai.OpenAI", side_effect=RuntimeError("init fail")):
        c, err = kb_get_client()
    assert c is None and "init fail" in err


def test_kb_get_openai_client_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    fake = MagicMock()
    with patch("openai.OpenAI", return_value=fake):
        c, err = kb_get_client()
    assert c is fake and err is None


def test_kb_get_openai_import_fails(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai" or (fromlist and "OpenAI" in fromlist):
            raise ImportError("blocked openai")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    c, err = kb_get_client()
    assert c is None and "openai" in err.lower()


# ---------------------------------------------------------------------------
# Helpers and CatalogKB construction
# ---------------------------------------------------------------------------


def test_fmt_price():
    assert _fmt_price(float("nan"), "RUB") == "-"
    assert "1" in _fmt_price(1000, "RUB") and "RUB" in _fmt_price(1000, "RUB")


def test_fmt_price_non_numeric():
    assert "x" in _fmt_price("x", "RUB").lower() or _fmt_price("x", "RUB").startswith("x")


def test_slim_row():
    row = pd.Series(
        {
            "title": "T",
            "category": "C",
            "description": "D",
            "price": 10,
            "price_currency": "RUB",
            "search_text": "s",
            "extra": 1,
        }
    )
    d = _slim_row(row)
    assert "extra" not in d
    assert d["title"] == "T"


def test_detect_lang():
    assert _detect_lang("Привет") == "ru"
    assert _detect_lang("Hello") == "en"


def test_catalog_kb_raises_when_retriever_unavailable(monkeypatch, simple_catalog_df):
    monkeypatch.setattr(knowledge_base, "Retriever", None)
    monkeypatch.setattr(knowledge_base, "_retriever_import_err", None)
    with pytest.raises(ImportError, match="Retriever"):
        CatalogKB(simple_catalog_df)


def test_catalog_kb_adds_missing_columns():
    df = pd.DataFrame({"title": ["Только название"]})
    kb = CatalogKB(df)
    for c in (
        "category",
        "description",
        "price",
        "price_currency",
        "search_text",
    ):
        assert c in kb.df.columns


def test_catalog_kb_retrieve(simple_catalog_df):
    kb = CatalogKB(simple_catalog_df)
    out = kb.retrieve("hammer", k=5)
    assert not out.empty


def test_catalog_kb_context_from_items(simple_catalog_df):
    kb = CatalogKB(simple_catalog_df)
    items = kb.retrieve("tool", k=2)
    ctx = kb._context_from_items(items, limit=2)
    assert "title" in ctx


def test_llm_answer_no_client(monkeypatch, simple_catalog_df):
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (None, "x"),
    )
    kb = CatalogKB(simple_catalog_df)
    assert kb._llm_answer("q", "", "ru") is None


def test_llm_answer_success(monkeypatch, simple_catalog_df):
    c = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Краткий ответ."))]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    c.chat.completions.create.return_value = resp
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (c, None),
    )
    kb = CatalogKB(simple_catalog_df)
    assert kb._llm_answer("вопрос", "ctx", "ru") == "Краткий ответ."


def test_llm_answer_exception(monkeypatch, simple_catalog_df):
    c = MagicMock()
    c.chat.completions.create.side_effect = RuntimeError("x")
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (c, None),
    )
    kb = CatalogKB(simple_catalog_df)
    assert kb._llm_answer("q", "", "ru") is None


def test_answer_consultation_fallback(monkeypatch, simple_catalog_df):
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (None, "x"),
    )
    kb = CatalogKB(simple_catalog_df)
    text = kb.answer_consultation("что такое грунтовка")
    assert "не удалось" in text.lower() or "переформулировать" in text.lower()


def test_answer_consultation_with_llm(monkeypatch, simple_catalog_df):
    c = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Грунтовка — это..."))]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    c.chat.completions.create.return_value = resp
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (c, None),
    )
    kb = CatalogKB(simple_catalog_df)
    assert "Грунтовка" in kb.answer_consultation("что такое грунтовка")


def test_answer_with_llm_and_relevant_items(monkeypatch, simple_catalog_df):
    c = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Совет по инструменту."))]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    c.chat.completions.create.return_value = resp
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (c, None),
    )
    kb = CatalogKB(simple_catalog_df)
    ans = kb.answer("молоток стальной инструмент", top_k=10, recommend_k=3)
    assert isinstance(ans, KBAnswer)
    assert "Совет" in ans.answer
    assert not ans.items.empty


def test_answer_llm_ok_but_no_relevant_scores(monkeypatch):
    df = pd.DataFrame(
        {
            "title": ["Другой товар"],
            "category": ["x"],
            "description": [""],
            "price": [1.0],
            "price_currency": ["RUB"],
            "search_text": ["совсем другая лексика без совпадений"],
        }
    )
    c = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Только текст."))]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    c.chat.completions.create.return_value = resp
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (c, None),
    )
    kb = CatalogKB(df)
    ans = kb.answer("xyznonexistentquery12345", top_k=5)
    assert "Только текст" in ans.answer
    assert ans.items.empty


def test_answer_fallback_catalog_only(monkeypatch):
    df = pd.DataFrame(
        {
            "title": ["Пила"],
            "category": ["Инструмент"],
            "description": ["Острая"],
            "price": [900.0],
            "price_currency": ["RUB"],
            "search_text": ["пила инструмент острая"],
        }
    )
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (None, "x"),
    )
    kb = CatalogKB(df)
    ans = kb.answer("пила", top_k=5, recommend_k=1)
    assert "Пила" in ans.answer or "пила" in ans.answer.lower()
    assert not ans.items.empty


def test_answer_no_llm_no_relevant_items(monkeypatch):
    df = pd.DataFrame(
        {
            "title": ["Другой товар"],
            "category": ["x"],
            "description": [""],
            "price": [1.0],
            "price_currency": ["RUB"],
            "search_text": ["совсем другая лексика без совпадений"],
        }
    )
    monkeypatch.setattr(
        "src.services.knowledge_base._get_openai_client",
        lambda: (None, "x"),
    )
    kb = CatalogKB(df)
    ans = kb.answer("xyznonexistentquery12345", top_k=5)
    assert "не удалось найти" in ans.answer.lower()
    assert ans.items.empty
