# -*- coding: utf-8 -*-
import pandas as pd
import pytest

from src.services import llm_counter


@pytest.fixture
def simple_catalog_df():
    return pd.DataFrame(
        {
            "title": ["Молоток стальной", "Отвёртка"],
            "category": ["Инструмент", "Инструмент"],
            "description": ["", ""],
            "price": [100.0, 50.0],
            "price_currency": ["RUB", "RUB"],
            "search_text": [
                "молоток стальной инструмент",
                "отвёртка инструмент",
            ],
        }
    )


@pytest.fixture(autouse=True)
def reset_llm_counter_callbacks():
    llm_counter.set_llm_counter_callback(None)
    llm_counter.set_llm_response_callback(None)
    yield
    llm_counter.set_llm_counter_callback(None)
    llm_counter.set_llm_response_callback(None)
