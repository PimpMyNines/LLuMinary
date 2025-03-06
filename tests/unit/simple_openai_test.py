import pytest
from unittest.mock import MagicMock, patch
from src.llmhandler.models.providers.openai import OpenAILLM

def test_openai_init():
    with patch.object(OpenAILLM, "auth"):
        llm = OpenAILLM("gpt-4o")
        assert llm.model_name == "gpt-4o"

