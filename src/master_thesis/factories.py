from typing import List

from master_thesis.base import BaseModel
from master_thesis.models import (
    AnthropicChatModel,
    GoogleChatModel,
    GoogleTextModel,
    HuggingFaceDecTextModel,
    HuggingFaceEncDecTextModel,
    OpenAIChatModel,
    OpenAITextModel,
)


class ModelFactory:
    _MODEL_IDS = {
        "anthropic/claude-v1": AnthropicChatModel,
        "anthropic/claude-v1-100k": AnthropicChatModel,
        "anthropic/claude-instant-v1": AnthropicChatModel,
        "anthropic/claude-instant-v1-100k": AnthropicChatModel,
        "google/text-bison-001": GoogleTextModel,
        "google/chat-bison-001": GoogleChatModel,
        "openai/gpt2": HuggingFaceDecTextModel,
        "bigscience/bloom-560m": HuggingFaceDecTextModel,
        "bigscience/bloom-1b1": HuggingFaceDecTextModel,
        "bigscience/bloom-1b7": HuggingFaceDecTextModel,
        "bigscience/bloom-3b": HuggingFaceDecTextModel,
        "bigscience/bloom-7b1": HuggingFaceDecTextModel,
        "bigscience/bloom": HuggingFaceDecTextModel,
        "bigscience/bloomz-560m": HuggingFaceDecTextModel,
        "bigscience/bloomz-1b1": HuggingFaceDecTextModel,
        "bigscience/bloomz-1b7": HuggingFaceDecTextModel,
        "bigscience/bloomz-3b": HuggingFaceDecTextModel,
        "bigscience/bloomz-7b1": HuggingFaceDecTextModel,
        "bigscience/bloomz": HuggingFaceDecTextModel,
        "facebook/xglm-564M": HuggingFaceDecTextModel,
        "facebook/xglm-1.7B": HuggingFaceDecTextModel,
        "facebook/xglm-2.9B": HuggingFaceDecTextModel,
        "facebook/xglm-4.5B": HuggingFaceDecTextModel,
        "facebook/xglm-7.5B": HuggingFaceDecTextModel,
        "meta-llama/Llama-2-7b-hf": HuggingFaceDecTextModel,
        "meta-llama/Llama-2-7b-chat-hf": HuggingFaceDecTextModel,
        "meta-llama/Llama-2-13b-hf": HuggingFaceDecTextModel,
        "meta-llama/Llama-2-13b-chat-hf": HuggingFaceDecTextModel,
        "meta-llama/Llama-2-70b-hf": HuggingFaceDecTextModel,
        "meta-llama/Llama-2-70b-chat-hf": HuggingFaceDecTextModel,
        "google/t5-small": HuggingFaceEncDecTextModel,
        "google/t5-base": HuggingFaceEncDecTextModel,
        "google/t5-large": HuggingFaceEncDecTextModel,
        "google/t5-3b": HuggingFaceEncDecTextModel,
        "google/t5-11b": HuggingFaceEncDecTextModel,
        "google/flan-t5-small": HuggingFaceEncDecTextModel,
        "google/flan-t5-base": HuggingFaceEncDecTextModel,
        "google/flan-t5-large": HuggingFaceEncDecTextModel,
        "google/flan-t5-xl": HuggingFaceEncDecTextModel,
        "google/flan-t5-xxl": HuggingFaceEncDecTextModel,
        "openai/gpt-4": OpenAIChatModel,
        "openai/gpt-3.5-turbo": OpenAIChatModel,
        "openai/text-davinci-003": OpenAITextModel,
        "openai/text-curie-001": OpenAITextModel,
        "openai/text-babbage-001": OpenAITextModel,
        "openai/text-ada-001": OpenAITextModel,
    }

    def list_models(self) -> List[str]:
        return list(self._MODEL_IDS.keys())

    def get_model(self, model_id: str) -> BaseModel:
        if not self._MODEL_IDS.get(model_id):
            raise ValueError("invalid model")

        return self._MODEL_IDS[model_id](model_id)

    def __repr__(self):
        return f"<{self.__class__.__name__}()>"
