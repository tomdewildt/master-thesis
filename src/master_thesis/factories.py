from typing import List, Optional

from master_thesis.base import BaseModel, BasePrompt, BaseDataset
from master_thesis.models import (
    AnthropicChatModel,
    GoogleChatModel,
    GoogleTextModel,
    HuggingFaceAutoregressiveTextModel,
    HuggingFaceSeq2SeqTextModel,
    OpenAIChatModel,
    OpenAITextModel,
)
from master_thesis.prompts import (
    FewShotChainOfThoughtPrompt,
    FewShotPrompt,
    InstructionPrompt,
    LeastToMostPrompt,
    PlanAndSolvePrompt,
    SelfAskPrompt,
    SelfConsistencyPrompt,
    ZeroShotChainOfThoughtPrompt,
    ZeroShotPrompt,
)
from master_thesis.datasets import (
    COQADataset,
    HotpotQADataset,
    SQUADv1Dataset,
    SQUADv2Dataset,
    TriviaQADataset,
)


class ModelFactory:
    _MODEL_IDS = {
        "anthropic/claude-v1": AnthropicChatModel,
        "anthropic/claude-v1-100k": AnthropicChatModel,
        "anthropic/claude-instant-v1": AnthropicChatModel,
        "anthropic/claude-instant-v1-100k": AnthropicChatModel,
        "google/text-bison-001": GoogleTextModel,
        "google/chat-bison-001": GoogleChatModel,
        "openai/gpt2": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloom-560m": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloom-1b1": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloom-1b7": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloom-3b": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloom-7b1": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloom": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloomz-560m": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloomz-1b1": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloomz-1b7": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloomz-3b": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloomz-7b1": HuggingFaceAutoregressiveTextModel,
        "bigscience/bloomz": HuggingFaceAutoregressiveTextModel,
        "facebook/xglm-564M": HuggingFaceAutoregressiveTextModel,
        "facebook/xglm-1.7B": HuggingFaceAutoregressiveTextModel,
        "facebook/xglm-2.9B": HuggingFaceAutoregressiveTextModel,
        "facebook/xglm-4.5B": HuggingFaceAutoregressiveTextModel,
        "facebook/xglm-7.5B": HuggingFaceAutoregressiveTextModel,
        "meta-llama/Llama-2-7b-hf": HuggingFaceAutoregressiveTextModel,
        "meta-llama/Llama-2-7b-chat-hf": HuggingFaceAutoregressiveTextModel,
        "meta-llama/Llama-2-13b-hf": HuggingFaceAutoregressiveTextModel,
        "meta-llama/Llama-2-13b-chat-hf": HuggingFaceAutoregressiveTextModel,
        "meta-llama/Llama-2-70b-hf": HuggingFaceAutoregressiveTextModel,
        "meta-llama/Llama-2-70b-chat-hf": HuggingFaceAutoregressiveTextModel,
        "google/t5-small": HuggingFaceSeq2SeqTextModel,
        "google/t5-base": HuggingFaceSeq2SeqTextModel,
        "google/t5-large": HuggingFaceSeq2SeqTextModel,
        "google/t5-3b": HuggingFaceSeq2SeqTextModel,
        "google/t5-11b": HuggingFaceSeq2SeqTextModel,
        "google/flan-t5-small": HuggingFaceSeq2SeqTextModel,
        "google/flan-t5-base": HuggingFaceSeq2SeqTextModel,
        "google/flan-t5-large": HuggingFaceSeq2SeqTextModel,
        "google/flan-t5-xl": HuggingFaceSeq2SeqTextModel,
        "google/flan-t5-xxl": HuggingFaceSeq2SeqTextModel,
        "openai/gpt-4": OpenAIChatModel,
        "openai/gpt-3.5-turbo": OpenAIChatModel,
        "openai/text-davinci-003": OpenAITextModel,
        "openai/text-curie-001": OpenAITextModel,
        "openai/text-babbage-001": OpenAITextModel,
        "openai/text-ada-001": OpenAITextModel,
    }

    def list_models(self) -> List[str]:
        return list(self._MODEL_IDS.keys())

    def get_model(
        self,
        model_id: str,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> BaseModel:
        if not self._MODEL_IDS.get(model_id):
            raise ValueError("invalid model")

        return self._MODEL_IDS[model_id](
            model_id,
            max_tokens,
            stop_sequences,
            temperature,
            top_p,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}()>"


class PromptFactory:
    _PROMPT_IDS = {
        "few-shot-cot": FewShotChainOfThoughtPrompt,
        "few-shot": FewShotPrompt,
        "instruction": InstructionPrompt,
        "least-to-most": LeastToMostPrompt,
        "plan-and-solve": PlanAndSolvePrompt,
        "self-ask": SelfAskPrompt,
        "self-consistency": SelfConsistencyPrompt,
        "zero-shot-cot": ZeroShotChainOfThoughtPrompt,
        "zero-shot": ZeroShotPrompt,
    }

    def list_prompts(self) -> List[str]:
        return list(self._PROMPT_IDS.keys())

    def get_prompt(self, prompt_id: str, model: BaseModel) -> BasePrompt:
        if not self._PROMPT_IDS.get(prompt_id):
            raise ValueError("invalid prompt")

        return self._PROMPT_IDS[prompt_id](model)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}()>"


class DatasetFactory:
    _DATASET_IDS = {
        "coqa": COQADataset,
        "hotpotqa": HotpotQADataset,
        "squad-v1": SQUADv1Dataset,
        "squad-v2": SQUADv2Dataset,
        "triviaqa": TriviaQADataset,
    }

    def list_datasets(self) -> List[str]:
        return list(self._DATASET_IDS.keys())

    def get_dataset(
        self,
        dataset_id: str,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> BaseDataset:
        if not self._DATASET_IDS.get(dataset_id):
            raise ValueError("invalid dataset")

        return self._DATASET_IDS[dataset_id](
            train_limit=train_limit,
            test_limit=test_limit,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}()>"
