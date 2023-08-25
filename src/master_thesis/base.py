import inspect
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from datasets import Dataset

from master_thesis.examples import Example


class BaseDataset(ABC):
    _dataset_id: str
    _train_limit: Optional[int]
    _test_limit: Optional[int]

    def __init__(
        self,
        dataset_id: str,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> None:
        self._dataset_id = dataset_id
        self._train_limit = train_limit
        self._test_limit = test_limit

    @property
    @abstractmethod
    def train(self) -> Dataset:
        pass

    @property
    @abstractmethod
    def test(self) -> Dataset:
        pass

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}()>"


class BaseModel(ABC):
    _model_id: str
    _max_tokens: int
    _stop_sequences: List[str]
    _temperature: float
    _top_p: float

    def __init__(
        self,
        model_id: str,
        max_tokens: int,
        stop_sequences: List[str],
        temperature: float,
        top_p: float,
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._stop_sequences = stop_sequences
        self._temperature = temperature
        self._top_p = top_p

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def finetune(self, dataset: BaseDataset) -> "BaseModel":
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model_id={self._model_id}, max_tokens={self._max_tokens}, stop_sequences={self._stop_sequences}, temperature={self._temperature}, top_p={self._top_p})>"


class BasePrompt(ABC):
    _model: BaseModel

    def __init__(self, model: BaseModel) -> None:
        self._model = model

    @abstractmethod
    def run(self, references: List[str], question: str) -> str:
        raise NotImplementedError

    def _format_references(self, references: Union[List[str], None]) -> str:
        if not references:
            return ""

        return "\n".join(references)

    def _format_examples(self, examples: Union[List[Example], None]) -> str:
        if not examples:
            return ""

        prompts = [
            self._format_prompt(
                f"""
                {self._format_references(example.references)}
                
                Q: {example.question}
                A: {example.answer}
                ---
                """
            )
            for example in examples
        ]

        return "\n".join(prompts)

    def _format_prompt(self, text: str) -> str:
        return inspect.cleandoc(text).replace("  ", "").strip()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model={self._model})>"
