from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from datasets import Dataset

from master_thesis.examples import Example
from master_thesis.utils import format_examples, format_prompt, format_references


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


class BaseMetric(ABC):
    @abstractmethod
    def test(self, question: str, answer: str, prediction: str) -> float:
        pass

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
    def finetune(
        self,
        dataset: BaseDataset,
        format_function: Callable[[List[str], str, str], str] = lambda r, q, a: r,
    ) -> "BaseModel":
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model_id={self._model_id}, max_tokens={self._max_tokens}, stop_sequences={self._stop_sequences}, temperature={self._temperature}, top_p={self._top_p})>"


class BasePrompt(ABC):
    _model: BaseModel

    def __init__(self, model: BaseModel) -> None:
        self._model = model

    @abstractmethod
    def run(self, references: List[str], question: str) -> str:
        pass

    def _format_references(self, references: Optional[List[str]]) -> str:
        return format_references(references)

    def _format_examples(self, examples: Optional[List[Example]]) -> str:
        return format_examples(examples)

    def _format_prompt(self, text: str) -> str:
        return format_prompt(text)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model={self._model})>"
