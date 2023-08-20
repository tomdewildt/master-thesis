import inspect
from abc import ABC, abstractmethod
from typing import List, Union

from datasets import Dataset

from master_thesis.examples import Example


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


class BaseDataset(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def train(self) -> Dataset:
        pass

    @property
    @abstractmethod
    def test(self) -> Dataset:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}()>"
