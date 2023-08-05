# pylint: disable=line-too-long
from abc import ABC, abstractmethod
from typing import List


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
    ):
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._stop_sequences = stop_sequences
        self._temperature = temperature
        self._top_p = top_p

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}(model_id={self._model_id}, max_tokens={self._max_tokens}, stop_sequences={self._stop_sequences}, temperature={self._temperature}, top_p={self._top_p})>"
