import os
from typing import List, NamedTuple, Optional


class Example(NamedTuple):
    references: Optional[List[str]]
    question: str
    answer: str

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(context={'[...]' if self.references else None}, question={self.question}, answer={self.answer})>"

    def __str__(self) -> str:
        return f"{os.linesep.join(self.references) if self.references else ''}Q: {self.question}\nA: {self.answer}"
