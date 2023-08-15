from typing import List, Union

from langchain.evaluation import load_evaluator

from master_thesis.base import BaseModel, BasePrompt
from master_thesis.examples import Example


class FewShotChainOfThoughtPrompt(BasePrompt):
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def run(
        self,
        references: Union[List[str], None],
        question: str,
        examples: Union[List[Example], None] = None,
    ) -> str:
        prompt = f"""
        {self._format_examples(examples)}
        {self._format_references(references)}

        Q: {question}
        A:
        """

        return self._model.generate(self._format_prompt(prompt))


class FewShotPrompt(BasePrompt):
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def run(
        self,
        references: Union[List[str], None],
        question: str,
        examples: Union[List[Example], None] = None,
    ) -> str:
        prompt = f"""
        {self._format_examples(examples)}
        {self._format_references(references)}

        Q: {question}
        A:
        """

        return self._model.generate(self._format_prompt(prompt))


class InstructionPrompt(BasePrompt):
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def run(
        self,
        references: Union[List[str], None],
        question: str,
        instruction: str = "",
    ) -> str:
        prompt = f"""
        {instruction}

        {self._format_references(references)}

        Q: {question}
        A:
        """

        return self._model.generate(self._format_prompt(prompt))


class LeastToMostPrompt(BasePrompt):
    _EXAMPLES = [
        Example(
            references=["Elsa has 5 apples. Anna has 2 more apples than Elsa."],
            question="How many apples do they have together?",
            answer="""Let's break down this problem: 1. How many apples does Anna have? 2. How many apples do Elsa and Anna have together?
                      1. Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples.
                      2. Elsa and Anna have 5 + 7 = 12 apples together.""",
        ),
    ]

    def __init__(self, model: BaseModel):
        super().__init__(model)

    def run(self, references: Union[List[str], None], question: str) -> str:
        prompt = f"""
        {self._format_examples(self._EXAMPLES)}
        {self._format_references(references)}

        Q: {question}
        A: Let's break down this problem:
        """

        return self._model.generate(self._format_prompt(prompt))


class PlanAndSolvePrompt(BasePrompt):
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def run(self, references: Union[List[str], None], question: str) -> str:
        prompt = f"""
        {self._format_references(references)}

        Q: {question}
        A: Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step.
        """

        return self._model.generate(self._format_prompt(prompt))


class SelfAskPrompt(BasePrompt):
    _EXAMPLES = [
        Example(
            references=None,
            question="""Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
                        Are Follow Up Questions Needed Here: Yes.
                        Follow Up: How old was Theodor Haecker when he died?
                        Intermediate Answer: Theodor Haecker was 65 years old when he died.
                        Follow Up: How old was Harry Vaughan Watkins when he died?
                        Intermediate Answer: Harry Vaughan Watkins was 69 years old when he died.""",
            answer="Harry Vaughan Watkins.",
        ),
    ]

    def __init__(self, model: BaseModel):
        super().__init__(model)

    def run(self, references: Union[List[str], None], question: str) -> str:
        prompt = f"""
        {self._format_examples(self._EXAMPLES)}
        {self._format_references(references)}

        Q: {question}
        A:
        """

        return self._model.generate(self._format_prompt(prompt))


class SelfConsistencyPrompt(BasePrompt):
    def __init__(
        self,
        model: BaseModel,
        eval_type: Union["embedding_distance", "string_distance"] = "string_distance",
    ):
        super().__init__(model)

        self._eval_type = eval_type

    def run(
        self,
        references: Union[List[str], None],
        question: str,
        n: int = 3,
        examples: Union[List[Example], None] = None,
    ) -> str:
        prompt = f"""
        {self._format_examples(examples)}
        {self._format_references(references)}

        Q: {question}
        A:
        """

        generations = [
            self._model.generate(self._format_prompt(prompt)) for _ in range(n)
        ]

        return self._majority_vote(generations)

    def _majority_vote(self, generations=List[str]) -> str:
        evaluator = load_evaluator(self._eval_type)

        top_similarity = 0.0
        top_generation = None

        for generation_a in generations:
            for generation_b in generations:
                similarity = evaluator.evaluate_strings(
                    prediction=generation_a,
                    reference=generation_b,
                )["score"]

                if similarity > top_similarity:
                    top_similarity = similarity
                    top_generation = generation_a

        return top_generation


class ZeroShotChainOfThoughtPrompt(BasePrompt):
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def run(self, references: Union[List[str], None], question: str) -> str:
        prompt = f"""
        {self._format_references(references)}

        Q: {question}
        A: Let's think step by step.
        """

        return self._model.generate(self._format_prompt(prompt))


class ZeroShotPrompt(BasePrompt):
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def run(self, references: Union[List[str], None], question: str) -> str:
        prompt = f"""
        {self._format_references(references)}

        Q: {question}
        A:
        """

        return self._model.generate(self._format_prompt(prompt))
