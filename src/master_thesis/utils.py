import inspect
from typing import List, Optional

from master_thesis.examples import Example


def format_references(references: Optional[List[str]]) -> str:
    if not references:
        return ""

    return "\n".join(references)


def format_examples(examples: Optional[List[Example]]) -> str:
    if not examples:
        return ""

    prompts = [
        format_prompt(
            f"""
            {format_references(example.references)}
            
            Q: {example.question}
            A: {example.answer}
            ---
            """
        )
        for example in examples
    ]

    return "\n".join(prompts)


def format_prompt(text: str) -> str:
    return inspect.cleandoc(text).replace("  ", "").strip()
