import os
from typing import List

from langchain.llms import HuggingFacePipeline, OpenAI, VertexAI
from langchain.chat_models import ChatAnthropic, ChatOpenAI, ChatVertexAI
from langchain.schema import HumanMessage
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from master_thesis.base import BaseModel


class AnthropicChatModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ):
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._anthropic_model_id = model_id.split("/")[1]

        self._model = ChatAnthropic(
            model=self._anthropic_model_id,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            top_p=top_p,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            verbose=True,
        )

    def generate(self, prompt: str) -> str:
        output = self._model([HumanMessage(content=prompt)], stop=self._stop_sequences)

        return output.content


class GoogleChatModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ):
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._google_model_id = model_id.split("/")[1]

        self._model = ChatVertexAI(
            model=self._google_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            credentials=None,
            verbose=True,
        )

    def generate(self, prompt: str) -> str:
        output = self._model([HumanMessage(content=prompt)], stop=self._stop_sequences)

        return output.content


class GoogleTextModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ):
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._google_model_id = model_id.split("/")[1]

        self._model = VertexAI(
            model=self._google_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            credentials=None,
            verbose=True,
        )

    def generate(self, prompt: str) -> str:
        output = self._model(prompt, stop=self._stop_sequences)

        return output


class HuggingFaceDecTextModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ):
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._hf_model_id = (
            model_id.split("/")[1]
            if any([x in model_id for x in ["openai/gpt2", "google/t5"]])
            else model_id
        )

        self._model = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=AutoModelForCausalLM.from_pretrained(
                    self._hf_model_id,
                    device_map="auto",
                    torch_dtype="auto",
                    use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    self._hf_model_id,
                    device_map="auto",
                    torch_dtype="auto",
                    use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                ),
                device_map="auto",
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
            verbose=True,
        )

    def generate(self, prompt: str) -> str:
        output = self._model(prompt, stop=self._stop_sequences)

        return output


class HuggingFaceEncDecTextModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ):
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._hf_model_id = (
            model_id.split("/")[1]
            if any([x in model_id for x in ["openai/gpt2", "google/t5"]])
            else model_id
        )

        self._model = HuggingFacePipeline(
            pipeline=pipeline(
                "text2text-generation",
                model=AutoModelForSeq2SeqLM.from_pretrained(
                    self._hf_model_id,
                    device_map="auto",
                    torch_dtype="auto",
                    use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    self._hf_model_id,
                    device_map="auto",
                    torch_dtype="auto",
                    use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                ),
                device_map="auto",
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
            verbose=True,
        )

    def generate(self, prompt: str) -> str:
        output = self._model(prompt, stop=self._stop_sequences)

        return output


class OpenAIChatModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ):
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._openai_model_id = model_id.split("/")[1]

        self._model = ChatOpenAI(
            model=self._openai_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            model_kwargs={"top_p": top_p},
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            verbose=True,
        )

    def generate(self, prompt: str) -> str:
        output = self._model([HumanMessage(content=prompt)], stop=self._stop_sequences)

        return output.content


class OpenAITextModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ):
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._openai_model_id = model_id.split("/")[1]

        self._model = OpenAI(
            model=self._openai_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            verbose=True,
        )

    def generate(self, prompt: str) -> str:
        output = self._model(prompt, stop=self._stop_sequences)

        return output
