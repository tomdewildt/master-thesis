import os
from typing import List

import torch
from langchain.llms import HuggingFacePipeline, OpenAI, VertexAI
from langchain.chat_models import ChatAnthropic, ChatOpenAI, ChatVertexAI
from langchain.schema import HumanMessage
from peft import (
    AutoPeftModelForCausalLM,
    AutoPeftModelForSeq2SeqLM,
    PeftConfig,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

from master_thesis.defaults import DEFAULT_PEFT_CONFIG, DEFAULT_TRAIN_CONFIG
from master_thesis.base import BaseModel, BaseDataset


class AnthropicChatModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> None:
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

    def finetune(self, dataset: BaseDataset) -> BaseModel:
        raise NotImplementedError("Finetuning is not supported for this model.")


class GoogleChatModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> None:
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

    def finetune(self, dataset: BaseDataset) -> BaseModel:
        raise NotImplementedError("Finetuning is not supported for this model.")


class GoogleTextModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> None:
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

    def finetune(self, dataset: BaseDataset) -> BaseModel:
        raise NotImplementedError("Finetuning is not supported for this model.")


class HuggingFaceAutoregressiveTextModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> None:
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._hf_model_id = (
            model_id.split("/")[1]
            if any([x in model_id for x in ["openai/gpt2", "google/t5"]])
            else model_id
        )

        self._config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._model = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=AutoModelForCausalLM.from_pretrained(
                    self._hf_model_id,
                    device_map="auto",
                    quantization_config=self._config,
                    use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    self._hf_model_id,
                    device_map="auto",
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

    def finetune(
        self,
        dataset: BaseDataset,
        peft_config: PeftConfig = DEFAULT_PEFT_CONFIG,
        train_config: TrainingArguments = DEFAULT_TRAIN_CONFIG,
    ) -> BaseModel:
        # Set model config
        self._model.pipeline.model.config.use_cache = False
        self._model.pipeline.tokenizer.pad_token = (
            self._model.pipeline.tokenizer.eos_token
        )
        self._model.pipeline.tokenizer.padding_side = "right"

        # Set peft config
        peft_config.task_type = TaskType.CAUSAL_LM

        # Set train config
        train_config.output_dir = (
            f"../models/{self._hf_model_id}_{dataset.dataset_id}_finetuned"
        )

        # Prepare datasets
        train_dataset = dataset.train.map(
            lambda ex: {
                "text": [" ".join(references) for references in ex["references"]]
            },
            batched=True,
        )
        test_dataset = dataset.test.map(
            lambda ex: {
                "text": [" ".join(references) for references in ex["references"]]
            },
            batched=True,
        )

        # Create trainer
        trainer = SFTTrainer(
            model=self._model.pipeline.model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self._model.pipeline.tokenizer,
            args=train_config,
            packing=False,
        )

        # Train model
        trainer.train()

        # Store model
        trainer.save_model()

        # Set model
        self._model = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model=AutoPeftModelForCausalLM.from_pretrained(
                    train_config.output_dir,
                    device_map="auto",
                    quantization_config=self._config,
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    train_config.output_dir,
                    device_map="auto",
                    padding_side="right",
                    use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                ),
                device_map="auto",
                max_length=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
            ),
            verbose=True,
        )

        return self


class HuggingFaceSeq2SeqTextModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> None:
        super().__init__(model_id, max_tokens, stop_sequences, temperature, top_p)

        self._hf_model_id = (
            model_id.split("/")[1]
            if any([x in model_id for x in ["openai/gpt2", "google/t5"]])
            else model_id
        )

        self._config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._model = HuggingFacePipeline(
            pipeline=pipeline(
                "text2text-generation",
                model=AutoModelForSeq2SeqLM.from_pretrained(
                    self._hf_model_id,
                    device_map="auto",
                    quantization_config=self._config,
                    use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    self._hf_model_id,
                    device_map="auto",
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

    def finetune(
        self,
        dataset: BaseDataset,
        peft_config: PeftConfig = DEFAULT_PEFT_CONFIG,
        train_config: TrainingArguments = DEFAULT_TRAIN_CONFIG,
    ) -> BaseModel:
        # Set model config
        self._model.pipeline.model.config.use_cache = False
        self._model.pipeline.tokenizer.pad_token = (
            self._model.pipeline.tokenizer.eos_token
        )
        self._model.pipeline.tokenizer.padding_side = "right"

        # Set peft config
        peft_config.task_type = TaskType.SEQ_2_SEQ_LM

        # Set train config
        train_config.output_dir = (
            f"../models/{self._hf_model_id}_{dataset.dataset_id}_finetuned"
        )

        # Prepare datasets
        train_dataset = dataset.train.map(
            lambda ex: {
                "text": [" ".join(references) for references in ex["references"]]
            },
            batched=True,
        )
        test_dataset = dataset.test.map(
            lambda ex: {
                "text": [" ".join(references) for references in ex["references"]]
            },
            batched=True,
        )

        # Create trainer
        trainer = SFTTrainer(
            model=self._model.pipeline.model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self._model.pipeline.tokenizer,
            args=train_config,
            packing=False,
        )

        # Train model
        trainer.train()

        # Store model
        trainer.save_model()

        # Set model
        self._model = HuggingFacePipeline(
            pipeline=pipeline(
                "text2text-generation",
                model=AutoPeftModelForSeq2SeqLM.from_pretrained(
                    train_config.output_dir,
                    device_map="auto",
                    quantization_config=self._config,
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    train_config.output_dir,
                    device_map="auto",
                    padding_side="right",
                    use_auth_token=os.getenv("HUGGINGFACE_API_KEY"),
                ),
                device_map="auto",
                max_length=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
            ),
            verbose=True,
        )

        return self


class OpenAIChatModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> None:
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

    def finetune(self, dataset: BaseDataset) -> BaseModel:
        raise NotImplementedError("Finetuning is not supported for this model.")


class OpenAITextModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        stop_sequences: List[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> None:
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

    def finetune(self, dataset: BaseDataset) -> BaseModel:
        raise NotImplementedError("Finetuning is not supported for this model.")
