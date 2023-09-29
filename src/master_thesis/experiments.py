import os
import gc
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import torch
from tqdm import tqdm

from master_thesis.base import BaseDataset, BaseMetric, BaseModel, BasePrompt
from master_thesis.defaults import DEFAULT_PEFT_CONFIG, DEFAULT_TRAIN_CONFIG
from master_thesis.factories import (
    DatasetFactory,
    MetricFactory,
    ModelFactory,
    PromptFactory,
)
from master_thesis.prompts import ZeroShotPrompt


class Experiment:
    _model: BaseModel
    _model_config: Dict[str, Any]
    _dataset: BaseDataset
    _dataset_config: Dict[str, Any]
    _metric: BaseMetric
    _metric_config: Dict[str, Any]

    _finetune: Optional[BaseDataset]
    _finetune_config: Dict[str, Any]
    _prompt: Optional[BasePrompt]
    _prompt_config: Dict[str, Any]

    _experiment_name: str
    _experiment_threshold: float

    def __init__(
        self,
        model: str | Type[BaseModel],
        model_config: Dict[str, Any],
        dataset: str | Type[BaseDataset],
        dataset_config: Dict[str, Any],
        metric: str | Type[BaseMetric],
        metric_config: Dict[str, Any],
        finetune: Optional[str | Type[BaseDataset]] = None,
        finetune_config: Dict[str, Any] = {
            "dataset_config": {},
            "peft_config": None,
            "train_config": None,
        },
        prompt: Optional[str | Type[BasePrompt]] = None,
        prompt_config: Dict[str, Any] = {},
        experiment_name: Optional[str] = None,
        experiment_threshold: float = 0.8,
    ) -> None:
        # Set model
        self._model = self._get_model(model, model_config)
        self._model_config = model_config

        # Set dataset
        self._dataset = self._get_dataset(dataset, dataset_config)
        self._dataset_config = dataset_config

        # Set metric
        self._metric = self._get_metric(metric, metric_config)
        self._metric_config = metric_config

        # Set finetune (optional)
        if finetune is not None:
            self._finetune = self._get_dataset(finetune, finetune_config["dataset"])
        else:
            self._finetune = finetune
        self._finetune_config = finetune_config

        # Set prompt (optional)
        if prompt is not None:
            self._prompt = self._get_prompt(prompt, self._model)
        else:
            self._prompt = prompt
        self._prompt_config = prompt_config

        # Set name
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        self._experiment_name = experiment_name or f"experiment_{date}"

        # Set threshold
        self._experiment_threshold = experiment_threshold

    def run(self) -> Dict[str, Any]:
        os.makedirs(f"../data/{self._experiment_name}", exist_ok=True)

        # Train
        if self._finetune:
            # Finetune model
            self._model = self._model.finetune(
                self._finetune,
                peft_config=self._finetune_config.get(
                    "peft_config", DEFAULT_PEFT_CONFIG
                ),
                train_config=self._finetune_config.get(
                    "train_config", DEFAULT_TRAIN_CONFIG
                ),
            )

            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()

        # Test
        test = self._dataset.test
        similarities = []
        for row in tqdm(test):
            references = row["references"]
            question = row["question"]
            answer = row["answer"]

            # Prompt model
            if not self._prompt:
                prediction = ZeroShotPrompt(self._model).run(references, question)
            else:
                prediction = self._prompt.run(
                    references,
                    question,
                    **self._prompt_config,
                )
                print(prediction)

            # Calculate score
            similarity = self._metric.test(question, answer, prediction)
            similarities.append(similarity)

            # Write
            self._write_prediction(references, question, answer, prediction, similarity)

        # Evaluate
        results = {
            "name": self._experiment_name,
            "model": self._model.__class__.__name__,
            "model_config": self._model_config,
            "finetune": self._finetune.__class__.__name__ if self._finetune else None,
            "finetune_config": self._finetune_config if self._finetune else None,
            "prompt": self._prompt.__class__.__name__ if self._prompt else None,
            "prompt_config": self._prompt_config if self._prompt else None,
            "dataset": self._dataset.__class__.__name__,
            "dataset_config": self._dataset_config,
            "metric": self._metric.__class__.__name__,
            "metric_config": self._metric_config,
            "scores": {
                "accuracy": len(
                    [s for s in similarities if s >= self._experiment_threshold]
                )
                / len(similarities),
                "avg_similarity": sum(similarities) / len(similarities),
                "min_similarity": min(similarities),
                "max_similarity": max(similarities),
                "threshold": self._experiment_threshold,
            },
            "train_size": self._finetune.train.shape[0] if self._finetune else None,
            "test_size": test.shape[0],
        }
        self._write_experiment(results)

        return results

    def _get_model(
        self,
        model: str | Type[BaseModel],
        model_config: Dict[str, Any],
    ) -> BaseModel:
        if isinstance(model, str):
            factory = ModelFactory()
            return factory.get_model(model, **model_config)

        return model(**model_config)

    def _get_dataset(
        self,
        dataset: str | Type[BaseDataset],
        dataset_config: Dict[str, Any],
    ) -> BaseDataset:
        if isinstance(dataset, str):
            factory = DatasetFactory()
            return factory.get_dataset(dataset, **dataset_config)

        return dataset(**dataset_config)

    def _get_metric(
        self,
        metric: str | Type[BaseMetric],
        metric_config: Dict[str, Any],
    ) -> BaseMetric:
        if isinstance(metric, str):
            factory = MetricFactory()
            return factory.get_metric(metric, **metric_config)

        return metric(**metric_config)

    def _get_prompt(
        self,
        prompt: str | Type[BasePrompt],
        prompt_model: BaseModel,
    ) -> BasePrompt:
        if isinstance(prompt, str):
            factory = PromptFactory()
            return factory.get_prompt(prompt, prompt_model)

        return prompt(prompt_model)

    def _write_prediction(
        self,
        references: List[str],
        question: str,
        answer: str,
        prediction: str,
        similarity: float,
    ) -> None:
        with open(
            f"../data/{self._experiment_name}/predictions.jsonl",
            "a+",
            encoding="utf-8",
        ) as handle:
            prediction = {
                "references": references,
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "similarity": similarity,
            }
            handle.write(json.dumps(prediction) + "\n")

    def _write_experiment(self, experiment: Dict[str, Any]) -> None:
        with open(
            f"../data/{self._experiment_name}/experiment.json",
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(experiment, indent=4))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}()>"
