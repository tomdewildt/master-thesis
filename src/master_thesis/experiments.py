import os
import json
from datetime import datetime
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Type

from tqdm.autonotebook import tqdm

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

    _finetune_dataset: Optional[BaseDataset]
    _finetune_format_function: Callable[[List[str], str, str], str]
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
        finetune_dataset: Optional[str | Type[BaseDataset]] = None,
        finetune_format_function: Optional[
            Callable[
                [List[str], str, str],
                str,
            ]
        ] = lambda r, q, a: "\n".join(r),
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
        if finetune_dataset is not None:
            self._finetune_dataset = self._get_dataset(
                finetune_dataset,
                finetune_config["dataset_config"],
            )
            self._finetune_format_function = finetune_format_function
        else:
            self._finetune_dataset = finetune_dataset
            self._finetune_format_function = finetune_format_function
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
        train_start_time = timer()
        if self._finetune_dataset:
            # Finetune model
            self._model = self._model.finetune(
                self._finetune_dataset,
                format_function=self._finetune_format_function,
                peft_config=self._finetune_config.get(
                    "peft_config", DEFAULT_PEFT_CONFIG
                ),
                train_config=self._finetune_config.get(
                    "train_config", DEFAULT_TRAIN_CONFIG
                ),
            )
        train_end_time = timer()

        # Test
        test_start_time = timer()
        similarities = []
        inference_durations = []
        metric_durations = []
        for row in tqdm(self._dataset.test, desc="Evaluate"):
            references = row["references"]
            question = row["question"]
            answer = row["answer"]

            # Prompt model
            inference_start_time = timer()
            if not self._prompt:
                prediction = ZeroShotPrompt(self._model).run(references, question)
            else:
                prediction = self._prompt.run(
                    references,
                    question,
                    **self._prompt_config,
                )
            inference_end_time = timer()

            # Calculate similarity
            metric_start_time = timer()
            similarity = self._metric.test(question, answer, prediction)
            metric_end_time = timer()

            # Append results
            similarities.append(similarity)
            inference_durations.append(inference_end_time - inference_start_time)
            metric_durations.append(metric_end_time - metric_start_time)

            # Save results
            self._write_prediction(
                references,
                question,
                answer,
                prediction,
                similarity,
                inference_end_time - inference_start_time,
                metric_end_time - metric_start_time,
            )
        test_end_time = timer()

        # Save results
        results = {
            "name": self._experiment_name,
            "model": self._model.__class__.__name__,
            "model_config": self._model_config,
            "finetune_dataset": (
                self._finetune_dataset.__class__.__name__
                if self._finetune_dataset
                else None
            ),
            "finetune_config": (
                {
                    "dataset": self._finetune_config["dataset_config"],
                    "peft_config": self._finetune_config["peft_config"].to_dict(),
                    "train_config": self._finetune_config["train_config"].to_dict(),
                }
                if self._finetune_dataset
                else None
            ),
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
            "duration": {
                "train_duration": round(train_end_time - train_start_time),
                "test_duration": round(test_end_time - test_start_time),
                "avg_inference_duration": round(
                    sum(inference_durations) / len(inference_durations)
                ),
                "min_inference_duration": round(min(inference_durations)),
                "max_inference_duration": round(max(inference_durations)),
                "avg_metric_duration": round(
                    sum(metric_durations) / len(metric_durations)
                ),
                "min_metric_duration": round(min(metric_durations)),
                "max_metric_duration": round(max(metric_durations)),
            },
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
        inference_duration: float,
        metric_duration: float,
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
                "inference_duration": round(inference_duration),
                "metric_duration": round(metric_duration),
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
