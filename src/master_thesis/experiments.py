from typing import Any, Dict, Optional, Type

from master_thesis.base import BaseDataset, BaseMetric, BaseModel, BasePrompt
from master_thesis.factories import (
    DatasetFactory,
    MetricFactory,
    ModelFactory,
    PromptFactory,
)


class Experiment:
    _model: BaseModel
    _model_config: Dict[str, Any]
    _dataset: BaseDataset
    _dataset_config: Dict[str, Any]
    _metric: BaseMetric
    _metric_config: Dict[str, Any]

    _finetune: Optional[BaseDataset]
    _finetune_config: Optional[Dict[str, Any]]
    _prompt: Optional[BasePrompt]
    _prompt_config: Optional[Dict[str, Any]]

    def __init__(
        self,
        model: str | Type[BaseModel],
        model_config: Dict[str, Any],
        dataset: str | Type[BaseDataset],
        dataset_config: Dict[str, Any],
        metric: str | Type[BaseMetric],
        metric_config: Dict[str, Any],
        finetune: Optional[str | Type[BaseDataset]] = None,
        finetune_config: Optional[Dict[str, Any]] = None,
        prompt: Optional[str | Type[BasePrompt]] = None,
        prompt_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = self._get_model(model, model_config)
        self._model_config = model_config

        self._dataset = self._get_dataset(dataset, dataset_config)
        self._dataset_config = dataset_config

        self._metric = self._get_metric(metric, metric_config)
        self._metric_config = metric_config

        if finetune is not None and finetune_config is not None:
            self._finetune = self._get_dataset(finetune, finetune_config.dataset)
            self._finetune_config = finetune_config
        else:
            self._finetune = None
            self._finetune_config = None

        if prompt is not None and prompt_config is not None:
            self._prompt = self._get_prompt(prompt, self._model)
            self._prompt_config = prompt_config
        else:
            self._prompt = None
            self._prompt_config = None

    def run(self):
        pass

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

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}()>"
