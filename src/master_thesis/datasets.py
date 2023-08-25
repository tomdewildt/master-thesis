from typing import Any, Dict, List, Optional

from datasets import Dataset, load_dataset, concatenate_datasets

from master_thesis.base import BaseDataset


RANDOM_SEED = 42


class COQADataset(BaseDataset):
    _train: Optional[Dataset]
    _test: Optional[Dataset]

    def __init__(
        self,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> None:
        super().__init__("coqa", train_limit, test_limit)

        self._dataset_dict = load_dataset(self._dataset_id)
        self._train = None
        self._test = None

    @property
    def train(self) -> Dataset:
        if not self._train:
            self._train = (
                self._dataset_dict["train"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["train"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            if self._train_limit:
                self._train = self._train.select(range(self._train_limit))

        return self._train

    @property
    def test(self) -> Dataset:
        if not self._test:
            self._test = (
                self._dataset_dict["validation"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["validation"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            if self._test_limit:
                self._test = self._test.select(range(self._test_limit))

        return self._test

    def _format(self, batch: Dict[str, Any]) -> Dict[str, List]:
        references = []
        questions = []
        answers = []

        for story, question, answer in zip(
            batch["story"],
            batch["questions"],
            batch["answers"],
        ):
            references.extend([story] * len(question))
            questions.extend(question)
            answers.extend(answer["input_text"])

        return {"references": references, "question": questions, "answer": answers}


class HotpotQADataset(BaseDataset):
    _train: Optional[Dataset]
    _test: Optional[Dataset]

    def __init__(
        self,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> None:
        super().__init__("hotpot_qa", train_limit, test_limit)

        self._dataset_dict = load_dataset(self._dataset_id, "fullwiki")
        self._train = None
        self._test = None

    @property
    def train(self) -> Dataset:
        if not self._train:
            self._train = (
                self._dataset_dict["train"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["train"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            if self._train_limit:
                self._train = self._train.select(range(self._train_limit))

        return self._train

    @property
    def test(self) -> Dataset:
        if not self._test:
            validation = self._dataset_dict["validation"].map(
                self._format,
                batched=True,
                remove_columns=self._dataset_dict["validation"].column_names,
            )
            test = self._dataset_dict["test"].map(
                self._format,
                batched=True,
                remove_columns=self._dataset_dict["test"].column_names,
            )
            self._test = concatenate_datasets([validation, test]).shuffle(
                seed=RANDOM_SEED
            )

            if self._test_limit:
                self._test = self._test.select(range(self._test_limit))

        return self._test

    def _format(self, batch: Dict[str, Any]) -> Dict[str, List]:
        references = []
        questions = []
        answers = []

        for context, question, answer in zip(
            batch["context"],
            batch["question"],
            batch["answer"],
        ):
            references.append(["".join(text) for text in context["sentences"]])
            questions.append(question)
            answers.append(answer)

        return {"references": references, "question": questions, "answer": answers}


class SQUADv1Dataset(BaseDataset):
    _train: Optional[Dataset]
    _test: Optional[Dataset]

    def __init__(
        self,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> None:
        super().__init__("squad", train_limit, test_limit)

        self._dataset_dict = load_dataset(self._dataset_id)
        self._train = None
        self._test = None

    @property
    def train(self) -> Dataset:
        if not self._train:
            self._train = (
                self._dataset_dict["train"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["train"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            if self._train_limit:
                self._train = self._train.select(range(self._train_limit))

        return self._train

    @property
    def test(self) -> Dataset:
        if not self._test:
            self._test = (
                self._dataset_dict["validation"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["validation"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            if self._test_limit:
                self._test = self._test.select(range(self._test_limit))

        return self._test

    def _format(self, batch: Dict[str, Any]) -> Dict[str, List]:
        references = []
        questions = []
        answers = []

        for context, question, answer in zip(
            batch["context"],
            batch["question"],
            batch["answers"],
        ):
            references.extend([context] * len(answer["text"]))
            questions.extend([question] * len(answer["text"]))
            answers.extend(answer["text"])

        return {"references": references, "question": questions, "answer": answers}


class SQUADv2Dataset(BaseDataset):
    _train: Optional[Dataset]
    _test: Optional[Dataset]

    def __init__(
        self,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> None:
        super().__init__("squad_v2", train_limit, test_limit)

        self._dataset_dict = load_dataset(self.dataset_id)
        self._train = None
        self._test = None

    @property
    def train(self) -> Dataset:
        if not self._train:
            self._train = (
                self._dataset_dict["train"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["train"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            if self._train_limit:
                self._train = self._train.select(range(self._train_limit))

        return self._train

    @property
    def test(self) -> Dataset:
        if not self._test:
            self._test = (
                self._dataset_dict["validation"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["validation"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            if self._test_limit:
                self._test = self._test.select(range(self._test_limit))

        return self._test

    def _format(self, batch: Dict[str, Any]) -> Dict[str, List]:
        references = []
        questions = []
        answers = []

        for context, question, answer in zip(
            batch["context"],
            batch["question"],
            batch["answers"],
        ):
            references.extend([context] * len(answer["text"]))
            questions.extend([question] * len(answer["text"]))
            answers.extend(answer["text"])

        return {"references": references, "question": questions, "answer": answers}


class TriviaQADataset(BaseDataset):
    _train: Optional[Dataset]
    _test: Optional[Dataset]

    def __init__(
        self,
        train_limit: Optional[int] = None,
        test_limit: Optional[int] = None,
    ) -> None:
        super().__init__("trivia_qa", train_limit, test_limit)

        self._dataset_dict = load_dataset(self._dataset_id, "rc")
        self._train = None
        self._test = None

    @property
    def train(self) -> Dataset:
        if not self._train:
            self._train = (
                self._dataset_dict["train"]
                .map(
                    self._format,
                    batched=True,
                    remove_columns=self._dataset_dict["train"].column_names,
                )
                .shuffle(seed=RANDOM_SEED)
            )

            if self._train_limit:
                self._train = self._train.select(range(self._train_limit))

        return self._train

    @property
    def test(self) -> Dataset:
        if not self._test:
            validation = self._dataset_dict["validation"].map(
                self._format,
                batched=True,
                remove_columns=self._dataset_dict["validation"].column_names,
            )
            test = self._dataset_dict["test"].map(
                self._format,
                batched=True,
                remove_columns=self._dataset_dict["test"].column_names,
            )
            self._test = concatenate_datasets([validation, test]).shuffle(
                seed=RANDOM_SEED
            )

            if self._test_limit:
                self._test = self._test.select(range(self._test_limit))

        return self._test

    def _format(self, batch: Dict[str, Any]) -> Dict[str, List]:
        references = []
        questions = []
        answers = []

        for pages, question, answer in zip(
            batch["entity_pages"],
            batch["question"],
            batch["answer"],
        ):
            references.extend([pages["wiki_context"]] * len(answer["aliases"]))
            questions.extend([question] * len(answer["aliases"]))
            answers.extend(answer["aliases"])

        return {"references": references, "question": questions, "answer": answers}
