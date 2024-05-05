import torch

import pandas as pd

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import (
    load_dataset,
    Dataset,
    Image,
)

from transformers import AutoImageProcessor, AutoModelForSequenceClassification
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from transformers import (
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

from dataclasses import dataclass, field
import typing

DATA_PATH = '/kaggle/input/train-1-2k/train'  # Путь до места, где хранятся данные
MODEL_CHECKPOINT = "microsoft/swin-tiny-patch4-window7-224"
DEFAULT_NAME = '1st_try_w_classes'


@dataclass
class Config:
    name: str = DEFAULT_NAME
    MODEL_NAME: str = f"{MODEL_CHECKPOINT.split('/')[-1]}-{name}"
    default_repo_name: str = f"{MODEL_CHECKPOINT.split('/')[-1]}-{name}"
    # Параметры обучения сети
    model_checkpoint: str = MODEL_CHECKPOINT
    batch_size: int = 32
    remove_unused_columns: bool = False
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = batch_size
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = batch_size
    num_train_epochs: int = 5
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    push_to_hub: bool = False

    # Параметры данных
    dataset_type: str = ""
    data_dir: str = DATA_PATH  # + "/train"
    test_size: float = 0.1

    # Можно добавить словарь для других параметров
    other_params: typing.Dict[str, typing.Any] = field(default_factory=dict)

    # Метод для обновления параметров динамически
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Parameter {key} is not a valid config attribute.")


class ImageProcessor:
    """
    Подготавливает картинки к обучению и валидации.
    """
    def __init__(self, config: Config, dataset: Dataset):
        self.splits = None
        self.config = config
        self.dataset = dataset
        self.labels = self.dataset["train"].features["label"].names
        (self.label2id,
         self.id2label) = dict(), dict()
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.test_size = config.test_size

    def get_img_processor(self):
        return self.config.model_checkpoint

    def _normalize(self, image_processor):
        """
        Проводит аугментацию над батчами картинок. Среди аугментаций:
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize.
        `image_processor` наследуется от `AutoImageProcessor.from_pretrained(model_checkpoint)`
        """

        size = 0
        crop_size: tuple = (0, 0)
        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

        if "height" in image_processor.size:
            size = (image_processor.size["height"], image_processor.size["width"])
            crop_size = size
            max_size = None
        elif "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
            crop_size = (size, size)
            max_size = image_processor.size.get("longest_edge")

        train_transforms = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )
        return train_transforms, val_transforms

    def _preprocess_train(self, example_batch):
        """Применяем train_transforms к батчам."""
        transform = self._normalize(self.image_processor)[0]
        example_batch["pixel_values"] = [
            transform(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def _preprocess_val(self, example_batch):
        """Применяем val_transforms к батчам."""
        transform = self._normalize(self.image_processor)[1]
        example_batch["pixel_values"] = [
            transform(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def split_and_transform(self):
        """
        Делит данные на train и validation.
        Возвращает готовые данные для передачи в Trainer.
        """
        self.splits = self.dataset['train'].train_test_split(test_size=self.test_size)

        train_ds = self.splits['train']
        train_ds.set_transform(self._preprocess_train)

        val_ds = self.splits['test']
        val_ds.set_transform(self._preprocess_val)

        return train_ds, val_ds


class LabelManager:
    def __init__(self, dataset: Dataset):
        self.id2label = {}
        self.label2id = {}

    def make_label(self, labels: dict):
        """
        Создает маппинги из списка меток.
        :param labels: Список меток для обработки.
        """
        for i, label in enumerate(labels):
            self.label2id[label] = i
            self.id2label[i] = label

    def check_label(self):
        """
        Проверяет корректность маппинга между id и метками.
        """
        # Проверка, что маппинги корректны
        if len(self.id2label) > 1 and len(self.label2id) > 1:
            print(f'{self.id2label[0]} == {self.label2id[self.id2label[0]]}')
            print(f'{self.id2label[1]} == {self.label2id[self.id2label[1]]}')


class Model:
    def __init__(self, config: Config, labels: LabelManager):
        self.model_checkpoint = config.model_checkpoint
        self.model_name = config.MODEL_NAME
        self.args = TrainingArguments(
            f"{self.model_checkpoint}",
            push_to_hub=config.push_to_hub,
            learning_rate=config.learning_rate,
            save_strategy=config.save_strategy,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            num_train_epochs=config.num_train_epochs,
            per_device_eval_batch_size=config.batch_size,
            evaluation_strategy=config.evaluation_strategy,
            remove_unused_columns=config.remove_unused_columns,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_checkpoint,
            # labels.label2id,
            # labels.id2label,
            ignore_mismatched_sizes=True,
            # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )


class Train:
    def __init__(self, model: Model, config: Config, preprocessor: ImageProcessor):
        self.model = model.model
        self.args = model.args
        self.config = config
        self.train_ds, self.val_ds = preprocessor.split_and_transform()
        self.tokenizer = preprocessor.image_processor
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            tokenizer=self.tokenizer,
            data_collator=self._collate_fn,
        )

    def _collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def train(self):
        train_results = self.trainer.train()
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_results.metrics)
        self.trainer.save_metrics("train", train_results.metrics)
        self.trainer.save_state()

    def evaluate(self):
        eval_results = self.trainer.evaluate()
        self.trainer.log_metrics('eval', eval_results)
        self.trainer.save_metrics('eval', eval_results)

    def push_to_hub(self):
        self.trainer.push_to_hub()


class Predict:
    def __init__(self, model: Model, config: Config, device: int = 0):
        self.device = device
        self.repo_name = 't1msan/' + config.default_repo_name
        self.image_processor = AutoImageProcessor.from_pretrained(self.repo_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.repo_name)
        self.pipe = pipeline('image-classification', model=self.repo_name, device=self.device)

    def set_device(self, device):
        self.device = device
        print(f"Вычисления производятся на {'GPU' if not self.device else 'CPU'}")

    def show_device(self):
        print(f"Вычисления производятся на {'GPU' if not self.device else 'CPU'}")

    def set_repo_name(self, new_repo_name):
        self.repo_name = new_repo_name
        self.image_processor = AutoImageProcessor.from_pretrained(self.repo_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.repo_name)
        self.pipe = pipeline('image-classification', model=self.repo_name, device=self.device)
        print(f'Установлен новый репозиторий для загрузки предобученной модели. \n'
              f'Актуальный репозиторий: {self.repo_name}')

    def _data_preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dataset]:
        if df.empty:
            raise(Exception("Dataset is empty"))
        if 'id' not in df.columns:
            raise ValueError()
        else:
            df = df[['id']]
        if isinstance(df, pd.DataFrame):
            dataset_test = Dataset.from_pandas(df).cast_column("id", Image())
            return df, dataset_test

    def predict(self, dataset: pd.DataFrame) -> pd.DataFrame:

        df, dataset_test = self._data_preprocess(dataset)

        def check_fake_score(i):
            image = dataset_test[i]["id"]
            data = self.pipe(image)
            for item in data:
                if item['label'] == 'ai':
                    return item['score']

        df['target'] = df.index.to_series().apply(check_fake_score)

        return df
