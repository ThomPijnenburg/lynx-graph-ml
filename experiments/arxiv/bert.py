import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from time import time
from tqdm import tqdm
from pathlib import Path
from typing import Union, Dict
import pytorch_lightning as pl

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from transformers import AutoTokenizer, AutoModel

from lynks.models import LMEncoderClassificationModel
from lynks.data import load_arxiv, load_arxiv_text


class ArXivTextDataset(Dataset):
    def __init__(self, encodings, labels=None, meta: Union[Dict[str, torch.tensor], None] = None):
        self.encodings: dict = encodings
        self.labels = labels
        self.meta = meta

    @classmethod
    def from_dataframe(cls, dataframe, labels, tokenizer):
        inputs = dataframe["text"].values.tolist()
        # labels = dataframe["label"].values.tolist()

        encodings = tokenizer(inputs, truncation=True,
                              padding=True, max_length=512)
        return cls(encodings, labels)

    def __getitem__(self, idx) -> Dict[str, torch.tensor]:
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx].clone().detach()
        if self.meta is not None:
            for k, v in self.meta.items():
                item[k] = torch.tensor(v[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class ArXivTextDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, tokenizer_ckpt: str, train_batch_size: int = 32,
                 test_batch_size: int = 8, num_workers: int = 4,):
        super().__init__()

        self.data_path = data_path
        self.dataset = None
        self.tokenizer_ckpt = tokenizer_ckpt  # path to dir holder tokenizer dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.num_classes = None

    def prepare_data(self):

        _ = load_arxiv(data_path=self.data_path)
        text_filepath = load_arxiv_text(data_path=self.data_path)
        print(text_filepath)

    def setup(self, stage):

        tokenizer = AutoTokenizer.from_pretrained(
            Path(self.tokenizer_ckpt).joinpath("tokenizer"))

        # Load splits (node indices) and labels from graph dataset
        self.dataset = load_arxiv(data_path=self.data_path)
        split_idx = self.dataset.get_idx_split()

        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"]
        test_idx = split_idx["test"]

        graph, label = self.dataset[0]

        # From 'Mappings' load the node idx to mag id, need this to merge on the text data
        nodeidx2paperid_df = pd.read_csv(self.data_path.joinpath(
            "ogbn_arxiv/mapping/nodeidx2paperid.csv"), sep=",", header=0, index_col=None)
        nodeidx2paperid_df.columns = ["node_idx", "mag_id"]

        # Load text data (title + abstract)
        text_filepath = load_arxiv_text(data_path=self.data_path)
        title_abs_df = pd.read_csv(
            text_filepath.parent.joinpath("titleabs.tsv"), sep="\t", header=None, index_col=None)
        title_abs_df.columns = ["mag_id", "title", "abstract"]

        # Combine title and abstract into single block text
        title_abs_df["text"] = title_abs_df["title"] + \
            ". " + title_abs_df["abstract"]

        title_abs_df = title_abs_df.drop(columns=["title", "abstract"])

        # Merge node indices onto text dataframe, needed for splitting
        title_abs_ids_df = pd.merge(
            left=title_abs_df,
            right=nodeidx2paperid_df,
            how="inner",
            on="mag_id"
        ).sort_values(by=["node_idx"], ascending=True).set_index("node_idx")

        # Generate splits
        label = torch.squeeze(torch.tensor(label))
        train_df = title_abs_ids_df.loc[train_idx]
        train_labels = label[train_idx]
        valid_df = title_abs_ids_df.loc[valid_idx]
        valid_labels = label[valid_idx]
        test_df = title_abs_ids_df.loc[test_idx]
        test_labels = label[test_idx]

        # num_unique = len(np.unique(train_labels.ravel())) # TODO: draw num classes from data
        self.num_classes = 40

        self.train_ds = ArXivTextDataset.from_dataframe(
            dataframe=train_df, labels=train_labels, tokenizer=tokenizer)
        self.valid_ds = ArXivTextDataset.from_dataframe(
            dataframe=valid_df, labels=valid_labels, tokenizer=tokenizer)
        self.test_ds = ArXivTextDataset.from_dataframe(
            dataframe=test_df, labels=test_labels, tokenizer=tokenizer)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_ds, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.test_batch_size, num_workers=self.num_workers)


def compute_metrics(pred, labels):
    preds = pred.argmax(-1)
    labels = labels

    f1 = f1_score(labels, preds, average="weighted")
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": prec}


class LitLMEncoderClassificationModel(pl.LightningModule):
    """ TODO: update loss function used
    """

    def __init__(self, num_classes, checkpoint, hidden_dim=128, dropout=0.3, learning_rate=1e-3, label_smoothing=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = LMEncoderClassificationModel(
            hidden_dim=self.hparams.hidden_dim,
            n_labels=num_classes,
            dropout=self.hparams.dropout,
            checkpoint=checkpoint,
        )

    def forward(self, input_ids, attention_mask):
        # use forward for inference/predictions
        logits = self.backbone(input_ids, attention_mask)
        return logits

    def training_step(self, batch, batch_idx):
        # Forward pass
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        # loss = F.binary_cross_entropy_with_logits(
        # outputs, labels, label_smoothing=self.hparams.label_smoothing)

        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_id=None):
        # Forward pass
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        loss = F.cross_entropy(outputs, labels)

        self.log("valid_loss", loss, prog_bar=True)

        # temp
        # preds = torch.cat([x["preds"] for x in data]).detach().cpu()
        # labels = torch.cat([x["labels"] for x in data]).detach().cpu()
        # loss = torch.stack([x["loss"] for x in data]).mean()

        valid_metrics = compute_metrics(outputs.cpu(), labels.cpu())

        for metric, score in valid_metrics.items():
            self.log(f"valid_{metric}", float(score),
                     prog_bar=True, on_epoch=True)

        return {"loss": loss, "preds": outputs, "labels": labels}

    # def on_validation_epoch_end(self, outputs):
    #     preds = torch.cat([x["preds"] for x in outputs]).detach().cpu()
    #     labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()

    #     self.log("valid_loss", loss, prog_bar=True)

    #     valid_metrics = compute_metrics(preds, labels)

    #     for metric, score in valid_metrics.items():
    #         self.log(f"valid_{metric}", score, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_id=None):
        # Forward pass
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        loss = F.cross_entropy(outputs, labels)

        self.log("test_loss", loss, prog_bar=True)

        # preds = torch.cat([x["preds"] for x in data]).detach().cpu()
        # labels = torch.cat([x["labels"] for x in data]).detach().cpu()
        # loss = torch.stack([x["loss"] for x in data]).mean()

        test_metrics = compute_metrics(outputs.cpu(), labels.cpu())

        for metric, score in test_metrics.items():
            self.log(f"test_{metric}", float(score),
                     prog_bar=True, on_epoch=True)

        return {"loss": loss, "preds": outputs, "labels": labels}

    # def on_test_epoch_end(self, outputs):
    #     # relies on test being called in trainer through `trainer.test(dataloaders=[...])`

    #     for dataloader_index, dataloader_output in enumerate(outputs):
    #         preds = torch.cat([x["preds"]
    #                            for x in dataloader_output]).detach().cpu()
    #         labels = torch.cat([x["labels"]
    #                             for x in dataloader_output]).detach().cpu()
    #         loss = torch.stack([x["loss"] for x in dataloader_output]).mean()

    #         prefix = "test" if dataloader_index < 1 else f"test_{dataloader_index}"

    #         self.log(f"{prefix}_loss", loss, prog_bar=True)

    #         test_metrics = compute_metrics(preds, labels)

    #         for metric, score in test_metrics.items():
    #             self.log(f"{prefix}_{metric}", score, prog_bar=True)

    def pred_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        logits = self.backbone(input_ids, attention_mask=attention_mask)

        return logits

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def build_args() -> ArgumentParser:
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early_stopper", action="store_true", default=False)
    parser.add_argument("--model_name", type=str,
                        default="scibert-arxiv-clf")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--experiment_label", type=str,
                        default="arxiv-bert-clf")
    parser.add_argument_group(
        "LitRelationClassificationModel")
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=float, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    return parser


def download_tokenizer_and_model(model_ckpt, outdir):
    # fix in setup
    if not outdir.is_dir():
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        tokenizer_path = Path(outdir).joinpath("tokenizer")
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)

        model_path = Path(outdir).joinpath("model")
        model_path.mkdir(parents=True, exist_ok=True)

        model = AutoModel.from_pretrained(
            model_ckpt)
        model.save_pretrained(model_path)
    else:
        print("pretrained already available on machine...")


def main(args):
    # ------------
    # args
    # ------------
    # args.limit_train_batches = 0.1

    print(f"Running train script with args: {args}")

    # dataset = args.dataset
    # base_lm = args.base_lm
    # limit_train_batches = args.limit_train_batches
    limit_train_batches = 0.01

    hidden_dim = args.hidden_dim
    dropout = args.dropout
    pretrained = args.pretrained  # path to model and tokenizer

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    label_smoothing = args.label_smoothing
    early_stopper = args.early_stopper

    timestamp = str(int(time()))
    data_root = Path(args.data_root)
    dataset_path = data_root.joinpath("ogb-arxiv")
    model_name = args.model_name
    # model_path = data_root.joinpath(f"{model_name}/{timestamp}")
    model_path = data_root.joinpath(f"models/{model_name}/{timestamp}")

    checkpoint_path = model_path.joinpath(f"checkpoints")

    dataset_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    pretrained_path = data_root.joinpath("pretrained")
    pretrained_path.mkdir(parents=True, exist_ok=True)

    model_checkpoint = Path(pretrained).joinpath("model")
    logging_dir = data_root.joinpath("logs")

    # TODO: enable below
    # download_tokenizer_and_model(
    #     model_ckpt=pretrained, outdir=pretrained_path.joinpath("scibert"))

    experiment_label = args.experiment_label

    cpu_count = os.cpu_count()

    # ------------
    # data
    # ------------
    data_module = ArXivTextDataModule(
        data_path=dataset_path,
        tokenizer_ckpt=pretrained,
        train_batch_size=batch_size,
        num_workers=cpu_count,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # ------------
    # model
    # # ------------
    model = LitLMEncoderClassificationModel(
        num_classes=data_module.num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        checkpoint=model_checkpoint,
        learning_rate=learning_rate,
        label_smoothing=label_smoothing,
    )

    # ------------
    # training
    # ------------
    train_callbacks = [TQDMProgressBar(refresh_rate=10)]

    # logger
    tf_logger = TensorBoardLogger(
        save_dir=logging_dir, name="arxiv-bert-clf")

    if early_stopper:
        early_stop_callback = EarlyStopping(
            monitor="valid_accuracy", min_delta=0.01, patience=5, verbose=True, mode="max"
        )

        train_callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{checkpoint_path}",
        filename="checkpoint-{epoch}-{step}-{valid_loss:.2f}-{valid_accuracy:.2f}",
        monitor="valid_accuracy",
        save_top_k=1,
        mode="max",
        every_n_epochs=1,
    )
    train_callbacks.append(checkpoint_callback)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        logger=tf_logger,
        max_epochs=epochs,
        callbacks=train_callbacks,
        enable_progress_bar=True,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_train_batches,
        limit_test_batches=limit_train_batches,
        val_check_interval=0.20,
        max_time="00:24:00:00",  # 24 hour
        enable_checkpointing=True,
    )
    trainer.fit(model, datamodule=data_module)

    # ------------
    # testing
    # ------------

    result = trainer.test(datamodule=data_module)

    print(result)

    # torch.save(model.backbone.state_dict(), f"{model_path}/{model_name}.pt")


if __name__ == "__main__":
    parser = build_args()
    args = parser.parse_args()

    main(args)
