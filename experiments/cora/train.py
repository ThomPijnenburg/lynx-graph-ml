import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch_geometric as tg

from argparse import ArgumentParser


from pathlib import Path
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from lynks.models import GCNModel
from lynks.data import load_cora

AVAIL_GPUS = min(1, torch.cuda.device_count())


class LightningGNN(pl.LightningModule):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "GCN":
            self.model = GCNModel(**model_kwargs)
        else:
            raise ValueError(f"Model {model_name} not implemented...")

        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index

        # pass through model
        x = self.model(x, edge_index)

        # use 'mask' to filter nodes to compute loss
        mask = None

        if mode == "train":
            mask = data.train_mask
        elif mode == "valid":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, "Unsupported mode"

        loss = self.loss_module(x[mask], data.y[mask])

        acc = (x[mask].argmax(dim=-1) == data.y[mask]
               ).sum().float() / mask.sum()

        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="valid")
        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)


def run_training(model_name: str, dataset, epochs, outdir, logging_dir, **model_kwargs):
    outdir.mkdir(exist_ok=True, parents=True)

    # init dataloader
    dataloader = tg.loader.DataLoader(dataset, batch_size=1, num_workers=4)
    logger = TensorBoardLogger(save_dir=logging_dir, name="cora-gcn")

    # set up Trainer
    trainer = pl.Trainer(
        default_root_dir=outdir,
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        logger=logger,
        log_every_n_steps=1
    )

    # run training
    model = LightningGNN(model_name,
                         c_in=dataset.num_node_features,
                         c_out=dataset.num_classes,
                         **model_kwargs)
    trainer.fit(model, train_dataloaders=dataloader,
                val_dataloaders=dataloader)

    # perform testing
    test_scores = trainer.test(model, dataloaders=dataloader, verbose=True)

    # return results

    return test_scores


if __name__ == "__main__":
    parser = ArgumentParser(description="Model training routing")
    parser.add_argument("--data_root", type=str,
                        help="Path to data root dir")
    parser.add_argument('--model', type=str, default="GCN",
                        help='Path to write models output')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs to train')

    args = parser.parse_args()

    data_root = Path(args.data_root)

    dataset_path = data_root.joinpath("Planetoid")
    model_path = data_root.joinpath("models")
    logs_path = data_root.joinpath("logs")

    model_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    dataset = load_cora(dataset_path)

    test_scores = run_training(model_name=args.model,
                               dataset=dataset,
                               c_hidden=32,  # unused
                               epochs=args.epochs,
                               outdir=model_path,
                               logging_dir=logs_path)

    print(test_scores)
