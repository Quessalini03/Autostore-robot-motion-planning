import pytorch_lightning as pl
import torch

from lightning_model import DQNLightning

from config import args

def main():
    model = DQNLightning.load_from_checkpoint(args.train_ckpt)

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=args.num_epochs,
    )
    trainer.fit(model)

if __name__ == '__main__':
    main()