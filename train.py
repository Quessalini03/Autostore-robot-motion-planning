import pytorch_lightning as pl

from lightning_model import DQNLightning

def main():
    model = DQNLightning()

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=150,
    )
    trainer.fit(model)

if __name__ == '__main__':
    main()