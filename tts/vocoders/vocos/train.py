import pytorch_lightning as pl

from speechflow.logging.server import LoggingServer

assert int(pl.__version__[0]) == 1, RuntimeError("pytorch_lightning==1.8.6 required")

from pytorch_lightning.cli import LightningCLI

if __name__ == "__main__":
    with LoggingServer.ctx("vocos_logs.txt"):
        cli = LightningCLI(run=False)
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
