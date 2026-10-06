import hydra
from hydra.utils import instantiate
import torch


@hydra.main(version_base=None, config_path="config", config_name="show")
def main(config):
    torch.set_float32_matmul_precision("high")
    datamodule = instantiate(config.datamodule)
    litmodule = instantiate(config.litmodule)
    callbacks = instantiate(config.callbacks)
    trainer = instantiate(config.trainer, logger=False, callbacks=[cb for cb in callbacks.values()])
    trainer.validate(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
