import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import OmegaConf
import torch
import wandb


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):
    # set to prevent warning
    torch.set_float32_matmul_precision("high")

    # reproducibility
    if config.trainer.deterministic:
        seed_everything(42, workers=True)

    # dataset + dataloader = lightning datamodule
    datamodule = instantiate(config.datamodule)

    # network + loss + optimizer = lightning module
    network = instantiate(config.network)
    loss_fns = instantiate(config.loss_fns)
    optimizer = instantiate(config.optimizer)
    litmodule = instantiate(config.litmodule, network, loss_fns, optimizer)

    # callbacks
    callbacks = instantiate(config.callbacks)

    # logger
    # NOTE: https://docs.wandb.ai/guides/app/features/panels/code/
    # stores code as an artifact but doesn't work that well yet
    wandb.require("legacy-service")  # to have diff.patch stored
    logger = instantiate(config.logger)
    if logger is not None:
        logger.log_hyperparams(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
        enable_checkpointing = True
    else:
        logger = False
        enable_checkpointing = False
        callbacks.pop("checkpoint", None)

    # trainer and train!
    trainer = instantiate(
        config.trainer,
        logger=logger,
        callbacks=[cb for cb in callbacks.values()],
        enable_checkpointing=enable_checkpointing,
    )
    trainer.fit(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
