from copy import deepcopy
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import torch
import wandb


@hydra.main(version_base=None, config_path="config", config_name="validate")  # only hydra config and overrides
def main(overrides):
    # set to prevent warning
    torch.set_float32_matmul_precision("high")

    # get checkpoint
    api = wandb.Api()
    project_path = f"{overrides.wandb.entity}/{overrides.wandb.project}"
    checkpoint_path = f"{project_path}/model-{overrides.runid}:{overrides.checkpoint}"
    checkpoint = Path(api.artifact(checkpoint_path).download()) / "model.ckpt"

    # get training config and merge with overrides
    run = api.run(f"{project_path}/{overrides.runid}")
    config = OmegaConf.create(deepcopy(run.config))
    with open_dict(config):
        config.merge_with(overrides)

    # dataset + dataloader = lightning datamodule
    datamodule = instantiate(config.datamodule)

    # network + loss + optimizer = lightning module
    # not strict state dict loading because not using compiled network params
    network = instantiate(config.network)
    loss_fns = instantiate(config.loss_fns)
    litmodule = instantiate(config.litmodule, network, loss_fns, optimizer=None)
    litmodule.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"], strict=False)
    litmodule.eval()
    litmodule.freeze()

    # callbacks
    callbacks = instantiate(config.callbacks)
    callbacks.pop("checkpoint")

    # trainer and validate!
    trainer = instantiate(config.trainer, logger=False, callbacks=[cb for cb in callbacks.values()])
    trainer.validate(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
