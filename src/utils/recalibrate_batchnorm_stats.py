import torch
from torch import nn

@torch.no_grad()
def recalibrate_batchnorm_stats(lightning_module):
    model_training_mode = lightning_module.model.training
    lightning_module.model.eval()
    for module in lightning_module.model.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        module.reset_running_stats()
        module.momentum = None
        module.train()
    for x, *_ in lightning_module.trainer.datamodule.train_dataloader(override_batch_size=lightning_module.trainer.datamodule.eval_batch_size):
        x = x.to(lightning_module.device)
        _ = lightning_module.model(x)
    lightning_module.model.train(model_training_mode)