import torch
from torch import nn
from pytorch_lightning import LightningModule
import math
from utils import normalized_mse, normalized_mae


class TrainingModule(LightningModule):


    def __init__(self, model, loss_fn="mse", lr=1e-3, weight_decay=0., output_weight=.1):

        super(TrainingModule, self).__init__()
        self.model = model
        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "normalized_mse":
            self.loss_fn = normalized_mse
        else:
            raise NotImplementedError("Loss function not implemented")

        self.lr = lr
        self.weight_decay = weight_decay
        self.output_weight = output_weight

    def forward(self, x, u):
        return self.model(x, u)

    def training_step(self, batch, batch_idx):

        X, u, xdot, y = batch
        xdot_hat, y_hat = self.model(X, u)
        loss_xdot = self.loss_fn(xdot_hat, xdot)
        loss_y = self.loss_fn(y_hat, y)
        self.log("loss_xdot", loss_xdot, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss_y", loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_xdot + self.output_weight * loss_y

    def validation_step(self, batch, batch_idx):

        X, u, xdot, y = batch

        xdot_hat, y_hat = self.model(X, u)

        mse = torch.nn.functional.mse_loss(xdot_hat, xdot)
        mae = torch.nn.functional.l1_loss(xdot_hat, xdot)
        mse_rel = normalized_mse(xdot_hat, xdot)
        mae_rel = normalized_mae(xdot_hat, xdot)

        self.log(f"val_mse", mse, prog_bar=True, logger=True)
        self.log(f"val_mae", mae, prog_bar=True, logger=True)
        self.log(f"val_mse_rel", mse_rel, prog_bar=True, logger=True)
        self.log(f"val_mae_rel", mae_rel, prog_bar=True, logger=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step, warmup=.1):
            if step / total_steps < warmup:
                return step / (total_steps * warmup)
            actual_step = step - total_steps * warmup
            actual_total = total_steps * (1 - warmup)

            return .5 * (1 + math.cos(math.pi * actual_step / actual_total))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

if __name__ == "__main__":
    from model import SimpleMLPSystemIdentifier
    layer = SimpleMLPSystemIdentifier(4, 64)
    model = TrainingModule(layer)

    x = torch.rand(10, 4)
    y = torch.rand(10, 4)

    model.training_step((x, y), 0)

    trajectories = torch.rand(10, 120, 4)
    model.validation_step(trajectories, 0)





