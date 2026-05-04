from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeAlias
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader

from cancersd.losses.loss import ContrastiveLoss, BaseLoss
from cancersd.utils.metrics import get_statistic, get_performance_evaluation


Tensors: TypeAlias = tuple[torch.Tensor, ...] | list[torch.Tensor]


class Trainer:
    def __init__(self, config: dict[str, Any], model: nn.Module, dataloaders: dict[str, DataLoader], paths) -> None:
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.paths = paths

        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        trainer_cfg = config['experiment']['trainer']

        self.epochs = trainer_cfg['epochs']
        self.best_metric = float('-inf')

        optim_cfg = trainer_cfg['optimizer']
        optim_name = optim_cfg.get('name', 'AdamW')

        if optim_name == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optim_cfg.get('lr', 1e-4),
                weight_decay=optim_cfg.get('weight_decay', 1e-4)
            )
        elif optim_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optim_cfg.get("lr", 1e-3),
                weight_decay=optim_cfg.get("weight_decay", 0.0),
            )
        else:
            raise NotImplementedError(f'Unsupported optimizer: {optim_name}')

        sched_cfg = trainer_cfg['scheduler']
        sched_name = sched_cfg.get('name', 'CosineAnnealingWarmRestarts')
        if sched_name == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif sched_name == 'ExponentialLR':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        elif sched_name == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=25, T_mult=2)
        else:
            raise NotImplementedError(f'Unsupported scheduler: {sched_name}')

        self.loss_functions = {
            'contrastive': ContrastiveLoss(trainer_cfg['hyper_parameters']['tau']).to(self.device),
            'generation': nn.MSELoss(),
            'diagnosis': nn.CrossEntropyLoss(),
            'base': BaseLoss(coefficient=[1, 1, 1]).to(self.device),
        }

    def fit(self) -> None:
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            # validation_metrics = self.validate(epoch)

            print(
                f"[Epoch {epoch:03d}/{self.epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                # f"val_loss={validation_metrics['loss']:.4f} "
                # f"val_acc={validation_metrics.get('accuracy', 0.0):.4f}"
            )

        # self.save_checkpoint('last.pt', self.epochs, {'best_metric': self.best_metric})

    def train_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.toggle_stage('train')

        total_loss = 0.0
        total_num = 0
        correct = 0

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            batch = self.move_to_device(batch)

            outputs = self.forward_batch(batch)
            loss = self.compute_loss(outputs, batch)

            self.optimizer.zero_grad()
            loss['loss_base'].backward()
            self.optimizer.step()

            batch_size = batch['label'].size(0)
            total_loss += loss['loss_base'].item() * batch_size
            total_num += batch_size
            correct += (outputs[-1][:batch_size, :].argmax(dim=-1) == batch['label']).sum().item()

        self.scheduler.step()

        return {
            'loss': total_loss / max(total_num, 1),
            'accuracy': correct / max(total_num, 1)
        }

    def forward_batch(self, batch: dict[str, Any]) -> Tensors:
        return self.model(batch['features'])

    def compute_loss(self, outputs: Tensors, batch: Any) -> dict[str, torch.Tensor]:
        samples, labels = batch['features'], batch['label']
        complete, incomplete, available, origins, projections, reconstructed, generated, diagnoses = outputs

        generation_loss = torch.zeros(1).squeeze().to(self.device)
        if complete.numel() >= 1:
            complete = complete.view(-1)
            generation_loss += self.loss_functions['generation'](reconstructed, origins[complete])
        if incomplete.numel() >= 1:
            incomplete = incomplete.view(-1)
            generation_loss += self.loss_functions['generation']((generated * available)[incomplete],
                                                                 origins[incomplete])
        if complete.numel() + incomplete.numel() == 0:
            generation_loss += self.loss_functions['generation'](reconstructed, origins)

        contrastive_loss = torch.zeros(1).squeeze().to(self.device)
        if complete.numel() >= 2:
            contrastive_loss = self.loss_functions['contrastive'](projections)

        if complete.numel() + incomplete.numel() > 0:
            labels = torch.cat([labels[complete].repeat(2, ), labels], dim=0)
        diagnosis_loss = self.loss_functions['diagnosis'](diagnoses, labels)

        base_loss = self.loss_functions['base'](
            torch.cat([contrastive_loss.unsqueeze(0), generation_loss.unsqueeze(0), diagnosis_loss.unsqueeze(0)])
        )

        return {
            'loss_contrastive': contrastive_loss,
            'loss_generation': generation_loss,
            'loss_diagnosis': diagnosis_loss,
            'loss_base': base_loss
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.toggle_stage('test')

        total_loss = 0.0
        total_num = 0
        correct = 0

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            batch = self.move_to_device(batch)

            outputs = self.forward_batch(batch)
            loss = self.compute_loss(outputs, batch)

            batch_size = batch['label'].size(0)
            total_loss += loss['loss_base'].item() * batch_size
            total_num += batch_size
            correct += (outputs[-1][:batch_size, :].argmax(dim=-1) == batch['label']).sum().item()

        return {
            'loss': total_loss / max(total_num, 1),
            'accuracy': correct / max(total_num, 1)
        }

    @torch.no_grad()
    # def test(self, epoch: int) -> dict[str, float]:
    def test(self) -> dict[str, float]:
        # self.load_checkpoint('best.pt')

        self.model.toggle_stage('test')

        total_num = 0
        correct = 0

        for batch in self.dataloaders['test']:
            batch = self.move_to_device(batch)
            outputs = self.model.test(batch['features'])

            pred = outputs[-1].argmax(dim=1)
            label = batch['label']

            total_num += batch['label'].size(0)
            correct += (pred == label).sum().item()

        test_acc = correct / max(total_num, 1)
        print(f"[Test] acc={test_acc:.4f}")

        return {"test_acc": test_acc}

    def move_to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)

        if isinstance(batch, dict):
            return {
                key: self.move_to_device(value) for key, value in batch.items()
            }

        if isinstance(batch, list):
            return [self.move_to_device(value) for value in batch]

        if isinstance(batch, tuple):
            return tuple([self.move_to_device(value) for value in batch])

        return batch

    def save_checkpoint(self, filename: str, epoch: int, metrics: dict[str, Any]) -> None:
        path = Path(self.paths.checkpoint_dir) / filename

        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                    dir=path.parent,
                    prefix=path.stem + '_',
                    suffix='.tmp',
                    delete=False
            ) as tmp:
                tmp_name = tmp.name
                torch.save(ckpt, tmp_name)

            os.replace(tmp_name, path)

            # clean up the old checkpoints
            files = sorted(
                [f for f in path.parent.iterdir() if f.stem.startswith(path.stem) and f.suffix == '.pt'],
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            for old_file in files[2:]:
                try:
                    old_file.unlink()
                except:
                    pass

        except KeyboardInterrupt:
            if 'tmp_name' in locals():
                os.unlink(tmp_name)
            raise
        except Exception as e:
            if 'temp_name' in locals() and os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except:
                    pass

    def load_checkpoint(self, filename: str) -> None:
        path = Path(self.paths.checkpoint_dir) / filename

        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
