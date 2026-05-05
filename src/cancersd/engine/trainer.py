from __future__ import annotations

import os
import io
from pathlib import Path
from typing import Any, TypeAlias, Optional
import tempfile
import operator
import threading
import queue

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import zstandard as zstd

from cancersd.losses.loss import MaskedMSELoss, ContrastiveLoss, BaseLoss
from cancersd.utils.metrics import get_statistic, get_performance_evaluation


Tensors: TypeAlias = tuple[torch.Tensor, ...] | list[torch.Tensor]


# class AsyncCheckpointSaver:
#     def __init__(self, compression_level: int = 3):
#         self.compressor = zstd.ZstdCompressor(level=compression_level)
#         self.save_queue = queue.Queue(maxsize=1)
#         self.save_thread: Optional[threading.Thread] = None
#         self._stop_event = threading.Event()
#
#     def start_background_saver(self):
#         if self.save_thread is not None and self.save_thread.is_alive():
#             return
#
#         self._stop_event.clear()


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

        early_stopping_cfg = self.config['experiment']['trainer'].get('early_stopping', {})
        self.monitor = early_stopping_cfg.get('monitor', "val_acc")
        comparison_symbol = early_stopping_cfg.get('comparison', "greater")
        if comparison_symbol == 'greater':
            self.comparer = operator.gt
        elif comparison_symbol == 'less':
            self.comparer = operator.lt
        else:
            raise NotImplementedError(f'unsupported comparison symbol: {comparison_symbol}')

        if 'acc' in self.monitor:
            self.best_metric = float('-inf')
        elif 'loss' in self.monitor:
            self.best_metric = float('inf')
        else:
            raise NotImplementedError(f'unsupported monitor: {self.monitor}')

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
                lr=optim_cfg.get('lr', 1e-3),
                weight_decay=optim_cfg.get("weight_decay", 0.0),
            )
        else:
            raise NotImplementedError(f'unsupported optimizer: {optim_name}')

        sched_cfg = trainer_cfg['scheduler']
        sched_name = sched_cfg.get('name', 'CosineAnnealingWarmRestarts')
        if sched_name == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sched_cfg.get('step_size', 30), gamma=sched_cfg.get('gamma', 0.1)
            )
        elif sched_name == 'ExponentialLR':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=sched_cfg.get('gamma', 0.99)
            )
        elif sched_name == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=sched_cfg.get('T_0', 25), T_mult=sched_cfg.get('T_mult', 25)
            )
        else:
            raise NotImplementedError(f'unsupported scheduler: {sched_name}')

        self.loss_functions = {
            'contrastive': ContrastiveLoss(trainer_cfg['hyper_parameters']['tau']).to(self.device),
            'generation': nn.MSELoss(),
            'masked_generation': MaskedMSELoss(),
            'diagnosis': nn.CrossEntropyLoss(),
            'base': BaseLoss(coefficient=[1, 1, 1]).to(self.device),
        }

        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()

    def fit(self) -> None:
        pbar = tqdm(
            total=self.epochs,
            desc=f'[Epoch 000/{self.epochs}]',
            bar_format='{desc} {bar} {elapsed}<{remaining} {postfix}',
            unit='epoch'
        )
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            validation_metrics = self.validate('validation', epoch)

            current_metric = validation_metrics.get(self.monitor, None)
            if current_metric is None:
                print(f'no specified indicator was obtained in the {epoch}-th round')
                continue

            if self.comparer(current_metric, self.best_metric):
                self.best_metric = current_metric
                self.save_checkpoint('best.pt', epoch, validation_metrics)

            # print(
            #     f'[Epoch {epoch:03d}/{self.epochs}] '
            #     f'train_loss={train_metrics["loss"]:.4f} '
            #     f'val_loss={validation_metrics["val_loss"]:.4f} '
            #     f'val_acc={validation_metrics.get("val_acc", 0.0):.4f}'
            # )
            pbar.set_description(f'[Epoch {epoch:03d}/{self.epochs}]')
            pbar.set_postfix({
                'train_loss': f'{train_metrics["loss"]:.4f}',
                'val_loss': f'{validation_metrics["val_loss"]:.4f}',
                'val_acc': f'{validation_metrics.get("val_acc", 0.0):.4f}'
            })
            pbar.update(1)

        pbar.close()
        self.save_checkpoint('last.pt', self.epochs, {'best_metric': self.best_metric})

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
            correct += (outputs[-1][-batch_size:, :].argmax(dim=-1) == batch['label']).sum().item()

        self.scheduler.step()

        return {
            'loss': total_loss / max(total_num, 1),
            'accuracy': correct / max(total_num, 1)
        }

    def forward_batch(self, batch: dict[str, Any]) -> Tensors:
        return self.model(batch['features'])

    def compute_loss(self, outputs: Tensors, batch: Any) -> dict[str, torch.Tensor]:
        labels = batch['label']
        complete, incomplete, available, origins, projections, reconstructed, generated, diagnoses = outputs

        complete = complete.reshape(-1)
        incomplete = incomplete.reshape(-1)

        has_complete = complete.numel() > 0
        has_incomplete = incomplete.numel() > 0
        has_selected = has_complete or has_incomplete

        # generation loss
        generation_terms = []
        generation_weights = []
        if has_complete:
            gen_loss_complete = self.loss_functions['generation'](reconstructed, origins[complete])
            generation_terms.append(gen_loss_complete)
            generation_weights.append(complete.numel())
        if has_incomplete:
            gen_loss_incomplete = self.loss_functions['masked_generation'](
                pred=generated[incomplete],
                target=origins[incomplete],
                mask=available[incomplete]
            )
            generation_terms.append(gen_loss_incomplete)
            generation_weights.append(incomplete.numel())
        if generation_terms:
            weights = torch.tensor(generation_weights, dtype=origins.dtype, device=origins.device)
            generation_loss = sum(w * l for w, l in zip(weights, generation_terms)) / weights.sum()
        else:
            generation_loss = self.loss_functions['generation'](reconstructed, origins)

        # contrastive loss
        contrastive_loss = origins.new_zeros(())
        if complete.numel() >= 2:
            contrastive_loss = self.loss_functions['contrastive'](projections)

        # diagnosis loss
        if has_selected:
            diagnosis_labels = torch.cat([
                labels[complete].repeat(2, ), labels
            ], dim=0)
        else:
            diagnosis_labels = labels
        diagnosis_loss = self.loss_functions['diagnosis'](diagnoses, diagnosis_labels)

        # weighted base loss
        loss_items = torch.stack([contrastive_loss, generation_loss, diagnosis_loss])
        base_loss = self.loss_functions['base'](loss_items)

        return {
            'loss_contrastive': contrastive_loss,
            'loss_generation': generation_loss,
            'loss_diagnosis': diagnosis_loss,
            'loss_base': base_loss
        }

    @torch.no_grad()
    def validate(self, loader_name: str, epoch: int) -> dict[str, float]:
        self.model.toggle_stage('test')

        total_loss = 0.0
        total_num = 0
        correct = 0

        if loader_name not in self.dataloaders:
            raise ValueError(f'invalid loader: {loader_name}')

        for batch_idx, batch in enumerate(self.dataloaders[loader_name]):
            batch = self.move_to_device(batch)

            outputs = self.forward_batch(batch)
            loss = self.compute_loss(outputs, batch)

            batch_size = batch['label'].size(0)
            total_loss += loss['loss_base'].item() * batch_size
            total_num += batch_size
            correct += (outputs[-1][-batch_size:, :].argmax(dim=-1) == batch['label']).sum().item()

        return {
            'val_loss': total_loss / max(total_num, 1),
            'val_acc': correct / max(total_num, 1)
        }

    @torch.no_grad()
    # def test(self, epoch: int) -> dict[str, float]:
    def test(self) -> dict[str, float]:
        self.load_checkpoint('best.pt')

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
                # stream compression: write directly to a temporary file
                with self.compressor.stream_writer(tmp) as compressor_stream:
                    torch.save(ckpt, compressor_stream)
                # torch.save(ckpt, tmp_name)

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
            print(f'checkpoint not found: {path}')
            return

        with open(path, 'rb') as f:
            # check whether it is in zstd compression format
            header = f.read(4)
            f.seek(0)

            if header == b'\x28\xb5\x2f\xfd':  # zstd magic number
                decompressed_data = self.decompressor.decompress(f.read())
                buffer = io.BytesIO(decompressed_data)
                ckpt = torch.load(buffer, map_location=self.device)
            else:
                # old format, load directly
                ckpt = torch.load(f, map_location=self.device)
        # ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
