"""
TensorBoard logger for training visualization.

Usage:
  from util.tb_logger import TBLogger
  logger = TBLogger(log_dir='./logs/refcoco')
  logger.log_scalar('loss/total', loss_value, epoch * len(dataloader) + step)
  logger.log_scalars({'loss/ce': ce_loss, 'loss/bbox': bbox_loss}, global_step)
"""

import os
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Simple wrapper around TensorBoard SummaryWriter."""

    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, global_step):
        self.writer.add_scalar(tag, value, global_step)

    def log_scalars(self, tag_dict, global_step):
        """Log multiple scalars (each as a separate plot)."""
        for tag, value in tag_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(tag, value, global_step)

    def log_text(self, tag, text, global_step):
        self.writer.add_text(tag, text, global_step)

    def log_image(self, tag, image_tensor, global_step):
        """Log an image tensor [3, H, W] or [1, 3, H, W]."""
        self.writer.add_image(tag, image_tensor, global_step)

    def log_images(self, tag, image_tensors, global_step):
        """Log multiple images with predictions."""
        for i, img in enumerate(image_tensors):
            self.writer.add_image(f"{tag}/{i}", img, global_step)

    def log_histogram(self, tag, values, global_step):
        self.writer.add_histogram(tag, values, global_step)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()

    def add_graph(self, model, input_to_model=None):
        """Add model graph for visualization."""
        if input_to_model is not None:
            self.writer.add_graph(model, input_to_model)

    def log_hparams(self, hparams, metrics):
        """Log hyperparameters and final metrics."""
        self.writer.add_hparams(hparams, metrics)

    def log_lr(self, lr, global_step):
        self.writer.add_scalar('train/lr', lr, global_step)

    def log_losses(self, loss_dict, global_step, prefix='train'):
        """Log all losses from loss_dict."""
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{prefix}/{key}', value, global_step)

    def log_epoch_metrics(self, metrics, epoch, prefix='val'):
        """Log epoch-level metrics (accuracy, mAP, etc.)."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{prefix}/{key}', value, epoch)
