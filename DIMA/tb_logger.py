import os
import torch
import torch.utils.tensorboard as tb
from datetime import datetime

class TensorBoardLogger:
    def __init__(self):
        self.writer = None

    def initialize(self, log_dir=None):
        if log_dir is None:
            log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = tb.SummaryWriter(log_dir)
        self.step = 0

    def log_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalar_wo_step(self, tag, value):
        if self.writer:
            self.writer.add_scalar(tag, value, self.step)
            self.step += 1

    def log_histogram(self, tag, values, step):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag, text, step):
        if self.writer:
            self.writer.add_text(tag, text, step)

    def close(self):
        if self.writer:
            self.writer.close()


LOGGER = TensorBoardLogger()

