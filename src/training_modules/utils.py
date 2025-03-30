import torch
import lightning

@torch.no_grad()
def get_rms_grad(model):
    rms_grad, param_count = 0.0, 0
    for param in model.parameters():
        if param.grad is not None:
            rms_grad += (param.grad**2).sum().item()
            param_count += torch.numel(param)
    rms_grad = (rms_grad / param_count)**0.5
    return rms_grad

class TimingCallback(lightning.Callback):
    def __init__(self, start_batch=10, end_batch=20):
        self.start_batch = start_batch
        self.end_batch = end_batch
        self.batch_times = []
    
    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if self.start_batch <= batch_idx < self.end_batch:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.start_batch <= batch_idx < self.end_batch:
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
            self.batch_times.append(elapsed_time_ms)