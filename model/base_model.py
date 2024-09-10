import torch
from model.modules import PreNormLayer, PreNormException

class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-normalizing methods.
    """

    def pre_norm_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_norm_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_norm(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True