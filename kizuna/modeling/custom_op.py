import os

import torch.nn as nn


class CustomOp(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def dispatch_forward(self):
        if os.environ.get("KIZUNA_USE_PYTORCH", "0") == "1":
            return self.forward_native
        else:
            return self.forward_cuda
