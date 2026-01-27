import numpy as np
from ..core.module import Module
from ..utils.im2col import im2col, col2im

class MaxPool2D(Module):
    """
    Educational MaxPool2D for NCHW input.
    """
    def __init__(self, kernel_size: int | tuple[int, int], stride: int | None = None, padding: int = 0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.pH, self.pW = kernel_size, kernel_size
        else:
            self.pH, self.pW = kernel_size
        self.stride = int(stride) if stride is not None else self.pH
        self.padding = int(padding)

        self._x_shape = None
        self._col = None
        self._argmax = None
        self._out_hw = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"MaxPool2D expects x with shape (N,C,H,W), got {x.shape}")
        x = x.astype(np.float32, copy=False)
        N, C, H, W = x.shape

        # Pool per channel: treat N*C as batch and channel=1
        x_ = x.reshape(N * C, 1, H, W)
        col, out_h, out_w = im2col(x_, self.pH, self.pW, stride=self.stride, padding=self.padding)
        # col: (N*C*out_h*out_w, pH*pW)
        argmax = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, C, out_h, out_w)

        self._x_shape = x.shape
        self._col = col
        self._argmax = argmax
        self._out_hw = (out_h, out_w)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._x_shape is None or self._col is None or self._argmax is None or self._out_hw is None:
            raise RuntimeError("MaxPool2D.backward called before forward")

        N, C, H, W = self._x_shape
        out_h, out_w = self._out_hw

        dflat = dout.reshape(N * C * out_h * out_w)
        dcol = np.zeros_like(self._col)
        rows = np.arange(self._argmax.size)
        dcol[rows, self._argmax] = dflat

        dx_ = col2im(dcol, (N * C, 1, H, W), self.pH, self.pW, stride=self.stride, padding=self.padding)
        dx = dx_.reshape(N, C, H, W)
        return dx


class AvgPool2D(Module):
    """
    Educational AvgPool2D for NCHW input.
    """
    def __init__(self, kernel_size: int | tuple[int, int], stride: int | None = None, padding: int = 0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.pH, self.pW = kernel_size, kernel_size
        else:
            self.pH, self.pW = kernel_size
        self.stride = int(stride) if stride is not None else self.pH
        self.padding = int(padding)

        self._x_shape = None
        self._col = None
        self._out_hw = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"AvgPool2D expects x with shape (N,C,H,W), got {x.shape}")
        x = x.astype(np.float32, copy=False)
        N, C, H, W = x.shape

        x_ = x.reshape(N * C, 1, H, W)
        col, out_h, out_w = im2col(x_, self.pH, self.pW, stride=self.stride, padding=self.padding)

        out = np.mean(col, axis=1)
        out = out.reshape(N, C, out_h, out_w)

        self._x_shape = x.shape
        self._col = col
        self._out_hw = (out_h, out_w)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._x_shape is None or self._col is None or self._out_hw is None:
            raise RuntimeError("AvgPool2D.backward called before forward")

        N, C, H, W = self._x_shape
        out_h, out_w = self._out_hw

        dflat = dout.reshape(N * C * out_h * out_w)
        pool_area = self.pH * self.pW
        dcol = np.repeat((dflat / pool_area)[:, None], pool_area, axis=1).astype(np.float32)

        dx_ = col2im(dcol, (N * C, 1, H, W), self.pH, self.pW, stride=self.stride, padding=self.padding)
        dx = dx_.reshape(N, C, H, W)
        return dx
