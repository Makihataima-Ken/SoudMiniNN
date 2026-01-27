import numpy as np # type: ignore
from ..core.module import Module
from ..core.parameter import Parameter
from ..utils.im2col import im2col, col2im

class Conv2D(Module):
    """
    Educational 2D convolution layer (NCHW), similar in spirit to torch.nn.Conv2d.

    Forward:
        x: (N, C_in, H, W)
        W: (C_out, C_in, kH, kW)
        b: (C_out,)
        out: (N, C_out, H_out, W_out)

    Backward:
        computes gradients for W, b and returns dx.

    Notes:
    - This implementation uses im2col/col2im for clarity.
    - It is not meant to be the fastest possible implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        init: str = "kaiming"
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kH, kW = kernel_size, kernel_size
        else:
            kH, kW = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH, self.kW = kH, kW
        self.stride = int(stride)
        self.padding = int(padding)
        self.use_bias = bool(bias)

        # Kaiming/He init for conv: std = sqrt(2/fan_in)
        fan_in = in_channels * kH * kW
        if init.lower() in ("kaiming", "he"):
            scale = np.sqrt(2.0 / fan_in)
        elif init.lower() in ("xavier", "glorot"):
            fan_out = out_channels * kH * kW
            scale = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            scale = 0.01

        W = (np.random.randn(out_channels, in_channels, kH, kW) * scale).astype(np.float32)
        self.W = Parameter(W)

        if self.use_bias:
            b = np.zeros((out_channels,), dtype=np.float32)
            self.b = Parameter(b)
        else:
            self.b = None

        # cache
        self._x_shape = None
        self._col = None
        self._W_col = None
        self._out_hw = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects x with shape (N,C,H,W), got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Conv2D expected in_channels={self.in_channels}, got {x.shape[1]}")


        col, out_h, out_w = im2col(x, self.kH, self.kW, stride=self.stride, padding=self.padding)
        W_col = self.W.data.reshape(self.out_channels, -1)

        out = col @ W_col.T  # (N*out_h*out_w, C_out)
        if self.use_bias and self.b is not None:
            out += self.b.data.reshape(1, -1)

        out = out.reshape(x.shape[0], out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        # cache
        self._x_shape = x.shape
        self._col = col
        self._W_col = W_col
        self._out_hw = (out_h, out_w)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._x_shape is None or self._col is None or self._W_col is None or self._out_hw is None:
            raise RuntimeError("Conv2D.backward called before forward")

        N, C_out, out_h, out_w = dout.shape
        dout_2d = dout.transpose(0, 2, 3, 1).reshape(-1, C_out)  # (N*out_h*out_w, C_out)

        # grads
        dW_col = dout_2d.T @ self._col  # (C_out, C_in*kH*kW)
        self.W.grad = dW_col.reshape(self.W.data.shape)

        if self.use_bias and self.b is not None:
            self.b.grad = np.sum(dout_2d, axis=0)

        dcol = dout_2d @ self._W_col  # (N*out_h*out_w, C_in*kH*kW)
        dx = col2im(dcol, self._x_shape, self.kH, self.kW, stride=self.stride, padding=self.padding)
        return dx
