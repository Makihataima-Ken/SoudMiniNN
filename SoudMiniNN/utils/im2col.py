import numpy as np

def im2col(x: np.ndarray, kernel_h: int, kernel_w: int, stride: int = 1, padding: int = 0) -> tuple[np.ndarray, int, int]:
    """
    Convert NCHW image batch into a 2D matrix (columns) for fast conv/pool.

    Args:
        x: input array of shape (N, C, H, W)
        kernel_h, kernel_w: kernel size
        stride: stride
        padding: zero-padding on H and W

    Returns:
        col: shape (N*out_h*out_w, C*kernel_h*kernel_w)
        out_h, out_w: output spatial sizes
    """
    if x.ndim != 4:
        raise ValueError(f"im2col expects x with shape (N,C,H,W), got {x.shape}")

    N, C, H, W = x.shape
    H_p = H + 2 * padding
    W_p = W + 2 * padding

    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    else:
        x_padded = x

    out_h = (H_p - kernel_h) // stride + 1
    out_w = (W_p - kernel_w) // stride + 1

    # Collect patches
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=x.dtype)
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for xk in range(kernel_w):
            x_max = xk + stride * out_w
            col[:, :, y, xk, :, :] = x_padded[:, :, y:y_max:stride, xk:x_max:stride]

    # (N, C, kH, kW, out_h, out_w) -> (N, out_h, out_w, C, kH, kW) -> (N*out_h*out_w, C*kH*kW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col, out_h, out_w


def col2im(col: np.ndarray, x_shape: tuple[int, int, int, int], kernel_h: int, kernel_w: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Inverse of im2col for NCHW.

    Args:
        col: shape (N*out_h*out_w, C*kH*kW)
        x_shape: original input shape (N, C, H, W)
        kernel_h, kernel_w: kernel size
        stride, padding: same as used in im2col

    Returns:
        x: reconstructed gradient of shape (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_p = H + 2 * padding
    W_p = W + 2 * padding

    out_h = (H_p - kernel_h) // stride + 1
    out_w = (W_p - kernel_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H_p, W_p), dtype=col.dtype)
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for xk in range(kernel_w):
            x_max = xk + stride * out_w
            img[:, :, y:y_max:stride, xk:x_max:stride] += col[:, :, y, xk, :, :]

    if padding > 0:
        return img[:, :, padding:-padding, padding:-padding]
    return img
