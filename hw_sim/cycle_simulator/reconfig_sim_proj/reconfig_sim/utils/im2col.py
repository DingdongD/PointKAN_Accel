# -*- coding: utf-8 -*-
"""
im2col：将 NCHW 输入按卷积核展开为列矩阵，并完成 Conv2D -> GEMM 的等价计算，供仿真器分块。
"""
import numpy as np


def im2col_nchw(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, padding: int = 0):
    """
    将输入 x [N,C,H,W] 按滑动窗展开为 cols [N*OH*OW, C*k_h*k_w]。
    返回 cols 与输出高宽 oh, ow。
    """
    n, c, h, w = x.shape
    h_p = h + 2 * padding
    w_p = w + 2 * padding
    x_pad = np.zeros((n, c, h_p, w_p), dtype=x.dtype)
    x_pad[:, :, padding:padding+h, padding:padding+w] = x
    oh = (h + 2 * padding - k_h) // stride + 1
    ow = (w + 2 * padding - k_w) // stride + 1
    cols = np.zeros((n * oh * ow, c * k_h * k_w), dtype=x.dtype)
    row = 0
    for ni in range(n):
        for oy in range(oh):
            for ox in range(ow):
                patch = x_pad[ni, :, oy*stride:oy*stride+k_h, ox*stride:ox*stride+k_w]
                cols[row, :] = patch.reshape(-1)
                row += 1
    return cols, oh, ow


def conv2d_to_gemm(x: np.ndarray, wgt: np.ndarray, bias=None, stride: int = 1, padding: int = 0):
    """
    将 Conv2D 转为 GEMM：im2col 得到 cols，权重展成 w_mat [K, Cout]，y = cols @ w_mat + bias。
    返回 (cols, w_mat, y)，y 为 NCHW 形状 [N, C_out, OH, OW]。
    """
    cols, oh, ow = im2col_nchw(x, wgt.shape[2], wgt.shape[3], stride=stride, padding=padding)
    w_mat = wgt.reshape(wgt.shape[0], -1).T  # [K, Cout]
    y = cols.dot(w_mat)
    if bias is not None:
        y = y + bias.reshape(1, -1)
    return cols, w_mat, y.reshape(x.shape[0], oh, ow, wgt.shape[0]).transpose(0, 3, 1, 2)
