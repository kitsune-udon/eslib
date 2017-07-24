import numpy as np

def softmax(x):
    y = x - x.max()
    np.exp(y, out=y)
    y /= y.sum()
    return y

def elu(x, alpha=1.0):
    y = x.copy()
    neg_indices = x < 0
    y[neg_indices] = alpha * (np.exp(y[neg_indices]) - 1)
    return y

def selu(x,
        alpha=1.6732632423543772848170429916717,
        scale=1.0507009873554804934193349852946):
    return scale * elu(x, alpha=alpha)

def sigmoid(x):
    half = x.dtype.type(0.5)
    y = np.tanh(x * half) * half + half
    return y

def relu(x):
    return np.maximum(x, 0, dtype=x.dtype)

def _extract_gates(x):
    r = x.reshape((x.shape[0] // 4, 4) + x.shape[1:])
    return [r[:, i] for i in range(4)]

def lstm(c_prev, x):
    a, i, f, o = _extract_gates(x)
    a = np.tanh(a)
    i = sigmoid(i)
    f = sigmoid(f)
    o = sigmoid(o)
    c_next = a * i + f * c_prev
    h = o * np.tanh(c_next)
    return c_next, h
