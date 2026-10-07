"""la.py —— 最小线性代数库（纯 Python 标准库实现）

刻意不使用 numpy：手写矩阵运算与方程组求解，便于逐项追查中间量。
主要接口 gauss_solve / weighted_normal_equations。
"""

import sys
from math import isfinite

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def zeros(m, n):
    return [[0.0] * n for _ in range(m)]

def identity(n):
    """生成 n×n 单位阵（对角线为 1，其余为 0）。"""

    M = zeros(n, n)
    for i in range(n):
        M[i][i] = 1.0
    return M

def diag(vals):
    """由一维序列生成对角阵。diag([a, b]) -> [[a,0],[0,b]]"""
    n = len(vals)
    M = zeros(n, n)
    for i in range(n):
        M[i][i] = float(vals[i])
    return M

def shape(A):
    return (len(A), len(A[0]) if A else 0)

def col(vals):
    return [[float(v)] for v in vals]

def flatten_col(A):
    return [A[i][0] for i in range(len(A))]

def copy(A):
    return [row[:] for row in A]

def add(A, B):
    m, n = shape(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]

def sub(A, B):
    m, n = shape(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(m)]

def scale(A, s):
    m, n = shape(A)
    return [[A[i][j] * s for j in range(n)] for i in range(m)]

def matmul(A, B):
    """矩阵乘法 A·B（复杂度 O(m·n·p)，本实验规模下完全够用）。"""
    m, n = shape(A)
    n2, p = shape(B)
    assert n == n2, f"维度不匹配: {shape(A)} x {shape(B)}"
    C = zeros(m, p)
    for i in range(m):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                Ci[j] += a * Bk[j]
    return C

def transpose(A):
    m, n = shape(A)
    return [[A[i][j] for i in range(m)] for j in range(n)]

def matvec(A, x):
    """矩阵乘向量：A(列向量) -> 列向量。"""
    m, n = shape(A)
    assert n == len(x), f"维度不匹配: {shape(A)} x {len(x)}"
    return col([sum(A[i][k] * x[k] for k in range(n)) for i in range(m)])

def gauss_solve(A, b, eps=None):
    """解线性方程组 A x = b，用「列主元高斯消元法」。"""
    n = len(A)
    M = [A[i][:] + [b[i][0]] for i in range(n)]

    if eps is None:
        amax = 0.0
        for row in A:
            for x in row:
                if abs(x) > amax:
                    amax = abs(x)
        eps = 1e-10 * amax if amax > 0 else 1e-300

    for k in range(n):

        piv = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[piv][k]):
                piv = i
        if abs(M[piv][k]) < eps:
            raise ValueError(
                f"矩阵奇异（第 {k} 列主元 {abs(M[piv][k]):.3e} < 阈值 {eps:.3e}）。"
                "常见原因：窗口太小导致状态不可观，或测量不足。"
            )
        if piv != k:
            M[k], M[piv] = M[piv], M[k]

        pivval = M[k][k]
        for i in range(k + 1, n):
            factor = M[i][k] / pivval
            if factor == 0.0:
                continue
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]

    return col(x)

def inverse(A):
    """矩阵求逆（用高斯-若尔当法，逐列解 A x = e_i）。本实验只用于诊断。"""
    n = len(A)
    inv = zeros(n, n)
    for j in range(n):
        e = col([1.0 if i == j else 0.0 for i in range(n)])
        xj = gauss_solve(A, e)
        for i in range(n):
            inv[i][j] = xj[i][0]
    return inv

def weighted_normal_equations(H, W, r):
    """构造并求解 Gauss-Newton 正规方程，返回增量 dx。"""
    m, n = shape(H)

    if isinstance(W, list) and W and not isinstance(W[0], list):
        wdiag = W
    else:
        wdiag = [W[i][i] for i in range(m)]

    sw = [w ** 0.5 for w in wdiag]
    Hw = [[H[i][j] * sw[i] for j in range(n)] for i in range(m)]
    rw = [[r[i][0] * sw[i]] for i in range(m)]

    A = zeros(n, n)
    g = zeros(n, 1)
    for i in range(m):
        Hi = Hw[i]
        ri = rw[i][0]
        for a in range(n):
            ha = Hi[a]
            if ha == 0.0:
                continue
            g[a][0] += ha * ri
            for b in range(a, n):
                A[a][b] += ha * Hi[b]

    for a in range(n):
        for b in range(a + 1, n):
            A[b][a] = A[a][b]

    neg_g = scale(g, -1.0)
    return gauss_solve(A, neg_g)

def norm2(v):
    if isinstance(v, list) and v and isinstance(v[0], list):
        v = flatten_col(v)
    return sum(x * x for x in v) ** 0.5

def is_finite_matrix(A):
    for row in A:
        for x in row:
            if not isfinite(x):
                return False
    return True

def fmt(A, prec=4):
    m, n = shape(A)
    return "\n".join(
        "[" + ", ".join(f"{A[i][j]:.{prec}f}" for j in range(n)) + "]" for i in range(m)
    )

if __name__ == "__main__":

    A = [[2.0, 1.0], [1.0, 3.0]]
    b = col([5.0, 10.0])
    x = gauss_solve(A, b)
    print("解 A x = b  ->", flatten_col(x), "  （预期 [1.0, 3.0]）")

    inv = inverse(A)
    prod = matmul(A, inv)
    print("A·A⁻¹ =\n" + fmt(prod), "  （预期单位阵）")

    H = col([1.0, 1.0])
    r = col([0.0 - 3.0, 0.0 - 5.0])
    dx = weighted_normal_equations(H, [1.0, 9.0], r)
    print("加权最小二乘一步 ->", round(flatten_col(dx)[0], 6), "  （预期 4.8）")

