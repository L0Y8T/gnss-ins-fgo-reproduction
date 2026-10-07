"""robust.py —— 非高斯噪声模型 与 robust kernel（M-estimator）加权

实验 A 的两件工具：

① 噪声模型：把 GNSS 噪声从高斯换成拉普拉斯 / Student-t / 污染高斯。
   所有模型都归一化到 *相同的方差* sigma^2，
   因此实验只改变噪声的「形状」，不改变「大小」——
   这样才能把「非高斯性」从「噪声变大了」里分离出来。

② robust kernel：Huber / Cauchy 的 IRLS 权重。
   权重作用在 *整个因子* 上（二维残差的联合范数），不是逐分量。
"""

import math

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── 噪声模型 ──────────────────────────────────────────────────

def gaussian(rng, sigma, **kw):
    """高斯。方差 = sigma^2。"""
    return rng.gauss(0.0, sigma)

def laplace(rng, sigma, **kw):
    """拉普拉斯（双指数）。方差 = 2b^2，故 b = sigma/sqrt(2)。"""
    b = sigma / math.sqrt(2.0)
    u = rng.random() - 0.5
    s = 1.0 if u >= 0.0 else -1.0
    return -b * s * math.log(1.0 - 2.0 * abs(u))

def student_t(rng, sigma, nu=3, **kw):
    """Student-t。方差 = nu/(nu-2)，据此缩放到 sigma^2。nu 取整数。"""
    z = rng.gauss(0.0, 1.0)
    chi2 = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(nu))
    t = z / math.sqrt(chi2 / float(nu))
    return t * sigma / math.sqrt(nu / (nu - 2.0))

def contaminated(rng, sigma, eps=0.10, kappa=3.0, **kw):
    """污染高斯：(1-eps)·N(0, s1^2) + eps·N(0, (kappa·s1)^2)。

    总方差 = s1^2·(1 - eps + eps·kappa^2)，据此反解 s1 使总方差 = sigma^2。
    eps 是离群比例，kappa 是离群点的放大倍数。
    """
    s1 = sigma / math.sqrt(1.0 - eps + eps * kappa ** 2)
    s = s1 * (kappa if rng.random() < eps else 1.0)
    return rng.gauss(0.0, s)

MODELS = {
    "gaussian": gaussian,
    "laplace": laplace,
    "student_t3": student_t,
    "contaminated": contaminated,
}

MODEL_LABEL = {
    "gaussian": "高斯（基准）",
    "laplace": "拉普拉斯（重尾）",
    "student_t3": "Student-t (ν=3)",
    "contaminated": "污染高斯 (ε=0.1, κ=3)",
}

# ── robust kernel ─────────────────────────────────────────────

def huber_weight(s, delta):
    """Huber 的 IRLS 权重。s 是归一化残差范数 ||r||/sigma。"""
    return 1.0 if s <= delta else delta / s

def cauchy_weight(s, delta):
    """Cauchy 的 IRLS 权重。"""
    return 1.0 / (1.0 + (s / delta) ** 2)

KERNELS = {
    "l2": None,
    "huber": huber_weight,
    "cauchy": cauchy_weight,
}

KERNEL_LABEL = {
    "l2": "L2（无 kernel）",
    "huber": "Huber",
    "cauchy": "Cauchy",
}

def kernel_weight(residuals, base_weight, kernel="l2", delta=2.0):
    """由一个 *因子* 的全部残差分量算 robust 权重。

    base_weight = 1/sigma^2，所以 s = sqrt(base_weight * sum(r_i^2)) 就是
    归一化残差范数（高斯下 s^2 ~ chi2(dim)）。整组共用同一个权重，
    避免离群只压住某一个分量、另一个分量照旧把解拖走。
    """
    fn = KERNELS[kernel]
    if fn is None:
        return base_weight
    s2 = base_weight * sum(r * r for r in residuals)
    s = math.sqrt(s2)
    if s <= 0.0:
        return base_weight
    return base_weight * fn(s, delta)

# ── 可替换噪声模型的测量生成 ──────────────────────────────────

def _rng(seed, salt=0):
    import random
    return random.Random((seed * 1000003 + salt) % (2 ** 31))

def make_measurements_noise(cfg, states, model="gaussian", noise_seed=None,
                            outlier=None, **kw):
    """与 sim.make_measurements 同构，但 GNSS 噪声模型与离群点机制都可替换。

    outlier=None 时沿用 cfg 的设定；置 False 得到「纯高斯」对照组。
    model="gaussian" 且 outlier 沿用 cfg 时，结果与 sim.make_measurements 逐位相同。
    """
    fn = MODELS[model]
    seed = cfg.seed if noise_seed is None else noise_seed
    rg = _rng(seed, salt=2)
    ri = _rng(seed, salt=3)
    ro = _rng(seed, salt=4)

    if outlier is None:
        outlier = cfg.gnss_outlier_ratio > 0.0

    gnss, imu, flags = [], [], []
    for s in states:

        is_out = ro.random() < cfg.gnss_outlier_ratio if outlier else False
        flags.append(is_out)
        bias = cfg.gnss_outlier_bias if is_out else 0.0

        zx = s["p"][0] + fn(rg, cfg.gnss_sigma, **kw) + bias * 0.6
        zy = s["p"][1] + fn(rg, cfg.gnss_sigma, **kw) + bias * 0.8
        gnss.append([zx, zy])
        imu.append([s["a"][0] + ri.gauss(0.0, cfg.imu_accel_sigma),
                    s["a"][1] + ri.gauss(0.0, cfg.imu_accel_sigma)])

    return {"gnss": gnss, "imu": imu, "outlier": flags,
            "gnss_sigma_used": [cfg.gnss_sigma] * len(states)}

def effective_sigma(cfg, model="gaussian", outlier=None, **kw):
    """该噪声配置下的 *总体* 标准差（含离群点贡献）。"""
    if outlier is None:
        outlier = cfg.gnss_outlier_ratio > 0.0
    base = cfg.gnss_sigma
    if outlier:
        p = cfg.gnss_outlier_ratio
        b = cfg.gnss_outlier_bias
        var = (1 - p) * base ** 2 + p * (base ** 2 + b ** 2 * (0.6 ** 2 + 0.8 ** 2) / 2.0)
        return math.sqrt(var)
    return base

def empirical_sigma(values):
    n = len(values)
    mu = sum(values) / n
    return (sum((v - mu) ** 2 for v in values) / (n - 1)) ** 0.5

def sample_kurtosis(values):
    """超额峰度。高斯 = 0；越大尾巴越重。"""
    n = len(values)
    mu = sum(values) / n
    m2 = sum((v - mu) ** 2 for v in values) / n
    m4 = sum((v - mu) ** 4 for v in values) / n
    return m4 / (m2 * m2) - 3.0

if __name__ == "__main__":
    N = 400000
    sigma = 6.0
    print("=" * 74)
    print("噪声模型自检：是否都归一化到同一方差 sigma = %.1f" % sigma)
    print("=" * 74)
    print(f"  {'模型':<26} {'样本标准差':>12} {'相对偏差':>10} {'超额峰度':>10}")
    print("  " + "-" * 66)
    for name in MODELS:
        rng = _rng(12345, salt=2)
        xs = [MODELS[name](rng, sigma) for _ in range(N)]
        sd = empirical_sigma(xs)
        print(f"  {name:<26} {sd:12.4f} {sd/sigma - 1:10.2%} "
              f"{sample_kurtosis(xs):10.3f}")
    print()
    print("  理论超额峰度：高斯 0 ｜ 拉普拉斯 3 ｜ Student-t(3) 无穷 ｜ 污染高斯 ≈ 5.1")
    print()
    print("=" * 74)
    print("一致性自检：model='gaussian' 且沿用 cfg 的离群点设定")
    print("           应与 sim.make_measurements 逐位相同")
    print("=" * 74)
    from sim import Config, true_trajectory, make_measurements
    cfg = Config()
    st = true_trajectory(cfg)
    a = make_measurements(cfg, st)
    b = make_measurements_noise(cfg, st, model="gaussian")
    dg = max(abs(a["gnss"][k][i] - b["gnss"][k][i])
             for k in range(cfg.n_epochs) for i in (0, 1))
    di = max(abs(a["imu"][k][i] - b["imu"][k][i])
             for k in range(cfg.n_epochs) for i in (0, 1))
    do = sum(1 for k in range(cfg.n_epochs) if a["outlier"][k] != b["outlier"][k])
    print(f"  GNSS 最大差异 = {dg:.3e} m")
    print(f"  IMU  最大差异 = {di:.3e} m/s^2")
    print(f"  离群标记不一致数 = {do}")
    print("  " + ("通过：两条路径完全一致。" if dg == 0.0 and di == 0.0 and do == 0
                  else "注意：不一致，需排查。"))
    print()
    print("=" * 74)
    print("原版仿真的实际噪声水平")
    print("=" * 74)
    e = [a["gnss"][k][0] - st[k]["p"][0] for k in range(cfg.n_epochs)]
    f = [a["gnss"][k][1] - st[k]["p"][1] for k in range(cfg.n_epochs)]
    print(f"  估计器假定   sigma = {cfg.gnss_sigma:.2f} m")
    print(f"  实测 E 向    sigma = {empirical_sigma(e):.2f} m")
    print(f"  实测 N 向    sigma = {empirical_sigma(f):.2f} m")
    print(f"  两向合并     sigma = {empirical_sigma(e + f):.2f} m")
    print(f"  含离群点的理论 sigma = {effective_sigma(cfg):.2f} m")
    print(f"  离群比例     {cfg.gnss_outlier_ratio:.0%}，偏置 {cfg.gnss_outlier_bias:.0f} m")
    print()
    print("  -> 原版仿真的 GNSS 噪声本来就不是纯高斯：")
    print("     它 = 88% N(0, 6^2) + 12% [N(0, 6^2) + 确定性偏置 (15, 20)]。")
    print("     而估计器一直按 R = 6^2 加权，等于把噪声低估了约 %.1f 倍。"
          % (effective_sigma(cfg) / cfg.gnss_sigma))
