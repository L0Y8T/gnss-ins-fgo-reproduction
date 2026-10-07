"""make_figures.py —— 读 results/ 下的 CSV，画出四张图

运行：python make_figures.py
输出：../results/fig1..fig4_*.svg
"""

import os
import sys
import csv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from plot import line_chart, bar_chart

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

def read_csv(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print(f"  [警告] 找不到 {name}，跳过")
        return None
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows

def fig_window():
    """图 1：窗口大小 vs 精度（对应论文图 8）"""

    rows = read_csv("exp2_window.csv")
    if not rows:
        return
    K = [int(r["window"]) for r in rows]
    mean = [float(r["mean_m"]) for r in rows]

    ekf_ref = None
    r4 = read_csv("exp4_sensitivity.csv")
    if r4:
        ekf_ref = float(r4[0]["ekf_mean_m"]) if "ekf_mean_m" in r4[0] else None

    hl = [{"y": ekf_ref, "label": f"EKF 基准 ({ekf_ref:.2f} m)",
           "color": "#7f7f7f"}] if ekf_ref else None

    line_chart(os.path.join(RES, "fig1_window_vs_accuracy.svg"),
               [{"name": "FGO（滑动窗口批优化）", "x": K, "y": mean,
                 "color": "#1f77b4"}],
               "图 1  定位精度 vs 窗口大小   （对应 Wen 2021 图 8）",
               "窗口大小 K（历元数，GNSS 1 Hz → 1 历元 = 1 秒）",
               "二维水平平均误差 (m)",
               hlines=hl,
               annotate=[
                   {"text": "K=1：类 EKF 估计器\n（只保留多轮迭代）", "x": 1, "y": mean[0]},
                   {"text": "K=150：≈全量批优化", "x": 150, "y": mean[-1]},
               ])
    print("  图 1 已生成：fig1_window_vs_accuracy.svg")

def fig_iterations():
    """图 2：迭代次数消融"""
    rows = read_csv("exp1_iterations.csv")
    if not rows:
        return

    from collections import defaultdict
    by_w = defaultdict(list)
    for r in rows:
        by_w[int(r["window"])].append((int(r["n_iter"]), float(r["mean_m"])))

    series = []
    for i, w in enumerate(sorted(by_w)):
        pts = sorted(by_w[w])
        series.append({"name": f"窗口 K={w}", "x": [p[0] for p in pts],
                       "y": [p[1] for p in pts]})

    line_chart(os.path.join(RES, "fig2_iterations.svg"), series,
               "图 2  迭代次数消融：迭代次数对精度没有影响",
               "Gauss-Newton 迭代次数", "二维水平平均误差 (m)")
    print("  图 2 已生成：fig2_iterations.svg")

def fig_drift():
    """图 3：噪声突变（论文 4.4 节反例）"""
    rows = read_csv("exp3_drift.csv")
    if not rows:
        return
    K = [int(r["window"]) for r in rows]
    before = [float(r["mean_before_m"]) for r in rows]
    after = [float(r["mean_after_m"]) for r in rows]
    near = [float(r["mean_D_pm10_m"]) for r in rows]

    line_chart(os.path.join(RES, "fig3_noise_regime_shift.svg"),
               [{"name": "噪声突变前 (120~180 历元)", "x": K, "y": before},
                {"name": "噪声突变后 (180~240 历元)", "x": K, "y": after},
                {"name": "突变点±10 历元", "x": K, "y": near,
                 "color": "#d62728"}],
               "图 3  GNSS 噪声特性突变时的表现   （对应 Wen 2021 4.4 节）",
               "窗口大小 K（历元数）", "二维水平平均误差 (m)")
    print("  图 3 已生成：fig3_noise_regime_shift.svg")

def fig_sensitivity():
    """图 4：参数敏感性 EKF vs FGO"""
    rows = read_csv("exp4_sensitivity.csv")
    if not rows:
        return
    sv = [float(r["sigma_v"]) for r in rows]
    ekf = [float(r["ekf_mean_m"]) for r in rows]
    series = [{"name": "EKF（对参数不敏感）", "x": sv, "y": ekf,
               "color": "#7f7f7f", "dash": "8,5"}]
    for i, K in enumerate([5, 10, 30, 150]):
        key = f"fgo_K{K}_mean_m"
        if key in rows[0]:
            series.append({"name": f"FGO K={K}", "x": sv,
                           "y": [float(r[key]) for r in rows]})

    line_chart(os.path.join(RES, "fig4_param_sensitivity.svg"), series,
               "图 4  噪声协方差参数敏感性：EKF 稳健 vs FGO 敏感",
               "估计器假定的 IMU 因子标准差 σ_v (m/s)",
               "二维水平平均误差 (m)")
    print("  图 4 已生成：fig4_param_sensitivity.svg")

if __name__ == "__main__":
    print("=" * 70)
    print("生成图表（SVG，用浏览器打开）")
    print("=" * 70)
    fig_window()
    fig_iterations()
    fig_drift()
    fig_sensitivity()
    print("=" * 70)
    print(f"全部图在 {os.path.relpath(RES, HERE)} 目录下。")

