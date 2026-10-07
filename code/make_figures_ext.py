"""make_figures_ext.py —— 画扩展实验（A / B）的图

运行：python make_figures_ext.py
输出：../results/figA1..figB2_*.svg
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

NOISE_LABEL = {
    "gaussian": "纯高斯\nσ=6.0",
    "laplace": "拉普拉斯\nσ=6.0",
    "student_t3": "Student-t(3)\nσ=6.0",
    "contaminated": "污染高斯\nσ=6.0",
    "original": "原版含离群\nσ=8.6",
}

EST_ORDER = ["EKF", "L2（无 kernel）", "Huber", "Cauchy"]
EST_LABEL = {"EKF": "EKF", "L2（无 kernel）": "FGO L2",
             "Huber": "FGO Huber", "Cauchy": "FGO Cauchy"}


def read_csv(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print(f"  [警告] 找不到 {name}，跳过")
        return None
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _noise_key(row):
    return "original" if row["outlier"] == "True" else row["noise_model"]


def figA_bars(column, title, ylabel, out_name):
    rows = read_csv("expA1_noise.csv")
    if not rows:
        return
    K = 20
    keys = ["gaussian", "laplace", "student_t3", "contaminated", "original"]
    table = {}
    for r in rows:
        if r["estimator"] == "EKF":
            table.setdefault(_noise_key(r), {})["EKF"] = float(r[column])
        elif int(r["window"]) == K:
            table.setdefault(_noise_key(r), {})[r["estimator"]] = float(r[column])

    groups = []
    for k in keys:
        d = table.get(k)
        if not d:
            continue
        groups.append([NOISE_LABEL[k]] + [d.get(e, 0.0) for e in EST_ORDER])

    bar_chart(os.path.join(RES, out_name), groups,
              [EST_LABEL[e] for e in EST_ORDER],
              title, "GNSS 噪声模型（方差已对齐）", ylabel)
    print(f"  -> {out_name}")


def figB1_convergence():
    rows = read_csv("expB1_convergence.csv")
    if not rows:
        return

    def series_for(K, model, color, dash_name):
        pts = [(int(r["iteration"]), float(r["delta_norm"]))
               for r in rows if int(r["window"]) == K and r["model"] == model]
        pts.sort()
        return {"name": f"K={K}  {dash_name}", "x": [p[0] for p in pts],
                "y": [p[1] for p in pts], "color": color}

    line_chart(os.path.join(RES, "figB1_convergence.svg"),
               [series_for(10, "2D 仿射", "#1f77b4", "2D 仿射模型"),
                series_for(10, "5D 带旋转", "#d62728", "5D 带旋转模型")],
               "图 B1  同一个窗口的收敛剖面：仿射 vs 非线性（K=10）",
               "Gauss-Newton 迭代次数",
               "增量范数 |dX|（对数轴）",
               logy=True, legend_loc="upper right",
               ylim=(1e-14, 1e2),
               annotate=[
                   {"text": "2D：第 2 步已到 1e-13\n（泰勒展开 = 恒等式）",
                    "x": 2.2, "y": 1e-11},
                   {"text": "5D：每步只降 1~2 个数量级\n需 7 步才能收敛",
                    "x": 3.2, "y": 1e-1},
               ])
    print("  -> figB1_convergence.svg")


def figB2_steps_vs_K():
    rows = read_csv("expB1_convergence.csv")
    if not rows:
        return
    Ks = sorted({int(r["window"]) for r in rows})
    out = []
    for model, color in (("2D 仿射", "#1f77b4"), ("5D 带旋转", "#d62728")):
        ys = []
        for K in Ks:
            steps = max(int(r["iteration"]) for r in rows
                        if int(r["window"]) == K and r["model"] == model)
            ys.append(steps)
        out.append({"name": model, "x": Ks, "y": ys, "color": color})

    line_chart(os.path.join(RES, "figB2_steps_vs_K.svg"), out,
               "图 B2  收敛所需迭代步数 vs 窗口大小",
               "窗口大小 K（历元数）", "收敛所需迭代步数",
               legend_loc="upper left")
    print("  -> figB2_steps_vs_K.svg")


def figB3_perturb():
    rows = read_csv("expB2_perturbed_init.csv")
    if not rows:
        return
    out = []
    for model, color in (("2D", "#1f77b4"), ("5D", "#d62728")):
        sel = [r for r in rows if r["model"] == model and float(r["perturb_vel_mps"]) == 0.0]
        sel.sort(key=lambda r: float(r["perturb_yaw_deg"]))
        out.append({"name": model + " 模型",
                    "x": [float(r["perturb_yaw_deg"]) for r in sel],
                    "y": [float(r["steps_to_converge"]) for r in sel],
                    "color": color})

    line_chart(os.path.join(RES, "figB3_perturb_vs_steps.svg"), out,
               "图 B3  初始偏航猜测越差，需要几步收敛",
               "初始偏航猜测偏差（度）", "收敛所需迭代步数",
               legend_loc="upper left")
    print("  -> figB3_perturb_vs_steps.svg")


if __name__ == "__main__":
    print("生成扩展实验图：")
    figA_bars("mean_m", "图 A1  非高斯噪声下各估计器的平均定位误差（K=20，6 个种子平均）",
              "二维水平平均误差 (m)", "figA1_noise_mean.svg")
    figA_bars("p95_m", "图 A2  非高斯噪声下各估计器的 P95 误差（K=20）",
              "二维水平误差 P95 (m)", "figA2_noise_p95.svg")
    figA_bars("cat_rate_gt15m", "图 A3  误差超过 15 m 的历元占比（K=20）",
              "超 15 m 历元占比", "figA3_noise_tail_rate.svg")
    figB1_convergence()
    figB2_steps_vs_K()
    figB3_perturb()
    print("完成。")
