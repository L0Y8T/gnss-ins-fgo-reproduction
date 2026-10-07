"""run_all.py —— 一键跑完全部实验并把输出存成日志

用法：python run_all.py
"""

import io
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RES, exist_ok=True)
LOG = os.path.join(RES, "experiment_log.txt")

sys.path.insert(0, HERE)

class Tee:
    """把输出同时送到屏幕和日志文件。"""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", newline="\n")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        self.f.close()

def main():
    tee = Tee(LOG)
    old = sys.stdout
    sys.stdout = tee
    t0 = time.time()
    try:
        print("=" * 78)
        print("  Wen et al. (2021) 第 4.3 / 4.4 节消融实验 —— 完整复现")
        print(f"  时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 78)

        print()
        print("【步骤 1/4】自检线性代数模块")
        import la
        A = [[2.0, 1.0], [1.0, 3.0]]
        x = la.flatten_col(la.gauss_solve(A, la.col([5.0, 10.0])))
        assert abs(x[0] - 1.0) < 1e-12 and abs(x[1] - 3.0) < 1e-12, x
        print("  gauss_solve 自检通过（解 [1.0, 3.0]）")

        print()
        print("【步骤 2/4】自检数据生成器")
        import sim
        cfg = sim.Config()
        st = sim.true_trajectory(cfg)
        speeds = [(s["v"][0] ** 2 + s["v"][1] ** 2) ** 0.5 for s in st]
        print(f"  速度范围 {min(speeds):.1f} ~ {max(speeds):.1f} m/s")
        assert min(speeds) >= 0 and max(speeds) < 80, "速度不物理，检查加速度模型"
        print("  轨迹物理合理")

        print()
        print("【步骤 3/4】运行四组实验（这一步耗时最长）")
        import experiment
        experiment.exp1_iterations()
        experiment.exp2_window()
        experiment.exp3_drift()
        experiment.exp4_sensitivity()

        print()
        print("【步骤 4/4】生成图表")
        import make_figures
        make_figures.fig_window()
        make_figures.fig_iterations()
        make_figures.fig_drift()
        make_figures.fig_sensitivity()

        print()
        print("=" * 78)
        print(f"  全部完成，耗时 {time.time()-t0:.1f} 秒")
        print(f"  结果目录：{os.path.relpath(RES, HERE)}")
        print("=" * 78)
    finally:
        sys.stdout = old
        tee.close()

if __name__ == "__main__":
    main()

