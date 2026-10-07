import sys, math
sys.stdout.reconfigure(encoding="utf-8")
from sim import Config, true_trajectory

cfg = Config(); cfg.n_epochs=200
st = true_trajectory(cfg)
dt = cfg.dt

rp = [math.hypot(st[k]["p"][0]-(st[k-1]["p"][0]+st[k-1]["v"][0]*dt),
                 st[k]["p"][1]-(st[k-1]["p"][1]+st[k-1]["v"][1]*dt)) for k in range(1,len(st))]

rv = [math.hypot(st[k]["v"][0]-(st[k-1]["v"][0]+st[k-1]["a"][0]*dt),
                 st[k]["v"][1]-(st[k-1]["v"][1]+st[k-1]["a"][1]*dt)) for k in range(1,len(st))]
def rms(x): return (sum(v*v for v in x)/len(x))**0.5
print("运动/IMU 因子在【真值轨迹】上的残差（即模型的真实误差量级）:")
print(f"  位置残差 RMS = {rms(rp):.3f} m      最大 {max(rp):.3f} m")
print(f"  速度残差 RMS = {rms(rv):.3f} m/s    最大 {max(rv):.3f} m/s")
print()
print("由此反推正确的 σ_p 与 σ_v（应至少等于上面这两个 RMS）:")
print(f"  建议 σ_p ≈ {rms(rp):.2f} m,  σ_v ≈ {rms(rv):.2f} m/s")
print()
print("对比论文式(22)：σ_p = 0.3 m, σ_v = 0.01 m/s")
print(f"  -> 论文的 σ_p 比实际残差小 {rms(rp)/0.3:.1f} 倍")
print(f"  -> 论文的 σ_v 比实际残差小 {rms(rv)/0.01:.0f} 倍")

