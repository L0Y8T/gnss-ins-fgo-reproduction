import sys, math
sys.stdout.reconfigure(encoding="utf-8")
from sim import Config, true_trajectory, make_measurements

cfg = Config(); cfg.n_epochs=300
st = true_trajectory(cfg); ms = make_measurements(cfg, st)
dt = cfg.dt

rv = [math.hypot(st[k]["v"][0]-(st[k-1]["v"][0]+ms["imu"][k-1][0]*dt),
                 st[k]["v"][1]-(st[k-1]["v"][1]+ms["imu"][k-1][1]*dt)) for k in range(1,len(st))]
rp = [math.hypot(st[k]["p"][0]-(st[k-1]["p"][0]+st[k-1]["v"][0]*dt),
                 st[k]["p"][1]-(st[k-1]["p"][1]+st[k-1]["v"][1]*dt)) for k in range(1,len(st))]
def rms(x): return (sum(v*v for v in x)/len(x))**0.5
print("IMU 因子残差组成分解:")
print(f"  ① 真实速度增量 vs 用真值加速度预测 : {rms(rv):.4f} m/s  (含 IMU 噪声)")
print(f"  ② 运动模型位置残差                 : {rms(rp):.4f} m")
print()
print("理论值:")
print(f"  加速度计噪声 sigma={cfg.imu_accel_sigma} m/s² -> 速度增量误差 {cfg.imu_accel_sigma*dt:.3f} m/s")
print(f"  区间内加速度变化 a_rw={cfg.accel_rw_std} m/s² -> 额外贡献约 {cfg.accel_rw_std*dt/2:.3f} m/s")
print(f"  两者合成(RSS) = {math.hypot(cfg.imu_accel_sigma*dt, cfg.accel_rw_std*dt/2):.4f} m/s")
print()
print("=> 结论：IMU 因子的正确 sigma_v 应约为 0.2~0.7 m/s，")
print("   而论文式(27) 给 0.15 m/s —— 只是略紧，但配合式(22) 的 sigma_v=0.01 m/s")
print("   就整体偏紧一个数量级以上。")

