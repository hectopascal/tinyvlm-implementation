"""
Process results.csv: derive config labels, compute mean/std across repeated runs,
and plot scaling curves with error bars.

Recovers config from (world_size, activation_ckpt) since the label column is empty.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "/home/ubuntu/tinyvlm-implemention/fsdp_study/results1.csv"
PLOT_PATH = "plots/scaling.png"

# ---------- load and clean ----------

df = pd.read_csv(CSV_PATH)

# Drop the broken label column
df = df.drop(columns=["label"], errors="ignore")

# Synthesize a config label from world_size + activation_ckpt
df["config"] = df.apply(
    lambda r: f"{r['world_size']}gpu_{'ckpt' if r['activation_ckpt'] else 'nockpt'}",
    axis=1,
)

print("Raw data:")
print(df.to_string(index=False))
print()

# ---------- aggregate ----------

agg = (
    df.groupby(["config", "world_size", "activation_ckpt"])
      .agg(
          tps_mean=("tokens_per_sec", "mean"),
          tps_std=("tokens_per_sec", "std"),
          tps_min=("tokens_per_sec", "min"),
          tps_max=("tokens_per_sec", "max"),
          mem_mean=("peak_mem_gb", "mean"),
          n_runs=("tokens_per_sec", "count"),
      )
      .reset_index()
      .sort_values(["activation_ckpt", "world_size"])
)
# stddev is NaN for n=1 — fill with 0 so error bars don't break
agg["tps_std"] = agg["tps_std"].fillna(0)

print("Aggregated:")
print(agg.to_string(index=False))
print()

# ---------- coefficient of variation per config ----------

print("Run-to-run variance (CoV = std/mean):")
for _, row in agg.iterrows():
    if row["n_runs"] > 1:
        cov = row["tps_std"] / row["tps_mean"] * 100
        print(f"  {row['config']:20s} n={row['n_runs']}  "
              f"mean={row['tps_mean']:.0f}  std={row['tps_std']:.1f}  "
              f"CoV={cov:.1f}%  range=[{row['tps_min']:.0f}, {row['tps_max']:.0f}]")
print()

# ---------- strong scaling efficiency ----------

print("Strong-scaling efficiency (relative to 2-GPU baseline):")
for ckpt_val in [False, True]:
    sub = agg[agg["activation_ckpt"] == ckpt_val].sort_values("world_size")
    if len(sub) == 0:
        continue
    base_row = sub[sub["world_size"] == 2]
    if len(base_row) == 0:
        continue
    base_tps = base_row["tps_mean"].iloc[0]
    print(f"  activation_ckpt={ckpt_val}:")
    for _, row in sub.iterrows():
        n = row["world_size"]
        scaling_factor = row["tps_mean"] / base_tps
        ideal_factor = n / 2
        efficiency = scaling_factor / ideal_factor * 100
        print(f"    {n} GPU: {row['tps_mean']:.0f} tok/s  "
              f"({scaling_factor:.2f}x vs 2-GPU, {efficiency:.0f}% of ideal {ideal_factor:.1f}x)")
print()

# ---------- plot ----------

Path("plots").mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel 1: throughput vs GPU count
for ckpt_val in [False, True]:
    sub = agg[agg["activation_ckpt"] == ckpt_val].sort_values("world_size")
    if len(sub) == 0:
        continue
    label = "with activation ckpt" if ckpt_val else "no activation ckpt"
    axes[0].errorbar(
        sub["world_size"], sub["tps_mean"], yerr=sub["tps_std"],
        fmt="o-", capsize=5, capthick=1.5, label=label, linewidth=2, markersize=8,
    )

# Ideal linear scaling (anchored to 2-GPU no-ckpt baseline)
nockpt = agg[(agg["activation_ckpt"] == False) & (agg["world_size"] == 2)]
if len(nockpt) > 0:
    base = nockpt["tps_mean"].iloc[0]
    axes[0].plot([2, 4, 8], [base, base * 2, base * 4],
                 "k--", alpha=0.4, label="ideal linear (from 2-GPU)")

axes[0].set_xlabel("Number of GPUs")
axes[0].set_ylabel("Tokens / sec (global)")
axes[0].set_title("Strong scaling: throughput\n(mean ± std across repeated runs)")
axes[0].set_xticks([2, 4, 8])
axes[0].legend()
axes[0].grid(alpha=0.3)

# Panel 2: peak memory
for ckpt_val in [False, True]:
    sub = agg[agg["activation_ckpt"] == ckpt_val].sort_values("world_size")
    if len(sub) == 0:
        continue
    label = "with activation ckpt" if ckpt_val else "no activation ckpt"
    axes[1].plot(sub["world_size"], sub["mem_mean"],
                 "o-", label=label, linewidth=2, markersize=8)

axes[1].set_xlabel("Number of GPUs")
axes[1].set_ylabel("Peak GPU memory (GB)")
axes[1].set_title("Memory per GPU")
axes[1].set_xticks([2, 4, 8])
axes[1].axhline(16, color="red", linestyle=":", alpha=0.5, label="V100 16GB limit")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved plot to {PLOT_PATH}")

# ---------- save aggregated CSV ----------

agg.to_csv("/home/ubuntu/tinyvlm-implemention/fsdp_study/results_aggregated.csv", index=False)
print("Saved aggregated results to fsdp_study/results_aggregated.csv")