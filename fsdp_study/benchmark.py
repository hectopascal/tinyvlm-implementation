import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fsdp_study/results.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ckpt, group in df.groupby("activation_ckpt"):
    label = "with ckpt" if ckpt else "no ckpt"
    axes[0].plot(group["world_size"], group["tokens_per_sec"], "o-", label=label)
    axes[1].plot(group["world_size"], group["peak_mem_gb"], "o-", label=label)

# Ideal scaling line
base = df[df["world_size"] == 2]["tokens_per_sec"].iloc[0]
ideal_x = [2, 4, 8]
ideal_y = [base, base*2, base*4]
axes[0].plot(ideal_x, ideal_y, "k--", alpha=0.4, label="ideal (linear)")

axes[0].set_xlabel("Number of GPUs")
axes[0].set_ylabel("Tokens / sec (global)")
axes[0].set_title("Strong scaling: throughput")
axes[0].legend()
axes[0].set_xticks([2, 4, 8])

axes[1].set_xlabel("Number of GPUs")
axes[1].set_ylabel("Peak GPU memory (GB)")
axes[1].set_title("Memory per GPU")
axes[1].legend()
axes[1].set_xticks([2, 4, 8])

plt.tight_layout()
plt.savefig("plots/scaling.png", dpi=150)
print("Saved plots/scaling.png")