import re
import matplotlib.pyplot as plt

# ================= CONFIG =================
LOG_FILE = "access_latencies.txt"
OUT_FILE = "latency_vs_access_id.png"
LATENCY_THRESHOLD = 100  # cycles

# ================= PARSE LOG =================
ids = []
lats = []

pattern = re.compile(
    r"Access ID:\s*(\d+)\s*\|\s*Latency \(clock cycles\):\s*([\d.]+)"
)

with open(LOG_FILE, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            ids.append(int(m.group(1)))
            lats.append(float(m.group(2)))

if not ids:
    raise RuntimeError("No data parsed — check log format")

# ================= IDENTIFY SPIKES =================
spike_ids = [x for x, y in zip(ids, lats) if y > LATENCY_THRESHOLD]
spike_lats = [y for y in lats if y > LATENCY_THRESHOLD]

# ================= PLOT =================
plt.figure(figsize=(14, 4))

plt.plot(ids, lats, linewidth=1, alpha=0.7)
plt.scatter(ids, lats, s=18)

# --- Label Y values for spikes ---
for x, y in zip(spike_ids, spike_lats):
    plt.annotate(
        f"{int(y)}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

# --- Label X-axis ticks at spike locations ---
plt.xticks(spike_ids, [str(x) for x in spike_ids])

# --- Label stride between consecutive spikes ---
ymin, ymax = plt.ylim()
y_stride = ymin + 0.08 * (ymax - ymin)  # place near bottom of plot

for i in range(1, len(spike_ids)):
    x0 = spike_ids[i - 1]
    x1 = spike_ids[i]
    stride = x1 - x0
    x_mid = (x0 + x1) / 2

    plt.text(
        x_mid,
        y_stride,
        f"Δ={stride}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black"
    )

plt.xlabel("Access ID")
plt.ylabel("Latency (clock cycles)")
plt.title("Latency vs Access ID")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300)
plt.close()

print(f"Saved plot to {OUT_FILE}")
