import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === CONFIG ===
LOG_FILE = "record.log"
PLOT_FILE = "latency_plot.png"
TABLE_FILE = "latency_summary.csv"

# === REGEX PATTERNS ===
kb_pattern = re.compile(r"KB:\s*(\d+)")
avg_pattern = re.compile(r"Avg Load Latency \(clock cycles\)\s*([\d.]+)")
median_pattern = re.compile(r"Median ET\s*:\s*([\d.]+)")

# === PARSE LOG FILE ===
with open(LOG_FILE, "r") as f:
    text = f.read()

kb_matches = kb_pattern.findall(text)
avg_matches = avg_pattern.findall(text)
median_matches = median_pattern.findall(text)

if not (len(kb_matches) == len(avg_matches) == len(median_matches)):
    raise ValueError("Mismatch in parsed entries — check log format.")

kb_values = np.array([int(kb) for kb in kb_matches])
avg_latency = np.array([float(x) for x in avg_matches])
median_latency = np.array([float(x) for x in median_matches])

# === CACHE REGION DEFINITIONS (KB) ===
cache_boundaries = {
    "Vector Cache": 32,
    "L1 Cache": 256,
    "L2 Cache": 6 * 1024,               # 6 MB
    "Infinity Cache": 80 * 1024,        # 80 MB
    "VRAM": 24 * 1024 * 1024            # 24 GB
}

# === COMPUTE REGION-WISE AVERAGES ===
region_means = {}
sorted_bounds = list(cache_boundaries.items())
prev_limit = 0

for label, upper_limit in sorted_bounds:
    # Start VRAM at 1 GB to avoid MALL region effects
    if label == "VRAM":
        lower_limit = 1024 * 1024
    else:
        lower_limit = prev_limit

    mask = (kb_values > lower_limit) & (kb_values <= upper_limit)
    if np.any(mask):
        mean_val = np.mean(avg_latency[mask])
        median_val = np.mean(median_latency[mask])
    else:
        mean_val, median_val = np.nan, np.nan

    region_means[label] = (lower_limit, upper_limit, mean_val, median_val)
    prev_limit = upper_limit

# === PLOT ===
plt.figure(figsize=(10, 6))
# plt.plot(kb_values, avg_latency, marker='o', label='Mean Latency (ns)')
plt.plot(kb_values, median_latency, marker='s', label='Median Latency (clock cycles)')

for xi, yi in zip(kb_values, median_latency):
    plt.annotate(
        "{:.1f}".format(yi),
        (xi, yi),
        textcoords="offset points",
        xytext=(0, 6),
        ha="center",
        fontsize=8
    )
plt.xscale('log', base=2)
plt.xlabel('Working Set Size (KB)')
plt.ylabel('Latency (clock cycles)')
plt.title('Load Latency vs Working Set Size')
plt.grid(True, which="both", ls="--", alpha=0.6)

# Add cache boundary lines + labels
ymax = plt.ylim()[1]
for label, upper_limit in sorted_bounds:
    plt.axvline(x=upper_limit, color='red', linestyle='--', linewidth=1)
    plt.text(upper_limit, ymax * 0.95, f"{label}\n({upper_limit:,} KB)",
             rotation=90, va='top', ha='right', fontsize=12, color='red')

plt.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
plt.close()


# === CREATE & SAVE SUMMARY TABLE ===
data = []
for label, (low, high, mean_val, median_val) in region_means.items():
    data.append({
        "Region": label,
        "Range (KB)": f"{low:,} - {high:,}",
        "Avg of Mean Latency (ns)": f"{mean_val:.2f}" if not np.isnan(mean_val) else "-",
        "Avg of Median Latency (ns)": f"{median_val:.2f}" if not np.isnan(median_val) else "-"
    })

df = pd.DataFrame(data)
df.to_csv(TABLE_FILE, index=False)
print(f"✅ Table saved to: {TABLE_FILE}")

# === ALSO PRINT SUMMARY ===
print("\n=== Average Latency by Memory Region ===")
print(df.to_string(index=False))
