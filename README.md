# Coarse Grain Pointer Chase
## Avg Load Latency vs Working Set Size

The plot below shows both mean and median load latency (in nanoseconds) across working set sizes, on a log₂ scale:

![Latency_Plot](./latency_plot.png)

---

## Average Latency by Memory Region

| Region | Range (KB) | Avg of Mean Latency (ns) | Avg of Median Latency (ns) |
|---------|-------------|---------------------------|-----------------------------|
| Vector Cache | 0 – 32 | 26.00 | 26.25 |
| L1 Cache | 32 – 256 | 39.75 | 40.00 |
| L2 Cache | 256 – 6,144 | 70.08 | 70.00 |
| Infinity Cache | 6,144 – 81,920 | 160.91 | 161.25 |
| VRAM | 1,048,576 – 25,165,824 | 825.65 | 828.12 |

> Source: [`latency_summary.csv`](./latency_summary.csv)

# Fine Grained Pointer Chase

![Fine_Latency_Plot](./fineGrained_latency_plot.png)

| Region | Range (KB) | Median Latency (clock cycles) | 
|---------|-------------|---------------------------|
| Vector Cache | 0 – 32 | 81 |
| L1 Cache | 32 – 256 | 136 | 
| L2 Cache | 512 – 6,144 | 226.6 |
| Infinity Cache | 16,384 – 81,920 | 575.5 |
| VRAM | 262144 – 25,165,824 | 2431.2 |

# Reverse Engineering Cache Block Size

![Line_Size_Plot](./latency_vs_access_id.png)

Each point represents the measured load latency for a single byte access. Low-latency points correspond to vector cache hits, while high-latency spikes indicate compulsory misses served by MALL. Because the array is not cache-line aligned, the first miss occurs at Access ID 0 and the next at Access ID 121; subsequent misses occur at a constant stride of 128 accesses. The fixed spacing between miss events reveals a cache block size of 128 bytes.
