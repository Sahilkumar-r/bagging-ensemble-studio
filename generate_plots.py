"""
Generate Publication-Quality Plots for PDC Matrix Multiplication Project.

Reads CSV benchmark results and produces plots suitable for
inclusion in the IEEE-style research paper.

Author: Asjad Abdullah (22i-2059)
Course: Parallel & Distributed Computing (PDC)
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Color palette (professional, colorblind-friendly)
COLORS = {
    'V1_Sequential': '#2c3e50',  # Dark blue-gray
    'V2_Parallel':   '#e74c3c',  # Red
    'V3_CacheAware': '#27ae60',  # Green
    'V4_Combined':   '#2980b9',  # Blue
}

MARKERS = {
    'V1_Sequential': 'o',
    'V2_Parallel':   's',
    'V3_CacheAware': '^',
    'V4_Combined':   'D',
}

LABELS = {
    'V1_Sequential': 'V1: Sequential',
    'V2_Parallel':   'V2: Parallel (OpenMP)',
    'V3_CacheAware': 'V3: Cache-Aware (Tiling)',
    'V4_Combined':   'V4: Combined',
}

# Use serif font for IEEE paper style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
})


def read_csv(filename):
    """Read CSV file and return list of dicts."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# Plot 1: Execution Time vs Matrix Size (Log Scale)
# ============================================================

def plot_execution_time():
    """All 4 versions, execution time vs matrix size."""
    data = read_csv('main_benchmark.csv')

    fig, ax = plt.subplots(figsize=(7, 5))

    for version in ['V1_Sequential', 'V2_Parallel', 'V3_CacheAware', 'V4_Combined']:
        rows = [r for r in data if r['version'] == version]
        sizes = [int(r['matrix_size']) for r in rows]
        times = [float(r['time_seconds']) for r in rows]

        ax.plot(sizes, times,
                color=COLORS[version],
                marker=MARKERS[version],
                label=LABELS[version],
                linewidth=2)

    ax.set_xlabel('Matrix Size (n)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time vs Matrix Size')
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(PLOTS_DIR, 'execution_time.pdf')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {filepath}")


# ============================================================
# Plot 2: Speedup vs Matrix Size
# ============================================================

def plot_speedup():
    """Speedup relative to V1 for all versions."""
    data = read_csv('main_benchmark.csv')

    fig, ax = plt.subplots(figsize=(7, 5))

    for version in ['V2_Parallel', 'V3_CacheAware', 'V4_Combined']:
        rows = [r for r in data if r['version'] == version]
        sizes = [int(r['matrix_size']) for r in rows]
        speedups = [float(r['speedup']) for r in rows]

        ax.plot(sizes, speedups,
                color=COLORS[version],
                marker=MARKERS[version],
                label=LABELS[version],
                linewidth=2)

    # Reference line at speedup=1
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (V1)')

    ax.set_xlabel('Matrix Size (n)')
    ax.set_ylabel('Speedup (relative to V1)')
    ax.set_title('Speedup vs Matrix Size')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(PLOTS_DIR, 'speedup.pdf')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {filepath}")


# ============================================================
# Plot 3: Thread Scalability
# ============================================================

def plot_thread_scalability():
    """Speedup vs thread count for V2 and V4 at n=1024 and n=2048."""
    data = read_csv('thread_scalability.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, n in enumerate([1024, 2048]):
        ax = axes[idx]

        for version in ['V2_Parallel', 'V4_Combined']:
            rows = [r for r in data
                    if r['version'] == version and int(r['matrix_size']) == n]
            threads = [int(r['threads']) for r in rows]
            speedups = [float(r['speedup']) for r in rows]

            ax.plot(threads, speedups,
                    color=COLORS[version],
                    marker=MARKERS[version],
                    label=LABELS[version],
                    linewidth=2)

        # Ideal linear speedup line
        max_threads = max(int(r['threads']) for r in data)
        ideal = list(range(1, max_threads + 1))
        ax.plot(ideal, ideal, 'k--', alpha=0.3, label='Ideal Linear')

        ax.set_xlabel('Number of Threads')
        ax.set_ylabel('Speedup')
        ax.set_title(f'Thread Scalability (n={n})')
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'thread_scalability.pdf')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {filepath}")


# ============================================================
# Plot 4: Tile Size Comparison
# ============================================================

def plot_tile_comparison():
    """Speedup for different tile sizes for V3 and V4."""
    data = read_csv('tile_comparison.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, n in enumerate([1024, 2048]):
        ax = axes[idx]

        for version in ['V3_CacheAware', 'V4_Combined']:
            rows = [r for r in data
                    if r['version'] == version and int(r['matrix_size']) == n]
            tiles = [int(r['tile_size']) for r in rows]
            speedups = [float(r['speedup']) for r in rows]

            ax.bar([t + (4 if version == 'V4_Combined' else -4) for t in range(len(tiles))],
                   speedups,
                   width=7,
                   color=COLORS[version],
                   label=LABELS[version],
                   alpha=0.85)

        ax.set_xlabel('Tile Size (B)')
        ax.set_ylabel('Speedup (relative to V1)')
        ax.set_title(f'Effect of Tile Size (n={n})')
        ax.set_xticks(range(len(TILE_SIZES := [16, 32, 64, 128])))
        ax.set_xticklabels([str(t) for t in TILE_SIZES])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'tile_comparison.pdf')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {filepath}")


# ============================================================
# Plot 5: Cache Miss Rate Comparison
# ============================================================

def plot_cache_misses():
    """L1 and L2 cache miss rates across versions."""
    data = read_csv('cache_misses.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # L1 Cache Miss Rate
    ax1 = axes[0]
    versions = ['V1_Sequential', 'V2_Parallel', 'V3_CacheAware', 'V4_Combined']
    x = np.arange(len(MATRIX_SIZES := [256, 512, 1024, 2048, 4096]))
    width = 0.18

    for i, version in enumerate(versions):
        rows = [r for r in data if r['version'] == version]
        miss_rates = [float(r['l1_miss_rate_pct']) for r in rows]
        ax1.bar(x + i * width, miss_rates, width,
                color=COLORS[version], label=LABELS[version], alpha=0.85)

    ax1.set_xlabel('Matrix Size (n)')
    ax1.set_ylabel('L1 Cache Miss Rate (%)')
    ax1.set_title('L1 Data Cache Miss Rate')
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels([str(n) for n in MATRIX_SIZES])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # IPC Comparison
    ax2 = axes[1]
    for i, version in enumerate(versions):
        rows = [r for r in data if r['version'] == version]
        ipcs = [float(r['ipc']) for r in rows]
        ax2.bar(x + i * width, ipcs, width,
                color=COLORS[version], label=LABELS[version], alpha=0.85)

    ax2.set_xlabel('Matrix Size (n)')
    ax2.set_ylabel('Instructions Per Cycle (IPC)')
    ax2.set_title('IPC Comparison')
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels([str(n) for n in MATRIX_SIZES])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'cache_misses.pdf')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {filepath}")


# ============================================================
# Plot 6: Combined Summary Bar Chart
# ============================================================

def plot_summary_bars():
    """Summary bar chart showing speedup for largest matrix (4096)."""
    data = read_csv('main_benchmark.csv')

    # Get n=4096 data
    rows_4096 = [r for r in data if int(r['matrix_size']) == 4096]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    versions = ['V1_Sequential', 'V2_Parallel', 'V3_CacheAware', 'V4_Combined']
    speedups = []
    for v in versions:
        row = [r for r in rows_4096 if r['version'] == v][0]
        speedups.append(float(row['speedup']))

    bars = ax.bar(range(len(versions)), speedups,
                  color=[COLORS[v] for v in versions],
                  alpha=0.85, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, speed in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{speed:.1f}×', ha='center', va='bottom', fontweight='bold',
                fontsize=12)

    ax.set_xlabel('Implementation Version')
    ax.set_ylabel('Speedup (relative to V1)')
    ax.set_title('Performance Summary: Matrix Size 4096×4096')
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels([LABELS[v].replace('V1: ', '').replace('V2: ', '')
                        .replace('V3: ', '').replace('V4: ', '')
                        for v in versions], fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, 'summary_bars.pdf')
    plt.savefig(filepath)
    plt.savefig(filepath.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {filepath}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 50)
    print("  Generating Publication-Quality Plots")
    print("=" * 50)

    plot_execution_time()
    plot_speedup()
    plot_thread_scalability()
    plot_tile_comparison()
    plot_cache_misses()
    plot_summary_bars()

    print(f"\n✓ All plots saved to: {PLOTS_DIR}")
