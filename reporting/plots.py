"""
reporting/plots.py
------------------
Generates publication-quality visualizations for research reports.
Creates comparative bar charts and trade-off scatter plots.
"""
import os
from typing import Dict, List, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_benchmark_plots(results: List[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    """
    Generates a suite of publication-ready vector/PNG plots from benchmark results.
    Returns dictionary mapping plot_key -> absolute file path.
    """
    if not HAS_MATPLOTLIB or not results:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    generated_plots = {}

    principles = [r["routing_principle"].upper() for r in results]
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#64748b']
    bar_colors = colors[:len(principles)]

    plt.style.use('ggplot')

    # 1. Precision, Recall, F1, Accuracy Bar Chart
    plt.figure(figsize=(10, 5))
    x = range(len(principles))
    width = 0.2
    
    precisions = [r.get("precision", 0) * 100 for r in results]
    recalls = [r.get("recall", 0) * 100 for r in results]
    f1s = [r.get("f1", 0) * 100 for r in results]
    accuracies = [r.get("downstream_accuracy", 0) * 100 for r in results]

    plt.bar([i - 1.5*width for i in x], precisions, width=width, label='Precision', color='#3b82f6')
    plt.bar([i - 0.5*width for i in x], recalls, width=width, label='Recall', color='#10b981')
    plt.bar([i + 0.5*width for i in x], f1s, width=width, label='F1 Score', color='#f59e0b')
    plt.bar([i + 1.5*width for i in x], accuracies, width=width, label='Accuracy', color='#8b5cf6')

    plt.xticks(x, principles, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.title('Quality Metrics by Context Routing Principle', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    path_quality = os.path.join(output_dir, "fig_quality_metrics.png")
    plt.savefig(path_quality, dpi=300)
    plt.close()
    generated_plots["quality_metrics"] = path_quality

    # 2. Compression % vs Token Savings %
    plt.figure(figsize=(8, 4.5))
    compressions = [r.get("compression_pct", 0) for r in results]
    savings = [r.get("token_savings_pct", 0) for r in results]

    plt.bar([i - width/2 for i in x], compressions, width=width, label='Compression Ratio %', color='#06b6d4')
    plt.bar([i + width/2 for i in x], savings, width=width, label='Token Savings %', color='#10b981')
    plt.xticks(x, principles, fontweight='bold')
    plt.ylabel('Percentage (%)')
    plt.title('Context Reduction Efficiency', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    path_eff = os.path.join(output_dir, "fig_context_efficiency.png")
    plt.savefig(path_eff, dpi=300)
    plt.close()
    generated_plots["context_efficiency"] = path_eff

    # 3. Compression vs Downstream Accuracy (Scatter Trade-off)
    plt.figure(figsize=(7, 5))
    for i, r in enumerate(results):
        plt.scatter(
            r.get("compression_pct", 0),
            r.get("downstream_accuracy", 0) * 100,
            s=180,
            color=bar_colors[i],
            label=r["routing_principle"].upper(),
            zorder=5
        )
        plt.annotate(
            r["routing_principle"].upper(),
            (r.get("compression_pct", 0) + 1, r.get("downstream_accuracy", 0) * 100 + 0.5),
            fontsize=10,
            fontweight='bold'
        )

    plt.xlabel('Compression Ratio (%)', fontweight='bold')
    plt.ylabel('Downstream Gemini Accuracy (%)', fontweight='bold')
    plt.title('Trade-off: Context Compression vs. Model Accuracy', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    path_tradeoff = os.path.join(output_dir, "fig_compression_vs_accuracy.png")
    plt.savefig(path_tradeoff, dpi=300)
    plt.close()
    generated_plots["compression_vs_accuracy"] = path_tradeoff

    # 4. Latency vs Token Cost Bar Chart
    plt.figure(figsize=(8, 4.5))
    costs = [r.get("total_cost_usd", 0) * 1000 for r in results]  # Cost in mUSD
    latencies = [r.get("avg_latency_sec", 0) for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    ax1.bar([i - width/2 for i in x], latencies, width=width, color='#ef4444', label='Avg Latency (s)')
    ax2.bar([i + width/2 for i in x], costs, width=width, color='#8b5cf6', label='Cost (mUSD)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(principles, fontweight='bold')
    ax1.set_ylabel('Latency (seconds)', color='#ef4444', fontweight='bold')
    ax2.set_ylabel('Total Cost (mUSD)', color='#8b5cf6', fontweight='bold')
    plt.title('Latency & Token Cost Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path_cost_lat = os.path.join(output_dir, "fig_cost_latency.png")
    plt.savefig(path_cost_lat, dpi=300)
    plt.close()
    generated_plots["cost_latency"] = path_cost_lat

    # 5. Compression vs. Latency (Scatter Trade-off)
    plt.figure(figsize=(7, 5))
    for i, r in enumerate(results):
        plt.scatter(
            r.get("compression_pct", 0),
            r.get("avg_latency_sec", 0),
            s=180,
            color=bar_colors[i],
            label=r["routing_principle"].upper(),
            zorder=5
        )
        plt.annotate(
            r["routing_principle"].upper(),
            (r.get("compression_pct", 0) + 1, r.get("avg_latency_sec", 0) + 0.05),
            fontsize=10,
            fontweight='bold'
        )

    plt.xlabel('Compression Ratio (%)', fontweight='bold')
    plt.ylabel('Average Latency (seconds)', fontweight='bold')
    plt.title('Trade-off: Context Compression vs. Average Latency', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    path_comp_lat = os.path.join(output_dir, "fig_compression_vs_latency.png")
    plt.savefig(path_comp_lat, dpi=300)
    plt.close()
    generated_plots["compression_vs_latency"] = path_comp_lat

    # 6. Token Savings vs. Downstream Accuracy (Scatter Trade-off)
    plt.figure(figsize=(7, 5))
    for i, r in enumerate(results):
        plt.scatter(
            r.get("token_savings_pct", 0),
            r.get("downstream_accuracy", 0) * 100,
            s=180,
            color=bar_colors[i],
            label=r["routing_principle"].upper(),
            zorder=5
        )
        plt.annotate(
            r["routing_principle"].upper(),
            (r.get("token_savings_pct", 0) + 1, r.get("downstream_accuracy", 0) * 100 + 0.5),
            fontsize=10,
            fontweight='bold'
        )

    plt.xlabel('Token Savings (%)', fontweight='bold')
    plt.ylabel('Downstream Gemini Accuracy (%)', fontweight='bold')
    plt.title('Trade-off: Token Savings vs. Model Accuracy', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    path_sav_acc = os.path.join(output_dir, "fig_savings_vs_accuracy.png")
    plt.savefig(path_sav_acc, dpi=300)
    plt.close()
    generated_plots["savings_vs_accuracy"] = path_sav_acc

    return generated_plots
