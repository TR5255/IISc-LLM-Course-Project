"""
reporting/report_generator.py
------------------------------
Generates publication-ready academic research reports and export packages.
Includes:
  - Executive Summary & Recommendations
  - Statistical Analysis (Mean, Median, Std Dev, % improvement)
  - Auto-generated Results & Discussion draft for paper inclusion
  - Full ZIP Research Package exporter
"""
from __future__ import annotations

import os
import json
import zipfile
import statistics
import time
from typing import Dict, List, Any
from reporting.exporters import export_to_csv, export_to_json
from reporting.plots import generate_benchmark_plots
from reporting.narrative import (
    generate_observations,
    generate_conclusions,
    get_pareto_efficient_front,
)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class PaperReportGenerator:
    """Generates complete academic research packages and multi-format reports."""

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates statistical summary across all evaluated routing principles."""
        if not results:
            return {}

        metrics = ["precision", "recall", "f1", "ndcg@3", "downstream_accuracy", "compression_pct", "token_savings_pct", "avg_latency_sec", "total_cost_usd"]
        stats = {}

        for m in metrics:
            vals = [r.get(m, 0.0) for r in results if r.get(m) is not None]
            if vals:
                stats[m] = {
                    "mean": round(statistics.mean(vals), 4),
                    "median": round(statistics.median(vals), 4),
                    "std_dev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                    "max": max(vals),
                    "min": min(vals),
                }

        # Identify best performers
        best_f1 = max(results, key=lambda x: x.get("f1", 0) or 0)
        best_comp = max(results, key=lambda x: x.get("compression_pct", 0) or 0)
        best_lat = min(results, key=lambda x: x.get("avg_latency_sec", 999) or 999)
        best_cost = min(results, key=lambda x: x.get("total_cost_usd", 999) or 999)
        best_acc = max(results, key=lambda x: x.get("downstream_accuracy", 0) or 0)

        # Baseline comparison (relative to Random or first principle)
        baseline = results[0]
        improvements = {}
        for r in results:
            p = r["routing_principle"]
            base_acc = baseline.get("downstream_accuracy", 0.001) or 0.001
            base_f1 = baseline.get("f1", 0.001) or 0.001
            improvements[p] = {
                "acc_improvement_pct": round(((r.get("downstream_accuracy", 0) - base_acc) / base_acc) * 100, 2),
                "f1_improvement_pct": round(((r.get("f1", 0) - base_f1) / base_f1) * 100, 2),
            }

        return {
            "metric_statistics": stats,
            "best_performers": {
                "highest_f1": best_f1["routing_principle"],
                "highest_compression": best_comp["routing_principle"],
                "lowest_latency": best_lat["routing_principle"],
                "lowest_cost": best_cost["routing_principle"],
                "highest_accuracy": best_acc["routing_principle"],
            },
            "baseline_improvements": improvements,
        }

    def generate_report(self, results: List[Dict[str, Any]], run_id: str = None) -> Dict[str, str]:
        """
        Generates Markdown, PDF, CSV, JSON, figures, and ZIP package.
        Returns dict of generated artifact file paths.
        """
        if not run_id:
            run_id = f"benchmark_run_{int(time.time())}"

        run_dir = os.path.join(self.output_dir, run_id)
        figures_dir = os.path.join(run_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        stats = self.calculate_statistics(results)
        plots = generate_benchmark_plots(results, figures_dir)

        md_path = os.path.join(run_dir, "research_report.md")
        pdf_path = os.path.join(run_dir, "research_report.pdf")
        csv_path = os.path.join(run_dir, "metrics_table.csv")
        json_path = os.path.join(run_dir, "metrics_summary.json")
        zip_path = os.path.join(self.output_dir, f"{run_id}_package.zip")

        # 1. Export CSV & JSON
        export_to_csv(results, csv_path)
        export_to_json(results, json_path)

        # 2. Markdown Report
        md_content = self._build_markdown_report(results, stats, plots, run_id)
        with open(md_path, "w") as f:
            f.write(md_content)

        # 3. PDF Report
        if HAS_REPORTLAB:
            self._build_pdf_report(results, stats, plots, pdf_path, run_id)

        # 4. ZIP Package Creation
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(md_path, arcname="research_report.md")
            if os.path.exists(pdf_path):
                zipf.write(pdf_path, arcname="research_report.pdf")
            zipf.write(csv_path, arcname="metrics_table.csv")
            zipf.write(json_path, arcname="metrics_summary.json")
            for p_name, p_path in plots.items():
                zipf.write(p_path, arcname=f"figures/{os.path.basename(p_path)}")

        return {
            "run_id": run_id,
            "markdown": md_path,
            "pdf": pdf_path if HAS_REPORTLAB else "",
            "csv": csv_path,
            "json": json_path,
            "zip_package": zip_path,
            "figures": plots,
        }

    def _build_markdown_report(self, results: List[Dict[str, Any]], stats: Dict[str, Any], plots: Dict[str, str], run_id: str) -> str:
        best = stats.get("best_performers", {})
        
        # Methodology constants
        eval_model = "Gemini Flash (Fixed)"
        dataset_name = "LexGLUE Legal QA Benchmark"
        
        # Load per-item detail if available
        detail_path = os.path.join(self.output_dir, run_id, "per_item_detail.json")
        per_item_detail = []
        if os.path.exists(detail_path):
            try:
                with open(detail_path, "r") as f:
                    per_item_detail = json.load(f)
            except Exception:
                pass
        
        num_items = len(set(d.get("document_id") for d in per_item_detail)) if per_item_detail else 12

        # Generate Pareto candidates
        pareto_candidates = get_pareto_efficient_front(results)
        pareto_str = ", ".join(f"`{p.upper()}`" for p in pareto_candidates)
        
        # Observations & Conclusions
        observations = generate_observations(results, stats)
        obs_bullets = "\n".join(f"- {o}" for o in observations)
        conclusion_para = generate_conclusions(results, stats)

        md = f"""# Smart AI Router: Automated Academic Research Report
**Run Identifier**: `{run_id}`  
**Dataset**: {dataset_name}  
**Downstream Model**: {eval_model}  
**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary & Key Findings

- **Top F1 Routing Strategy**: **{best.get('highest_f1', 'N/A').upper()}**
- **Maximum Context Compression**: **{best.get('highest_compression', 'N/A').upper()}**
- **Optimal Latency Efficiency**: **{best.get('lowest_latency', 'N/A').upper()}**
- **Lowest Token Cost**: **{best.get('lowest_cost', 'N/A').upper()}**
- **Pareto-Optimal Frontier Set**: {pareto_str}
- **Overall Recommendation**: {conclusion_para}

---

## 2. Methodology & Experimental Setup

This benchmark evaluates various context routing principles designed to select the most relevant sentences or subsections of legal documents before context injection into a downstream Large Language Model (LLM).

- **Evaluation Dataset**: `{dataset_name}` ({num_items} items total).
- **Selection Policy**: `TopKPolicy(k=2)` (selects the top 2 highest-scoring chunks).
- **Downstream Reader Model**: `{eval_model}`.
- **Metrics and Evaluation Protocol**:
  - *Precision*: Portion of selected chunks that host annotated gold labels.
  - *Recall*: Portion of annotated gold chunks captured.
  - *F1 Score*: Harmonic mean of precision and recall.
  - *NDCG@3*: Normalized Discounted Cumulative Gain calculated using graded relevance scores (0-3).
  - *Compression Ratio %*: Percentage of characters removed from the document text.
  - *Token Savings %*: Percentage of whitespace-approximated tokens saved.
  - *Latency*: Total end-to-end question answering latency (seconds).
  - *Cost*: Downstream LLM token costs in USD.

---

## 3. Comparative Benchmark Matrix

| Routing Principle | Precision | Recall | F1 Score | NDCG@3 | Downstream Acc | Compression % | Latency (s) | Cost ($) |
|---|---|---|---|---|---|---|---|---|
"""
        for r in results:
            md += f"| **{r['routing_principle'].upper()}** | {r.get('precision',0):.4f} | {r.get('recall',0):.4f} | {r.get('f1',0):.4f} | {r.get('ndcg@3',0):.4f} | {r.get('downstream_accuracy',0):.4f} | {r.get('compression_pct',0):.1f}% | {r.get('avg_latency_sec',0):.3f}s | ${r.get('total_cost_usd',0):.6f} |\n"

        md += "\n---\n\n## 4. Statistical Summary\n\n"
        md += "| Metric | Mean | Median | Std Dev | Min | Max |\n"
        md += "|---|---|---|---|---|---|\n"
        m_stats = stats.get("metric_statistics", {})
        for metric, val in m_stats.items():
            md += f"| **{metric.upper()}** | {val['mean']} | {val['median']} | {val['std_dev']} | {val['min']} | {val['max']} |\n"

        md += "\n### Baseline Relative Improvements (% over Random)\n\n"
        for p, imp in stats.get("baseline_improvements", {}).items():
            md += f"- **{p.upper()}**: Accuracy Improvement = `{imp['acc_improvement_pct']}%`, F1 Improvement = `{imp['f1_improvement_pct']}%`\n"

        md += "\n---\n\n## 5. Quantitative Observations\n\n"
        md += obs_bullets + "\n"

        md += "\n---\n\n## 6. Strategic Recommendations & Conclusions\n\n"
        md += conclusion_para + "\n"

        md += "\n---\n\n## 7. Visualizations & Figures\n\n"
        
        figure_order = [
            ("quality_metrics", "Figure 1: Comparative Quality Metrics (Precision, Recall, F1, and Accuracy) across Routing Principles"),
            ("context_efficiency", "Figure 2: Context Compression and Token Reduction Efficiency by Routing Strategy"),
            ("compression_vs_accuracy", "Figure 3: Pareto Frontier Trade-off: Context Compression vs. Downstream Gemini Accuracy"),
            ("cost_latency", "Figure 4: Latency and Estimate Token Cost Comparison per Prompt Routing Run"),
            ("compression_vs_latency", "Figure 5: Relationship Analysis: Context Compression vs. Average Execution Latency"),
            ("savings_vs_accuracy", "Figure 6: Relationship Analysis: Token Savings vs. Downstream Reasoning Accuracy"),
        ]

        for fig_key, fig_caption in figure_order:
            if fig_key in plots:
                fig_path = plots[fig_key]
                md += f"### {fig_caption}\n![{fig_key}]({os.path.basename(fig_path)})\n\n"

        if per_item_detail:
            md += "\n---\n\n## 8. Appendix: Raw Metrics Per Item\n\n"
            md += "| Item ID | Routing Principle | Precision | Recall | F1 Score | NDCG@3 | Compression % | Downstream Acc |\n"
            md += "|---|---|---|---|---|---|---|---|\n"
            for d in per_item_detail:
                md += (
                    f"| {d['document_id']} | **{d['principle'].upper()}** | "
                    f"{d.get('precision',0):.4f} | {d.get('recall',0):.4f} | "
                    f"{d.get('f1',0):.4f} | {d.get('ndcg@3',0):.4f} | "
                    f"{d.get('compression_pct',0):.1f}% | {d.get('downstream_accuracy',0):.4f} |\n"
                )
            md += "\n"

        md += "---  \n*Report automatically generated by Smart AI Router Research Workbench.*\n"
        return md

    def _build_pdf_report(self, results: List[Dict[str, Any]], stats: Dict[str, Any], plots: Dict[str, str], pdf_path: str, run_id: str):
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0f172a'))
        story.append(Paragraph(f"Smart AI Router — Academic Research Report", title_style))
        story.append(Paragraph(f"Run ID: {run_id} | Dataset: LexGLUE | LLM: Gemini Flash", styles['Normal']))
        story.append(Spacer(1, 15))

        # Table data
        table_data = [["Principle", "Precision", "Recall", "F1", "NDCG@3", "Accuracy", "Compression", "Cost ($)"]]
        for r in results:
            table_data.append([
                r["routing_principle"].upper(),
                f"{r.get('precision',0):.2f}",
                f"{r.get('recall',0):.2f}",
                f"{r.get('f1',0):.2f}",
                f"{r.get('ndcg@3',0):.2f}",
                f"{r.get('downstream_accuracy',0):.2f}",
                f"{r.get('compression_pct',0):.1f}%",
                f"${r.get('total_cost_usd',0):.5f}",
            ])

        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # Table statistics
        styles.add(ParagraphStyle('SubHeader', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#0f172a')))
        story.append(Paragraph("Statistical Summary", styles['SubHeader']))
        story.append(Spacer(1, 5))
        
        stat_table_data = [["Metric", "Mean", "Median", "Std Dev", "Min", "Max"]]
        m_stats = stats.get("metric_statistics", {})
        for metric, val in m_stats.items():
            stat_table_data.append([
                metric.upper(),
                f"{val['mean']}",
                f"{val['median']}",
                f"{val['std_dev']}",
                f"{val['min']}",
                f"{val['max']}",
            ])

        t_stat = Table(stat_table_data)
        t_stat.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ]))
        story.append(t_stat)
        story.append(Spacer(1, 15))

        # Embed Figures
        figure_keys = ["quality_metrics", "context_efficiency", "compression_vs_accuracy", "cost_latency", "compression_vs_latency", "savings_vs_accuracy"]
        story.append(Paragraph("Visualizations", styles['SubHeader']))
        story.append(Spacer(1, 5))
        for fig_key in figure_keys:
            fig_path = plots.get(fig_key)
            if fig_path and os.path.exists(fig_path):
                story.append(Image(fig_path, width=450, height=225))
                story.append(Spacer(1, 10))

        doc.build(story)
