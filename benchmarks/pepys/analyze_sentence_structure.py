#!/usr/bin/env python3
"""
Sentence Structure Analysis for Pepys Diary Corpus

Analyzes sentence length patterns and memory chunk characteristics to inform
optimal chunking strategies for semantic memory systems.

Features:
- Sentence-level length distribution analysis
- Memory chunk size optimization recommendations
- Natural boundary detection
- Chunking strategy comparison
- Visualization of length patterns

Usage:
    python analyze_sentence_structure.py pepys_clean.txt [options]

Arguments:
    input_file              Path to parsed diary file (TIMESTAMP | TYPE | CATEGORY | CONTENT)

Options:
    --verbose, -v           Show detailed analysis
    --output PATH           Save markdown report (default: sentence_analysis_report.md)
    --sample N              Analyze only first N entries (default: all)
"""

import argparse
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def parse_entries(file_path: Path) -> List[Tuple[str, str]]:
    """Parse diary entries from file.

    :param file_path: Path to input file
    :return: List of (timestamp, content) tuples
    """
    entries = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse format: TIMESTAMP | TYPE | CATEGORY | CONTENT
            parts = line.split(' | ', maxsplit=3)
            if len(parts) == 4:
                timestamp, _, _, content = parts
                entries.append((timestamp, content))

    return entries


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex pattern.

    Handles multiple punctuation marks and preserves sentence boundaries.

    :param text: Input text
    :return: List of sentences
    """
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    sentence_pattern = r'[.!?]+(?=\s+[A-Z]|$)'
    sentences = re.split(sentence_pattern, text)

    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def analyze_entry_lengths(entries: List[Tuple[str, str]]) -> dict:
    """Analyze length distribution of diary entries.

    :param entries: List of (timestamp, content) tuples
    :return: Dict with statistics
    """
    lengths = [len(content) for _, content in entries]

    sorted_lengths = sorted(lengths)
    percentiles = {
        p: sorted_lengths[int(len(sorted_lengths) * p / 100)]
        for p in [10, 25, 50, 75, 90, 95, 99]
    }

    return {
        'count': len(lengths),
        'mean': statistics.mean(lengths),
        'median': statistics.median(lengths),
        'stdev': statistics.stdev(lengths) if len(lengths) > 1 else 0,
        'min': min(lengths),
        'max': max(lengths),
        'percentiles': percentiles,
    }


def analyze_sentences(entries: List[Tuple[str, str]]) -> dict:
    """Analyze sentence-level patterns.

    :param entries: List of (timestamp, content) tuples
    :return: Dict with sentence analysis
    """
    all_sentences = []
    sentences_per_entry = []

    for _, content in entries:
        sentences = split_into_sentences(content)
        all_sentences.extend(sentences)
        sentences_per_entry.append(len(sentences))

    sentence_lengths = [len(s) for s in all_sentences]
    sorted_lengths = sorted(sentence_lengths)

    percentiles = {
        p: sorted_lengths[int(len(sorted_lengths) * p / 100)]
        for p in [10, 25, 50, 75, 90, 95, 99]
    }

    # Distribution buckets
    buckets = [
        (0, 50, 'very_short'),
        (50, 100, 'short'),
        (100, 200, 'medium'),
        (200, 400, 'long'),
        (400, 800, 'very_long'),
        (800, float('inf'), 'extreme'),
    ]

    distribution = {}
    for low, high, label in buckets:
        count = sum(1 for l in sentence_lengths if low <= l < high)
        distribution[label] = {
            'range': f'{low}-{high if high != float("inf") else "∞"}',
            'count': count,
            'percentage': count / len(sentence_lengths) * 100 if sentence_lengths else 0,
        }

    return {
        'total_sentences': len(all_sentences),
        'mean_length': statistics.mean(sentence_lengths) if sentence_lengths else 0,
        'median_length': statistics.median(sentence_lengths) if sentence_lengths else 0,
        'stdev_length': statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0,
        'min_length': min(sentence_lengths) if sentence_lengths else 0,
        'max_length': max(sentence_lengths) if sentence_lengths else 0,
        'percentiles': percentiles,
        'distribution': distribution,
        'avg_sentences_per_entry': statistics.mean(sentences_per_entry) if sentences_per_entry else 0,
        'median_sentences_per_entry': statistics.median(sentences_per_entry) if sentences_per_entry else 0,
    }


def recommend_chunk_sizes(entry_stats: dict, sentence_stats: dict) -> dict:
    """Generate chunking recommendations based on analysis.

    :param entry_stats: Entry-level statistics
    :param sentence_stats: Sentence-level statistics
    :return: Dict with recommendations
    """
    median_entry = entry_stats['median']
    p75_entry = entry_stats['percentiles'][75]
    median_sentence = sentence_stats['median_length']
    avg_sentences_per_entry = sentence_stats['avg_sentences_per_entry']

    recommendations = {
        'current_state': {
            'median_entry_size': median_entry,
            'assessment': 'Very coarse' if median_entry > 1500 else 'Moderate' if median_entry > 800 else 'Fine-grained',
        },
        'strategies': []
    }

    # Strategy 1: Keep as-is
    recommendations['strategies'].append({
        'name': 'No Chunking (Current)',
        'target_size': median_entry,
        'pros': ['Preserves full context', 'No splitting overhead'],
        'cons': ['May overwhelm semantic search', 'Loses fine-grained retrieval'],
        'use_case': 'Full-text search, reading entire entries',
    })

    # Strategy 2: Split large entries
    if p75_entry > 2000:
        recommendations['strategies'].append({
            'name': 'Split Large Entries',
            'target_size': 1500,
            'method': 'Split entries >2000 chars at sentence boundaries',
            'pros': ['Handles outliers', 'Maintains reasonable chunk sizes'],
            'cons': ['Inconsistent chunk sizes', 'Complex logic'],
            'use_case': 'Hybrid approach for mixed content',
        })

    # Strategy 3: Semantic chunks (3-5 sentences)
    target_sentences = 4
    target_size = int(median_sentence * target_sentences)
    recommendations['strategies'].append({
        'name': 'Semantic Chunks (Recommended)',
        'target_size': target_size,
        'method': f'Group {target_sentences} sentences (~{target_size} chars)',
        'pros': ['Natural semantic boundaries', 'Consistent granularity', 'Better retrieval'],
        'cons': ['More chunks to process', 'Requires sentence detection'],
        'use_case': 'Semantic search, fine-grained retrieval',
    })

    # Strategy 4: Fixed size with sentence boundaries
    recommendations['strategies'].append({
        'name': 'Fixed Size (512 chars)',
        'target_size': 512,
        'method': 'Fill to 512 chars, break at sentence boundaries',
        'pros': ['Consistent size', 'Transformer-friendly'],
        'cons': [f'Breaks {sentence_stats["distribution"]["very_long"]["percentage"]:.1f}% of sentences', 'Arbitrary boundary'],
        'use_case': 'Fixed-size embeddings, standardized processing',
    })

    return recommendations


def display_entry_analysis(stats: dict):
    """Display entry-level analysis."""
    table = Table(title="📄 Entry-Level Analysis", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    table.add_row("Total Entries", f"{stats['count']:,}")
    table.add_row("Mean Length", f"{stats['mean']:.0f} chars")
    table.add_row("Median Length", f"{stats['median']:.0f} chars")
    table.add_row("Std Deviation", f"{stats['stdev']:.0f} chars")
    table.add_row("", "")  # Spacer
    table.add_row("Minimum", f"{stats['min']:,} chars")
    table.add_row("Maximum", f"{stats['max']:,} chars")
    table.add_row("", "")  # Spacer

    for p, val in stats['percentiles'].items():
        table.add_row(f"{p}th Percentile", f"{val:,} chars")

    console.print(table)


def display_sentence_analysis(stats: dict):
    """Display sentence-level analysis."""
    table = Table(title="📝 Sentence-Level Analysis", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    table.add_row("Total Sentences", f"{stats['total_sentences']:,}")
    table.add_row("Mean Length", f"{stats['mean_length']:.0f} chars")
    table.add_row("Median Length", f"{stats['median_length']:.0f} chars")
    table.add_row("Std Deviation", f"{stats['stdev_length']:.0f} chars")
    table.add_row("", "")  # Spacer
    table.add_row("Min / Max", f"{stats['min_length']} / {stats['max_length']:,} chars")
    table.add_row("", "")  # Spacer
    table.add_row("Avg per Entry", f"{stats['avg_sentences_per_entry']:.1f}")
    table.add_row("Median per Entry", f"{stats['median_sentences_per_entry']:.0f}")

    console.print(table)


def display_distribution(stats: dict):
    """Display sentence length distribution."""
    table = Table(title="📊 Sentence Length Distribution", box=box.ROUNDED)
    table.add_column("Range", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Percentage", style="green", justify="right")
    table.add_column("Visualization", style="blue")

    for label, data in stats['distribution'].items():
        count = data['count']
        pct = data['percentage']
        bar_width = int(pct / 2)  # Scale to 50 chars max
        visual_bar = "█" * bar_width

        label_text = label.replace('_', ' ').title()
        table.add_row(
            f"{label_text} ({data['range']})",
            f"{count:,}",
            f"{pct:.1f}%",
            visual_bar
        )

    console.print(table)


def display_recommendations(recommendations: dict):
    """Display chunking strategy recommendations."""
    # Current state panel
    current = recommendations['current_state']
    panel_content = Text()
    panel_content.append("Median Entry Size: ", style="bold cyan")
    panel_content.append(f"{current['median_entry_size']:.0f} chars\n", style="white")
    panel_content.append("Assessment: ", style="bold cyan")
    panel_content.append(f"{current['assessment']}", style="yellow")

    console.print(Panel(panel_content, title="🎯 Current State", border_style="cyan"))
    console.print()

    # Strategies
    table = Table(title="💡 Chunking Strategies", box=box.ROUNDED, show_lines=True)
    table.add_column("Strategy", style="cyan", width=25)
    table.add_column("Target Size", style="magenta", justify="right")
    table.add_column("Pros & Cons", style="white")
    table.add_column("Use Case", style="green")

    for strategy in recommendations['strategies']:
        pros = "\n".join([f"✓ {p}" for p in strategy['pros']])
        cons = "\n".join([f"✗ {c}" for c in strategy['cons']])
        pros_cons = f"{pros}\n{cons}"

        name = strategy['name']
        if 'Recommended' in name:
            name = f"[bold green]{name}[/bold green]"

        table.add_row(
            name,
            f"{strategy['target_size']:,}",
            pros_cons,
            strategy['use_case']
        )

    console.print(table)


def generate_markdown_report(
    entry_stats: dict,
    sentence_stats: dict,
    recommendations: dict,
    output_path: Path
):
    """Generate markdown analysis report.

    :param entry_stats: Entry-level statistics
    :param sentence_stats: Sentence-level statistics
    :param recommendations: Chunking recommendations
    :param output_path: Output file path
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Sentence Structure Analysis Report\n\n")
        f.write(f"**Generated**: {timestamp}\n\n")

        # Entry-level analysis
        f.write("## 📄 Entry-Level Analysis\n\n")
        f.write(f"- **Total Entries**: {entry_stats['count']:,}\n")
        f.write(f"- **Mean Length**: {entry_stats['mean']:.0f} chars\n")
        f.write(f"- **Median Length**: {entry_stats['median']:.0f} chars\n")
        f.write(f"- **Std Deviation**: {entry_stats['stdev']:.0f} chars\n")
        f.write(f"- **Range**: {entry_stats['min']:,} - {entry_stats['max']:,} chars\n\n")

        f.write("### Percentiles\n\n")
        f.write("| Percentile | Length (chars) |\n")
        f.write("|------------|---------------|\n")
        for p, val in entry_stats['percentiles'].items():
            f.write(f"| {p}th | {val:,} |\n")
        f.write("\n")

        # Sentence-level analysis
        f.write("## 📝 Sentence-Level Analysis\n\n")
        f.write(f"- **Total Sentences**: {sentence_stats['total_sentences']:,}\n")
        f.write(f"- **Mean Length**: {sentence_stats['mean_length']:.0f} chars\n")
        f.write(f"- **Median Length**: {sentence_stats['median_length']:.0f} chars\n")
        f.write(f"- **Std Deviation**: {sentence_stats['stdev_length']:.0f} chars\n")
        f.write(f"- **Range**: {sentence_stats['min_length']:,} - {sentence_stats['max_length']:,} chars\n")
        f.write(f"- **Avg Sentences per Entry**: {sentence_stats['avg_sentences_per_entry']:.1f}\n\n")

        f.write("### Distribution\n\n")
        f.write("| Range | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for label, data in sentence_stats['distribution'].items():
            label_text = label.replace('_', ' ').title()
            f.write(f"| {label_text} ({data['range']}) | {data['count']:,} | {data['percentage']:.1f}% |\n")
        f.write("\n")

        # Recommendations
        f.write("## 💡 Chunking Recommendations\n\n")
        f.write(f"**Current State**: {recommendations['current_state']['median_entry_size']:.0f} chars per entry ")
        f.write(f"({recommendations['current_state']['assessment']})\n\n")

        for strategy in recommendations['strategies']:
            name = strategy['name']
            if 'Recommended' in name:
                f.write(f"### {name} ⭐\n\n")
            else:
                f.write(f"### {name}\n\n")

            f.write(f"**Target Size**: {strategy['target_size']:,} chars\n\n")

            if 'method' in strategy:
                f.write(f"**Method**: {strategy['method']}\n\n")

            f.write("**Pros**:\n")
            for pro in strategy['pros']:
                f.write(f"- ✓ {pro}\n")
            f.write("\n**Cons**:\n")
            for con in strategy['cons']:
                f.write(f"- ✗ {con}\n")
            f.write(f"\n**Use Case**: {strategy['use_case']}\n\n")

        # Key insights
        f.write("## 🎯 Key Insights\n\n")

        # Insight 1: Sentence consistency
        cv = sentence_stats['stdev_length'] / sentence_stats['mean_length']
        f.write(f"1. **Sentence Length Consistency**: CV = {cv:.2f}\n")
        if cv < 0.5:
            f.write("   - Low variation - sentences are relatively uniform in length\n")
        elif cv < 1.0:
            f.write("   - Moderate variation - some diversity in sentence structure\n")
        else:
            f.write("   - High variation - wide range of sentence lengths\n")
        f.write("\n")

        # Insight 2: Chunking impact
        very_long_pct = sentence_stats['distribution']['very_long']['percentage']
        extreme_pct = sentence_stats['distribution']['extreme']['percentage']
        problematic_pct = very_long_pct + extreme_pct
        f.write(f"2. **Problematic Sentences**: {problematic_pct:.1f}% exceed 400 chars\n")
        if problematic_pct < 2:
            f.write("   - Most sentences fit well within typical chunk sizes\n")
        elif problematic_pct < 5:
            f.write("   - Small minority of very long sentences may need special handling\n")
        else:
            f.write("   - Significant number of very long sentences - consider pre-splitting\n")
        f.write("\n")

        # Insight 3: Optimal grouping
        target_size = 500
        sentences_per_chunk = target_size / sentence_stats['median_length']
        f.write(f"3. **Optimal Grouping**: For {target_size}-char chunks\n")
        f.write(f"   - Aim for ~{sentences_per_chunk:.1f} sentences per chunk\n")
        f.write(f"   - This preserves semantic coherence while maintaining consistent size\n")
        f.write("\n")

        f.write("---\n")
        f.write("\n*Report generated by Sentence Structure Analysis Pipeline*\n")

    console.print(f"[green]✓[/green] Markdown report saved: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze sentence structure for optimal chunking strategies"
    )
    parser.add_argument("input_file", type=str, help="Input diary file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output")
    parser.add_argument(
        "--output",
        type=str,
        default="sentence_analysis_report.md",
        help="Output markdown report path"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Analyze only first N entries"
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        console.print(f"[red]✗[/red] File not found: {input_path}")
        sys.exit(1)

    # Parse entries
    console.print(f"[cyan]📖 Loading diary entries from {input_path}...[/cyan]")
    entries = parse_entries(input_path)

    if args.sample:
        entries = entries[:args.sample]
        console.print(f"[dim]Using first {args.sample} entries for analysis[/dim]")

    console.print(f"[green]✓[/green] Loaded {len(entries):,} entries\n")

    # Analyze entry lengths
    console.print("[cyan]📊 Analyzing entry lengths...[/cyan]")
    entry_stats = analyze_entry_lengths(entries)

    # Analyze sentence patterns
    console.print("[cyan]📝 Analyzing sentence structure...[/cyan]")
    sentence_stats = analyze_sentences(entries)

    # Generate recommendations
    console.print("[cyan]💡 Generating chunking recommendations...[/cyan]\n")
    recommendations = recommend_chunk_sizes(entry_stats, sentence_stats)

    # Display results
    display_entry_analysis(entry_stats)
    console.print()
    display_sentence_analysis(sentence_stats)
    console.print()
    display_distribution(sentence_stats)
    console.print()
    display_recommendations(recommendations)
    console.print()

    # Generate markdown report
    output_path = Path(args.output)
    generate_markdown_report(entry_stats, sentence_stats, recommendations, output_path)

    console.print("\n[bold green]✓ Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
