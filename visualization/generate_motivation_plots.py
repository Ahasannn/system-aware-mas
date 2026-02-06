#!/usr/bin/env python3
"""
Generate publication-ready motivation figure for Resource Underutilization.

Creates a single figure with two panels:
  (a) Queue Depth: Bar chart showing small vs large models across arrival rates
  (b) Latency: Line chart with shaded area highlighting the underutilization gap
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CSV_NAME = 'baseline_motivation_sweep_math_test_1000_poisson_2.csv'

MODEL_NAME_MAP = {
    'Qwen/Qwen2.5-Coder-14B-Instruct': 'Qwen-14B',
    'mistralai/Mistral-Small-24B-Instruct-2501': 'Mistral-24B',
    'meta-llama/Llama-3.2-3B-Instruct': 'Llama-3B',
    'meta-llama/Llama-3.1-8B-Instruct': 'Llama-8B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-32B',
}

# Group models by size
SIZE_GROUPS = {
    'Small Models (3-8B)': {'Llama-3B', 'Llama-8B'},
    'Large Models (14-32B)': {'Qwen-14B', 'Mistral-24B', 'DeepSeek-32B'},
}

# Colors - professional and colorblind-friendly
COLORS = {
    'Small Models (3-8B)': '#2166AC',      # Professional blue
    'Large Models (14-32B)': '#D6604D',    # Professional red/coral
    'gap_fill': '#FFCCCC',                  # Light red for underutilization gap
}

GROUP_ORDER = ['Small Models (3-8B)', 'Large Models (14-32B)']


def resolve_csv_path(csv_arg: str) -> Path:
    """Find the CSV file in various locations."""
    candidate = Path(csv_arg)
    if candidate.exists():
        return candidate

    candidates = [
        Path('logs/motivation_plot_generator_data') / csv_arg,
        Path('..') / 'logs/motivation_plot_generator_data' / csv_arg,
        Path(__file__).parent.parent / 'logs/motivation_plot_generator_data' / csv_arg,
    ]
    for path in candidates:
        if path.exists():
            return path

    tried = ', '.join(str(p) for p in candidates)
    raise FileNotFoundError(f'Could not find CSV: {csv_arg}. Tried: {tried}')


def assign_size_group(model_short: str) -> str | None:
    """Assign a model to its size group."""
    for group, models in SIZE_GROUPS.items():
        if model_short in models:
            return group
    return None


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        # Figure
        'figure.figsize': (10, 4),
        'figure.dpi': 150,
        'figure.facecolor': 'white',

        # Font - use serif for publications
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,

        # Axes
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,

        # Legend
        'legend.fontsize': 10,
        'legend.frameon': False,

        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.color': '#CCCCCC',

        # Save
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def load_and_process_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and compute aggregated metrics."""
    df = pd.read_csv(csv_path)
    step_df = df[df['record_type'] == 'step'].copy()

    # Convert columns to numeric
    for col in ['latency_seconds', 'arrival_rate', 'llm_running', 'llm_waiting']:
        if col in step_df.columns:
            step_df[col] = pd.to_numeric(step_df[col], errors='coerce')

    # Compute queue depth
    step_df['queue_depth'] = step_df['llm_running'].fillna(0) + step_df['llm_waiting'].fillna(0)

    # Map model names and assign groups
    step_df['model_short'] = step_df['llm_name'].map(MODEL_NAME_MAP)
    step_df = step_df[step_df['model_short'].notna()].copy()
    step_df['size_group'] = step_df['model_short'].apply(assign_size_group)
    step_df = step_df[step_df['size_group'].notna()].copy()

    return step_df


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std for each group at each arrival rate."""
    stats = df.groupby(['arrival_rate', 'size_group']).agg({
        'latency_seconds': ['mean', 'std', 'count'],
        'queue_depth': ['mean', 'std'],
    }).reset_index()

    # Flatten column names
    stats.columns = [
        'arrival_rate', 'size_group',
        'latency_mean', 'latency_std', 'count',
        'queue_mean', 'queue_std'
    ]

    # Compute standard error
    stats['latency_se'] = stats['latency_std'] / np.sqrt(stats['count'])
    stats['queue_se'] = stats['queue_std'] / np.sqrt(stats['count'])

    return stats


def generate_figure_caption(stats: pd.DataFrame, arrival_rates: list,
                            idle_stats: dict = None) -> str:
    """Generate a publication-ready figure caption."""
    max_rate = max(arrival_rates)
    min_rate = min(arrival_rates)

    # Get values at max arrival rate
    small_at_max = stats[(stats['arrival_rate'] == max_rate) &
                         (stats['size_group'] == 'Small Models (3-8B)')]
    large_at_max = stats[(stats['arrival_rate'] == max_rate) &
                         (stats['size_group'] == 'Large Models (14-32B)')]

    small_queue_max = small_at_max['queue_mean'].values[0] if len(small_at_max) > 0 else 0
    large_queue_max = large_at_max['queue_mean'].values[0] if len(large_at_max) > 0 else 0
    small_latency_max = small_at_max['latency_mean'].values[0] if len(small_at_max) > 0 else 0
    large_latency_max = large_at_max['latency_mean'].values[0] if len(large_at_max) > 0 else 0

    queue_ratio = small_queue_max / large_queue_max if large_queue_max > 0 else float('inf')
    latency_ratio = small_latency_max / large_latency_max if large_latency_max > 0 else float('inf')

    # Format idle stats
    idle_str = ""
    if idle_stats:
        large_idle = idle_stats.get('Large Models', 0)
        small_idle = idle_stats.get('Small Models', 0)
        idle_str = f"{large_idle:.0f}% of the time (vs {small_idle:.0f}% for small models)"

    caption = f"""Figure 1: Resource underutilization in system-load-unaware multi-agent LLM routing. (a) Queue depth shows severe load imbalance—requests accumulate in small models while large models remain underutilized, with {queue_ratio:.0f}x disparity at high load. (b) Response latency reveals that although small models appear efficient at low load, they become significantly slower than large models as load increases. The shaded region represents avoidable latency where large models could serve requests faster but remain idle. (c) At low system load, large models are idle {idle_str}, representing capacity that could be leveraged for extended reasoning to improve output quality. MATH dataset, 1000 samples, Poisson arrivals."""

    return caption


def create_motivation_figure(df: pd.DataFrame, output_path: Path):
    """Create the three-panel motivation figure."""
    setup_publication_style()

    stats = compute_group_stats(df)
    arrival_rates = sorted(stats['arrival_rate'].unique())

    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # ========================================================================
    # Panel (a): Queue Depth - Grouped Bar Chart
    # ========================================================================
    x = np.arange(len(arrival_rates))
    width = 0.35

    small_queue = []
    large_queue = []
    small_queue_err = []
    large_queue_err = []

    for rate in arrival_rates:
        for group in GROUP_ORDER:
            row = stats[(stats['arrival_rate'] == rate) & (stats['size_group'] == group)]
            if group == 'Small Models (3-8B)':
                small_queue.append(row['queue_mean'].values[0] if len(row) > 0 else 0)
                small_queue_err.append(row['queue_se'].values[0] if len(row) > 0 else 0)
            else:
                large_queue.append(row['queue_mean'].values[0] if len(row) > 0 else 0)
                large_queue_err.append(row['queue_se'].values[0] if len(row) > 0 else 0)

    bars1 = ax1.bar(x - width/2, small_queue, width,
                    label='Small Models (3-8B)',
                    color=COLORS['Small Models (3-8B)'],
                    edgecolor='#1a1a1a', linewidth=0.8,
                    yerr=small_queue_err, capsize=4, error_kw={'linewidth': 1.2})
    bars2 = ax1.bar(x + width/2, large_queue, width,
                    label='Large Models (14-32B)',
                    color=COLORS['Large Models (14-32B)'],
                    edgecolor='#1a1a1a', linewidth=0.8,
                    yerr=large_queue_err, capsize=4, error_kw={'linewidth': 1.2})

    ax1.set_xlabel('Request Arrival Rate (req/min)', fontweight='medium', fontsize=11)
    ax1.set_ylabel('Average Queue Depth', fontweight='medium', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(r)}' for r in arrival_rates])
    ax1.legend(loc='upper left', frameon=True, fancybox=False,
               edgecolor='#cccccc', facecolor='white')
    # Set y-axis limit to accommodate annotations
    max_queue = max(small_queue)
    ax1.set_ylim(bottom=0, top=max_queue * 1.25)

    # Add annotation showing overload vs underutilized
    max_idx = np.argmax(small_queue)
    if small_queue[max_idx] > 50:
        # Add "Overloaded" annotation for small models (above the bar)
        ax1.annotate('Congested',
                     xy=(x[max_idx] - width/2, small_queue[max_idx] + small_queue_err[max_idx]),
                     xytext=(x[max_idx] - width/2, small_queue[max_idx] + 18),
                     fontsize=9, ha='center', va='bottom',
                     color=COLORS['Small Models (3-8B)'],
                     fontweight='bold')
        # Add "Idle" annotation for large models
        ax1.annotate('Idle',
                     xy=(x[max_idx] + width/2, large_queue[max_idx] + large_queue_err[max_idx]),
                     xytext=(x[max_idx] + width/2, large_queue[max_idx] + 25),
                     fontsize=9, ha='center', va='bottom',
                     color=COLORS['Large Models (14-32B)'],
                     fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=COLORS['Large Models (14-32B)'],
                                     lw=1.2, shrinkB=2))

    # Add panel label
    ax1.text(-0.14, 1.02, '(a)', transform=ax1.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # ========================================================================
    # Panel (b): Latency - Line Chart with Shaded Gap
    # ========================================================================
    small_latency = []
    large_latency = []
    small_latency_err = []
    large_latency_err = []

    for rate in arrival_rates:
        for group in GROUP_ORDER:
            row = stats[(stats['arrival_rate'] == rate) & (stats['size_group'] == group)]
            if group == 'Small Models (3-8B)':
                small_latency.append(row['latency_mean'].values[0] if len(row) > 0 else 0)
                small_latency_err.append(row['latency_se'].values[0] if len(row) > 0 else 0)
            else:
                large_latency.append(row['latency_mean'].values[0] if len(row) > 0 else 0)
                large_latency_err.append(row['latency_se'].values[0] if len(row) > 0 else 0)

    small_latency = np.array(small_latency)
    large_latency = np.array(large_latency)
    small_latency_err = np.array(small_latency_err)
    large_latency_err = np.array(large_latency_err)

    # Create smooth interpolation for the shaded area
    from scipy.interpolate import make_interp_spline
    try:
        # Smooth curve for better visualization
        rates_smooth = np.linspace(min(arrival_rates), max(arrival_rates), 100)
        if len(arrival_rates) >= 4:
            spl_small = make_interp_spline(arrival_rates, small_latency, k=2)
            spl_large = make_interp_spline(arrival_rates, large_latency, k=2)
            small_smooth = spl_small(rates_smooth)
            large_smooth = spl_large(rates_smooth)
        else:
            small_smooth = np.interp(rates_smooth, arrival_rates, small_latency)
            large_smooth = np.interp(rates_smooth, arrival_rates, large_latency)

        # Fill the gap only where small > large (underutilization region)
        ax2.fill_between(rates_smooth, large_smooth, small_smooth,
                         where=(small_smooth > large_smooth),
                         alpha=0.25, color='#FF6B6B',
                         label='Avoidable Latency')
    except ImportError:
        # Fallback without scipy
        ax2.fill_between(arrival_rates, large_latency, small_latency,
                         where=(small_latency > large_latency),
                         alpha=0.25, color='#FF6B6B',
                         label='Avoidable Latency')

    # Plot lines with markers
    ax2.plot(arrival_rates, small_latency, 'o-',
             color=COLORS['Small Models (3-8B)'],
             linewidth=2.5, markersize=9, markeredgecolor='white',
             markeredgewidth=1.5, label='Small Models (3-8B)', zorder=5)

    ax2.plot(arrival_rates, large_latency, 's-',
             color=COLORS['Large Models (14-32B)'],
             linewidth=2.5, markersize=9, markeredgecolor='white',
             markeredgewidth=1.5, label='Large Models (14-32B)', zorder=5)

    ax2.set_xlabel('Request Arrival Rate (req/min)', fontweight='medium', fontsize=11)
    ax2.set_ylabel('Average Latency (seconds)', fontweight='medium', fontsize=11)
    ax2.set_xticks(arrival_rates)
    ax2.set_xticklabels([f'{int(r)}' for r in arrival_rates])
    ax2.legend(loc='upper left', frameon=True, fancybox=False,
               edgecolor='#cccccc', facecolor='white')
    ax2.set_ylim(bottom=0)

    # Add panel label
    ax2.text(-0.14, 1.02, '(b)', transform=ax2.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # Add annotation showing the underutilization gap at high load
    last_idx = len(arrival_rates) - 1
    gap_size = small_latency[last_idx] - large_latency[last_idx]

    if gap_size > 3:
        gap_center = (small_latency[last_idx] + large_latency[last_idx]) / 2
        # Draw bracket-style annotation
        ax2.annotate('',
                     xy=(arrival_rates[last_idx] + 8, large_latency[last_idx]),
                     xytext=(arrival_rates[last_idx] + 8, small_latency[last_idx]),
                     arrowprops=dict(arrowstyle='<->', color='#8B0000', lw=1.5,
                                     shrinkA=2, shrinkB=2))
        ax2.text(arrival_rates[last_idx] + 15, gap_center,
                 'Wasted\nCapacity',
                 fontsize=9, ha='left', va='center',
                 color='#8B0000', fontweight='bold')

    # Extend x-axis slightly for annotation
    ax2.set_xlim(left=min(arrival_rates) - 10, right=max(arrival_rates) + 35)

    # ========================================================================
    # Panel (c): Idle Capacity at Low Load - Small vs Large Model Groups
    # ========================================================================
    low_load_rate = min(arrival_rates)
    low_load_df = df[df['arrival_rate'] == low_load_rate]

    # Compute idle percentage for each group
    small_low = low_load_df[low_load_df['size_group'] == 'Small Models (3-8B)']
    large_low = low_load_df[low_load_df['size_group'] == 'Large Models (14-32B)']

    small_idle = ((small_low['llm_running'] + small_low['llm_waiting']) == 0).mean() * 100
    large_idle = ((large_low['llm_running'] + large_low['llm_waiting']) == 0).mean() * 100

    # Store for caption
    idle_stats = {'Small Models': small_idle, 'Large Models': large_idle}

    # Dedicated colors for panel (c) - consistent for both bars
    active_color = '#7B2D8E'  # Purple for processing
    idle_color = '#4CAF50'    # Green for idle (opportunity)

    # Create horizontal stacked bars for both groups
    bar_height = 0.35
    y_positions = [-0.1, -0.55]
    group_names = ['Small Models (3-8B)', 'Large Models (14-32B)']
    idle_pcts = [small_idle, large_idle]

    for i, (y_pos, group_name, idle_pct) in enumerate(
            zip(y_positions, group_names, idle_pcts)):
        active_pct = 100 - idle_pct

        # Active bar in purple
        ax3.barh(y_pos, active_pct, height=bar_height, color=active_color,
                 edgecolor='#1a1a1a', linewidth=1.2,
                 label='Processing' if i == 0 else '')
        # Idle bar in green
        ax3.barh(y_pos, idle_pct, height=bar_height, left=active_pct,
                 color=idle_color, edgecolor='#1a1a1a', linewidth=1.2, alpha=0.7,
                 label='Idle (0 requests)' if i == 0 else '')

        # Add percentage labels
        if active_pct > 15:
            ax3.text(active_pct / 2, y_pos, f'{active_pct:.0f}%',
                     ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax3.text(active_pct + idle_pct / 2, y_pos, f'{idle_pct:.0f}%',
                 ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Group label on the left
        short_label = 'Small (3-8B)' if '3-8B' in group_name else 'Large (14-32B)'
        ax3.text(-2, y_pos, short_label, ha='right', va='center', fontsize=9,
                 fontweight='medium', color='#333333')

    # Add annotation showing idle capacity can be used for more reasoning
    ax3.annotate('Available for extended\nreasoning to improve quality',
                 xy=(100 - large_idle/2, -0.95),
                 xytext=(100 - large_idle/2, -1.25),
                 ha='center', va='top', fontsize=8,
                 color='#1B5E20', fontweight='medium',
                 arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                           edgecolor='#4CAF50', alpha=0.9))

    # Styling
    ax3.set_xlim(0, 105)
    ax3.set_ylim(-1.6, 0.9)
    ax3.set_xlabel('Capacity Utilization (%)', fontweight='medium', fontsize=11)
    ax3.set_yticks([])
    ax3.spines['left'].set_visible(False)

    # Low load context (top right area)
    ax3.text(0.98, 0.92, f'Low Load ({int(low_load_rate)} req/min)',
             transform=ax3.transAxes, fontsize=9, ha='right', va='top',
             color='#444444', fontweight='medium')

    # Legend
    ax3.legend(loc='upper left', frameon=True, fancybox=False,
               edgecolor='#cccccc', facecolor='white', fontsize=9,
               bbox_to_anchor=(0.0, 0.95))

    # Add panel label
    ax3.text(-0.18, 1.02, '(c)', transform=ax3.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # Add overall title
    fig.suptitle('Resource Underutilization in Static Multi-Agent LLM Routing',
                 fontsize=13, fontweight='bold', y=1.02)

    # Add model legend below the figure - centered, left-aligned vertically
    small_color = COLORS['Small Models (3-8B)']
    large_color = COLORS['Large Models (14-32B)']

    # Both squares aligned at same x position
    box_x = 0.32
    text_x = 0.33

    # Small models line
    fig.text(box_x, -0.03, '\u25A0', fontsize=11, color=small_color,
             ha='left', va='center', fontweight='bold')
    fig.text(text_x, -0.03, 'Small (3-8B): Llama-3.2-3B, Llama-3.1-8B',
             fontsize=9, ha='left', va='center', color='#333333')

    # Large models line below (same x alignment)
    fig.text(box_x, -0.065, '\u25A0', fontsize=11, color=large_color,
             ha='left', va='center', fontweight='bold')
    fig.text(text_x, -0.065, 'Large (14-32B): Qwen2.5-14B, Mistral-24B, DeepSeek-32B',
             fontsize=9, ha='left', va='center', color='#333333')

    # Footnote - dataset info (smaller, below model legend)
    fig.text(0.5, -0.10,
             'MAS Router baseline on MATH dataset (1000 samples, Poisson arrivals)',
             ha='center', va='top', fontsize=8, fontstyle='italic', color='#666666')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.28, top=0.88, bottom=0.15)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {output_path}')

    # Generate and save caption
    caption = generate_figure_caption(stats, arrival_rates, idle_stats)
    caption_path = output_path.with_suffix('.txt')
    caption_path.write_text(caption)
    print(f'Caption saved: {caption_path}')
    print(f'\n{"="*70}')
    print('FIGURE CAPTION (for paper):')
    print('='*70)
    print(caption)
    print('='*70)


def compute_idle_stats(df: pd.DataFrame, model_filter: str = 'DeepSeek') -> pd.DataFrame:
    """Compute idle percentage for a specific model across arrival rates."""
    # Filter for the specific model
    model_df = df[df['llm_name'].str.contains(model_filter, case=False)]

    stats = []
    for rate in sorted(model_df['arrival_rate'].unique()):
        rate_data = model_df[model_df['arrival_rate'] == rate]
        # Idle = zero queue (no running + no waiting requests)
        idle_mask = (rate_data['llm_running'] + rate_data['llm_waiting']) == 0
        idle_pct = idle_mask.mean() * 100
        n_samples = len(rate_data)

        stats.append({
            'arrival_rate': rate,
            'idle_pct': idle_pct,
            'active_pct': 100 - idle_pct,
            'n_samples': n_samples,
            'avg_running': rate_data['llm_running'].mean(),
            'avg_latency': rate_data['latency_seconds'].mean(),
        })

    return pd.DataFrame(stats)


def generate_issue2_caption(idle_pct: float, load_rate: int = 10) -> str:
    """Generate caption for Issue-2 figure."""
    caption = f"""Figure 2: Idle inference capacity represents missed opportunities for quality improvement.
At low system load ({load_rate} req/min), DeepSeek-R1-Distill-Qwen-32B has zero active requests {idle_pct:.0f}% of the time (measured via vLLM running/waiting queue metrics). This substantial idle capacity could be leveraged for additional reasoning passes, verification, or self-consistency checks to improve output quality.
Static task-aware routing (MAS Router) fails to exploit this available capacity, routing requests based solely on task complexity rather than infrastructure availability. An infrastructure-aware system could utilize this idle time for quality-enhancing operations.
Results from MATH dataset with 1000 test samples under Poisson arrival patterns."""

    return caption


def create_idle_capacity_figure(df: pd.DataFrame, output_path: Path):
    """Create the Issue-2 figure: Idle capacity for quality improvement at low load."""
    setup_publication_style()

    # Focus on LOW LOAD only (10 req/min)
    low_load_rate = 10.0
    low_load_df = df[df['arrival_rate'] == low_load_rate]

    # Get DeepSeek-32B stats at low load
    deepseek_df = low_load_df[low_load_df['llm_name'].str.contains('DeepSeek', case=False)]
    idle_pct = ((deepseek_df['llm_running'] + deepseek_df['llm_waiting']) == 0).mean() * 100
    active_pct = 100 - idle_pct
    avg_running = deepseek_df['llm_running'].mean()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Colors
    active_color = '#7B2D8E'  # Purple for active
    idle_color = '#4CAF50'    # Green for idle (opportunity)

    # Create horizontal stacked bar showing capacity utilization
    bar_height = 0.5
    y_pos = 0

    # Draw the stacked bar
    ax.barh(y_pos, active_pct, height=bar_height, color=active_color,
            edgecolor='#1a1a1a', linewidth=1.2, label='Active (≥1 request running)')
    ax.barh(y_pos, idle_pct, height=bar_height, left=active_pct, color=idle_color,
            edgecolor='#1a1a1a', linewidth=1.2, alpha=0.7, label='Idle (0 requests running)')

    # Add percentage labels on the bars
    ax.text(active_pct / 2, y_pos, f'{active_pct:.0f}%\nActive',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(active_pct + idle_pct / 2, y_pos, f'{idle_pct:.0f}%\nIdle',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Add annotation showing what idle capacity could be used for
    ax.annotate('',
                xy=(active_pct + 5, y_pos + bar_height/2 + 0.05),
                xytext=(95, y_pos + bar_height/2 + 0.05),
                arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=2))

    # Add text box explaining the opportunity
    opportunity_text = 'Opportunity for Quality Improvement:\n• Verification passes\n• Self-consistency checks\n• Extended reasoning'
    ax.text(active_pct + idle_pct/2, y_pos - 0.45, opportunity_text,
            ha='center', va='top', fontsize=10,
            color='#1B5E20', fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                      edgecolor='#4CAF50', alpha=0.9))

    # Styling
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.2, 0.8)
    ax.set_xlabel('Capacity Utilization (%)', fontweight='medium', fontsize=12)
    ax.set_yticks([])

    # Remove y-axis spine
    ax.spines['left'].set_visible(False)

    # Title with model name and context
    ax.set_title('DeepSeek-R1-Distill-Qwen-32B Capacity at Low Load',
                 fontsize=13, fontweight='bold', pad=15)

    # Subtitle showing the load condition
    ax.text(50, 0.65, f'System Load: {int(low_load_rate)} req/min',
            ha='center', va='bottom', fontsize=11, fontstyle='italic', color='#555555')

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='#cccccc', facecolor='white')

    # Footnote
    fig.text(0.5, -0.05,
             'MAS Router baseline on MATH dataset (1000 test samples, Poisson arrivals)',
             ha='center', va='top', fontsize=9, fontstyle='italic', color='#555555')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {output_path}')

    # Generate and save caption
    caption = generate_issue2_caption(idle_pct, int(low_load_rate))
    caption_path = output_path.with_suffix('.txt')
    caption_path.write_text(caption)
    print(f'Caption saved: {caption_path}')
    print(f'\n{"="*70}')
    print('FIGURE CAPTION (for paper):')
    print('='*70)
    print(caption)
    print('='*70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready motivation figure for Resource Underutilization.'
    )
    parser.add_argument(
        '--csv', default=DEFAULT_CSV_NAME,
        help='CSV filename or path'
    )
    parser.add_argument(
        '--output-dir', default=str(Path(__file__).parent / 'output_plots'),
        help='Output directory for generated figures'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_csv_path(args.csv)
    print(f'Loading data from: {data_path.resolve()}')

    df = load_and_process_data(data_path)
    print(f'Processed {len(df)} step records')
    print(f'Arrival rates: {sorted(df["arrival_rate"].unique())}')

    # Generate combined 3-panel motivation figure
    output_path = output_dir / 'motivation_figure.png'
    create_motivation_figure(df, output_path)


if __name__ == '__main__':
    main()
