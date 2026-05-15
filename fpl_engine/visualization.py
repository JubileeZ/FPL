import pandas as pd
import numpy as np
from datetime import datetime
import optuna
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

def _render_sync_plots(
    active_df: pd.DataFrame,
    mapping_table: pd.DataFrame,
    upside_metric: str,
    differential_weight: float,
    upside_weight_clipped: float,
):
    """
    Renders a 6-panel diagnostic dashboard for create_optimized_custom_score.

    Panel layout:
      [0] Perf_IDX vs custom_score scatter       — shows upside blend effect per player
      [1] dynamic_upside distribution by position — who gets the most upside weight
      [2] ceiling_gap vs gap_pct scatter          — ceiling gap rank distribution
      [3] score_std by position (box plot)        — variance spread per position
      [4] S-curve: ownership vs differential      — how the differential signal maps
      [5] Mapping table: position-level summary   — CV, weights, avg gap

    Parameters
    ----------
    active_df            : filtered dataframe (active players only)
    mapping_table        : per-position summary produced by create_optimized_custom_score
    upside_metric        : column name of the ceiling score ('ceiling_score')
    differential_weight  : user-supplied differential weight (0 = no differential panel)
    upside_weight_clipped: actual clipped upside weight used in the blend
    """

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor('#1a1a2e')

    POSITION_COLOURS = {
        'GKP': '#f5a623',   # amber
        'DEF': '#4a90d9',   # blue
        'MID': '#7ed321',   # green
        'FWD': '#e74c3c',   # red
    }
    DEFAULT_COLOUR = '#aaaaaa'



    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.35,
        left=0.06, right=0.97,
        top=0.91,  bottom=0.07,
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # Style every axis consistently
    for ax in axes:
        ax.set_facecolor('#0f0f23')
        ax.tick_params(colors='#cccccc', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # Position colour map per row
    pos_colours = active_df['position'].map(POSITION_COLOURS).fillna(DEFAULT_COLOUR) \
        if 'position' in active_df.columns else DEFAULT_COLOUR

    positions = sorted(active_df['position'].unique()) if 'position' in active_df.columns else []

    # =========================================================================
    # PANEL 0 — Perf_IDX vs custom_score
    # Shows how the upside blend shifts each player's score above their base.
    # Points above the diagonal = players benefiting from upside blend.
    # =========================================================================
    ax = axes[0]

    ax.scatter(
        active_df['Perf_IDX'],
        active_df['custom_score'],
        c=pos_colours,
        alpha=0.65,
        s=22,
        linewidths=0,
    )

    # Diagonal reference line (custom_score = Perf_IDX, i.e. no upside premium)
    lim_min = min(active_df['Perf_IDX'].min(), active_df['custom_score'].min()) - 0.2
    lim_max = max(active_df['Perf_IDX'].max(), active_df['custom_score'].max()) + 0.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color='#555577', lw=1, ls='--', label='No premium')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # Annotate top 5 players by upside premium
    if 'web_name' in active_df.columns:
        premium   = active_df['custom_score'] - active_df['Perf_IDX']
        top5_idx  = premium.nlargest(5).index
        for idx in top5_idx:
            ax.annotate(
                active_df.loc[idx, 'web_name'],
                xy=(active_df.loc[idx, 'Perf_IDX'], active_df.loc[idx, 'custom_score']),
                xytext=(4, 2), textcoords='offset points',
                fontsize=6.5, color='#ffffff', alpha=0.85,
            )

    _style_panel(ax, 'Perf_IDX vs Custom Score', 'Perf_IDX (expected pts)', 'Custom Score')
    ax.legend(handles=[
        Line2D([0], [0], color='#555577', ls='--', lw=1, label='No upside premium')
    ], fontsize=7, facecolor='#1a1a2e', labelcolor='#cccccc', framealpha=0.6)

    # =========================================================================
    # PANEL 1 — Dynamic Upside Weight Distribution by Position
    # Shows the distribution of dynamic_upside within each position group.
    # A wider spread = more variation in how much ceiling is blended per position.
    # =========================================================================
    ax = axes[1]

    if positions:
        violin_data   = [active_df.loc[active_df['position'] == pos, 'dynamic_upside'].dropna().values
                         for pos in positions]
        violin_colours = [POSITION_COLOURS.get(pos, DEFAULT_COLOUR) for pos in positions]

        parts = ax.violinplot(violin_data, positions=range(len(positions)),
                              showmedians=True, showextrema=False)

        for i, (body, colour) in enumerate(zip(parts['bodies'], violin_colours)):
            body.set_facecolor(colour)
            body.set_alpha(0.65)
        parts['cmedians'].set_colors('#ffffff')
        parts['cmedians'].set_linewidth(1.5)

        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, color='#cccccc', fontsize=9)

        # Draw the global upside_weight_clipped as a reference line
        ax.axhline(upside_weight_clipped, color='#f5a623', lw=1, ls='--',
                   label=f'Base weight ({upside_weight_clipped:.3f})')
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='#cccccc', framealpha=0.6)

    _style_panel(ax, 'Dynamic Upside Weight by Position', 'Position', 'dynamic_upside')
    ax.set_ylim(bottom=0)

    # =========================================================================
    # PANEL 2 — Ceiling Gap vs Gap Percentile
    # Shows how ceiling_gap (absolute pts room) translates to gap_pct (rank).
    # Players in the top-right corner get the most upside blend.
    # =========================================================================
    ax = axes[2]

    ax.scatter(
        active_df['gap_pct'],
        active_df['ceiling_gap'],
        c=pos_colours,
        alpha=0.60,
        s=20,
        linewidths=0,
    )

    # Reference line at the median gap_pct
    ax.axvline(0.5, color='#555577', lw=1, ls='--')
    ax.text(0.51, active_df['ceiling_gap'].max() * 0.95, 'Median',
            color='#888899', fontsize=7)

    # Annotate top 5 by ceiling_gap
    if 'web_name' in active_df.columns:
        top5_gap = active_df['ceiling_gap'].nlargest(5).index
        for idx in top5_gap:
            ax.annotate(
                active_df.loc[idx, 'web_name'],
                xy=(active_df.loc[idx, 'gap_pct'], active_df.loc[idx, 'ceiling_gap']),
                xytext=(4, 2), textcoords='offset points',
                fontsize=6.5, color='#ffffff', alpha=0.85,
            )

    _style_panel(ax, 'Ceiling Gap vs Gap Percentile', 'gap_pct (rank [0,1])', 'ceiling_gap (pts)')

    # =========================================================================
    # PANEL 3 — Score Standard Deviation by Position (box plot)
    # Shows the spread of total_std (from variance aggregation in ceiling_score).
    # Higher std = higher boom-bust potential within that position.
    # Only shown when score_std column exists (from _calculate_performance_indices).
    # =========================================================================
    ax = axes[3]

    if 'score_std' in active_df.columns and positions:
        box_data    = [active_df.loc[active_df['position'] == pos, 'score_std'].dropna().values
                       for pos in positions]
        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            medianprops=dict(color='white', linewidth=1.5),
            whiskerprops=dict(color='#777799'),
            capprops=dict(color='#777799'),
            flierprops=dict(marker='o', markersize=3, alpha=0.4,
                            markerfacecolor='#aaaaaa', markeredgewidth=0),
        )
        for patch, pos in zip(bp['boxes'], positions):
            patch.set_facecolor(POSITION_COLOURS.get(pos, DEFAULT_COLOUR))
            patch.set_alpha(0.70)

        ax.set_xticks(range(1, len(positions) + 1))
        ax.set_xticklabels(positions, color='#cccccc', fontsize=9)
        _style_panel(ax, 'Score Std Dev by Position\n(ceiling_score variance)', 'Position', 'score_std (pts)')
    else:
        ax.text(0.5, 0.5, 'score_std not available\n(run _calculate_performance_indices first)',
                ha='center', va='center', color='#888899', fontsize=9,
                transform=ax.transAxes)
        _style_panel(ax, 'Score Std Dev by Position', '', '')

    # =========================================================================
    # PANEL 4 — S-Curve: Ownership vs Differential Bonus
    # Shows how unowned_potential maps ownership % to a differential multiplier.
    # Useful for understanding at what ownership level the bonus kicks in.
    # Only meaningful when differential_weight > 0.
    # =========================================================================
    ax = axes[4]

    if differential_weight > 0 and 'selected_by_percent' in active_df.columns:
        ownership_range      = np.linspace(0, 1, 300)
        midpoint             = 0.18   # default — matches s_curve_midpoint in parent fn
        steepness            = 18.0   # default — matches s_curve_steepness in parent fn
        unowned_curve        = 1 / (1 + np.exp(steepness * (ownership_range - midpoint)))
        raw_bonus_curve      = differential_weight * unowned_curve

        ax.plot(ownership_range * 100, np.clip(raw_bonus_curve, 0, 0.15),
                color='#7ed321', lw=2, label='Bonus multiplier (clipped 0.15)')
        ax.axhline(0.15, color='#e74c3c', lw=1, ls='--', label='Hard cap (0.15)')
        ax.axvline(midpoint * 100, color='#f5a623', lw=1, ls='--',
                   label=f'Midpoint ({midpoint*100:.0f}%)')

        # Scatter actual player ownership vs their raw_bonus_multiplier
        if 'raw_bonus_multiplier' in active_df.columns:
            ax.scatter(
                active_df['selected_by_percent'],
                active_df['raw_bonus_multiplier'],
                c=pos_colours, alpha=0.50, s=15, linewidths=0, zorder=3,
            )

        ax.set_xlim(0, 80)
        ax.set_ylim(0, 0.17)
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='#cccccc', framealpha=0.6)
        _style_panel(ax, 'S-Curve: Ownership → Differential Bonus',
                     'Ownership (%)', 'raw_bonus_multiplier')
    else:
        ax.text(0.5, 0.5,
                'Differential weight = 0\nor selected_by_percent missing',
                ha='center', va='center', color='#888899', fontsize=9,
                transform=ax.transAxes)
        _style_panel(ax, 'S-Curve: Ownership → Differential Bonus', '', '')

    # =========================================================================
    # PANEL 5 — Position Mapping Summary Table
    # Renders the mapping_table as a formatted table inside the subplot.
    # Columns: avg gap ratio, avg dynamic upside, score std, smoothed CV, diff weight.
    # =========================================================================
    ax = axes[5]
    ax.axis('off')

    if mapping_table is not None and not mapping_table.empty:
        table_data   = mapping_table.reset_index()
        col_labels   = list(table_data.columns)
        cell_text    = table_data.values.tolist()

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.15, 1.6)

        # Style header row
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor('#2a2a4a')
            tbl[0, j].set_text_props(color='#f5a623', fontweight='bold')

        # Style data rows with alternating shades + position colour left column
        for i, row in enumerate(cell_text):
            bg = '#12122a' if i % 2 == 0 else '#1a1a2e'
            pos_val = str(row[0])
            for j in range(len(col_labels)):
                cell = tbl[i + 1, j]
                cell.set_facecolor(bg)
                cell.set_text_props(color='#dddddd')
                if j == 0:   # position column — use position colour
                    cell.set_text_props(
                        color=POSITION_COLOURS.get(pos_val, '#ffffff'),
                        fontweight='bold'
                    )

        ax.set_title('Position Mapping Summary', color='#dddddd', fontsize=9,
                     pad=6, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No mapping table available',
                ha='center', va='center', color='#888899', fontsize=9,
                transform=ax.transAxes)

    # =========================================================================
    # Position legend (shared across scatter panels)
    # =========================================================================
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=pos,
               markerfacecolor=POSITION_COLOURS.get(pos, DEFAULT_COLOUR),
               markersize=7)
        for pos in positions
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc='upper right',
            bbox_to_anchor=(0.99, 0.99),
            fontsize=8,
            facecolor='#1a1a2e',
            labelcolor='#cccccc',
            framealpha=0.7,
            ncol=len(positions),
        )

    fig.suptitle(
        f'Custom Score Diagnostic Dashboard  '
        f'(upside_weight={upside_weight_clipped:.3f}  |  '
        f'differential_weight={differential_weight:.3f})',
        color='#ffffff', fontsize=13, fontweight='bold', y=0.975,
    )

    plt.show()


def _style_panel(ax, title: str, xlabel: str, ylabel: str):
    """Apply consistent dark-theme styling to a single subplot."""
    ax.set_title(title, color='#dddddd', fontsize=9, pad=5, fontweight='bold')
    ax.set_xlabel(xlabel, color='#aaaaaa', fontsize=8)
    ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=8)
    ax.tick_params(colors='#aaaaaa', labelsize=7.5)
    ax.grid(True, color='#2a2a4a', linewidth=0.5, alpha=0.7)

# --- CELL 39 ---
