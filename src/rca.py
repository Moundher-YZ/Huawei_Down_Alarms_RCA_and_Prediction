"""
src/rca.py
==========
Reusable functions for Phase 3 — Root Cause Analysis.

Provides:
  1. RCA evidence aggregator   – fuse precursor, Jaccard, association-rule
                                 and network-centrality signals into a single
                                 ranked evidence table.
  2. Causal chain builder      – reconstruct multi-hop alarm propagation paths
                                 from the lag-correlation data.
  3. Site-level RCA            – identify the most alarm-active / at-risk sites.
  4. Temporal pattern analysis – extract hour-of-day / day-of-week failure
                                 patterns for each HS category.
  5. RCA report generator      – produce a structured Markdown summary.
"""

import pandas as pd
import numpy as np
from itertools import combinations


# ---------------------------------------------------------------------------
# 1. Evidence Aggregator
# ---------------------------------------------------------------------------

def build_rca_evidence(
    precursor_df: pd.DataFrame,
    cooc_pairs_df: pd.DataFrame,
    assoc_rules_df: pd.DataFrame,
    network_metrics_df: pd.DataFrame,
    w_precursor: float = 0.40,
    w_jaccard:   float = 0.20,
    w_lift:      float = 0.25,
    w_pagerank:  float = 0.15,
) -> pd.DataFrame:
    """
    Fuse four evidence sources into one ranked evidence table.

    Each alarm that appears as a precursor / co-occurring / antecedent alarm
    receives a composite score based on weighted, min-max-normalised signals.

    Parameters
    ----------
    precursor_df      : output of find_precursors()  — [precursor_alarm, hs_category, count, frequency]
    cooc_pairs_df     : co-occurrence pairs           — [alarm_A, alarm_B, cooccurrence_count, jaccard]
    assoc_rules_df    : association rules             — [antecedents_str, consequents_str, support, confidence, lift]
    network_metrics_df: graph metrics                 — [Unnamed:0 / alarm_name, degree_centrality, betweenness_centrality, pagerank, community]
    w_*               : weights (must sum to 1.0)

    Returns
    -------
    evidence : DataFrame – [alarm, hs_category, precursor_score, jaccard_score,
                            lift_score, pagerank_score, composite_score] sorted desc
    """

    # ── normalise helper ──────────────────────────────────────
    def _minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.0, index=s.index)

    # ── 1. Precursor signal (frequency per HS category) ──────
    prec = precursor_df[['precursor_alarm', 'hs_category', 'frequency']].copy()
    prec = prec.rename(columns={'precursor_alarm': 'alarm'})
    prec['precursor_score'] = _minmax(prec['frequency'])

    # ── 2. Jaccard / co-occurrence signal ────────────────────
    # Stack alarm_A and alarm_B so every alarm gets its best Jaccard score
    jac_a = cooc_pairs_df[['alarm_A', 'jaccard']].rename(columns={'alarm_A': 'alarm'})
    jac_b = cooc_pairs_df[['alarm_B', 'jaccard']].rename(columns={'alarm_B': 'alarm'})
    jac_all = pd.concat([jac_a, jac_b], ignore_index=True)
    jac_max = jac_all.groupby('alarm')['jaccard'].max().reset_index()
    jac_max['jaccard_score'] = _minmax(jac_max['jaccard'])

    # ── 3. Association-rule lift signal ───────────────────────
    # Expand antecedents (may be comma-separated lists)
    lift_rows = []
    for _, row in assoc_rules_df.iterrows():
        ants = [a.strip() for a in str(row['antecedents_str']).split(',')]
        for ant in ants:
            lift_rows.append({'alarm': ant, 'lift': row['lift']})
    if lift_rows:
        lift_df = pd.DataFrame(lift_rows)
        lift_max = lift_df.groupby('alarm')['lift'].max().reset_index()
        lift_max['lift_score'] = _minmax(lift_max['lift'])
    else:
        lift_max = pd.DataFrame(columns=['alarm', 'lift', 'lift_score'])

    # ── 4. PageRank signal ────────────────────────────────────
    nm = network_metrics_df.copy()
    alarm_col = 'Unnamed: 0' if 'Unnamed: 0' in nm.columns else nm.columns[0]
    nm = nm.rename(columns={alarm_col: 'alarm'})
    nm['pagerank_score'] = _minmax(nm['pagerank'])
    pr = nm[['alarm', 'pagerank_score']]

    # ── Merge everything on (alarm, hs_category) ─────────────
    base = prec.copy()

    base = base.merge(jac_max[['alarm', 'jaccard_score']], on='alarm', how='left')
    base = base.merge(lift_max[['alarm', 'lift_score']],   on='alarm', how='left')
    base = base.merge(pr,                                   on='alarm', how='left')

    for col in ['jaccard_score', 'lift_score', 'pagerank_score']:
        base[col] = base[col].fillna(0.0)

    base['composite_score'] = (
        w_precursor * base['precursor_score'] +
        w_jaccard   * base['jaccard_score']   +
        w_lift      * base['lift_score']      +
        w_pagerank  * base['pagerank_score']
    )

    return (base
            .sort_values('composite_score', ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 2. Causal Chain Builder
# ---------------------------------------------------------------------------

def build_causal_chains(
    lag_df: pd.DataFrame,
    precursor_df: pd.DataFrame,
    hs_category: str,
    top_n_precursors: int = 5,
    top_n_lag: int = 3,
    max_lag_hours: int = 6,
) -> list[dict]:
    """
    Reconstruct multi-hop causal chains for a given HS category.

    Strategy
    --------
    1. Start from the top-N precursors for *hs_category*.
    2. For each precursor, look up which alarms are most correlated with it
       at a *negative* lag (i.e., they precede the precursor) in *lag_df*.
    3. Return a list of chain dicts: {'root': ..., 'intermediate': ..., 'terminal': ...}

    Parameters
    ----------
    lag_df            : temporal lag correlations DataFrame
    precursor_df      : precursor rankings DataFrame
    hs_category       : e.g. '3G cell down'
    top_n_precursors  : how many precursors to chain from
    top_n_lag         : max upstream alarms per hop
    max_lag_hours     : only consider lags within this range (positive = earlier)

    Returns
    -------
    chains : list of dicts
    """
    sub = precursor_df[precursor_df['hs_category'] == hs_category]
    if sub.empty:
        return []

    top_prec = sub.head(top_n_precursors)['precursor_alarm'].tolist()

    chains = []
    for prec_alarm in top_prec:
        # find alarms that *lead* the precursor (negative lag → alarm_2 leads alarm_1)
        mask = (
            ((lag_df['alarm_2'] == prec_alarm) & (lag_df['lag'] < 0) & (lag_df['lag'] >= -max_lag_hours)) |
            ((lag_df['alarm_1'] == prec_alarm) & (lag_df['lag'] > 0) & (lag_df['lag'] <=  max_lag_hours))
        )
        sub_lag = lag_df[mask].copy()

        if sub_lag.empty:
            chains.append({'root': None, 'intermediate': prec_alarm,
                           'terminal': hs_category, 'correlation': None})
            continue

        sub_lag['abs_corr'] = sub_lag['correlation'].abs()
        best = sub_lag.sort_values('abs_corr', ascending=False).head(top_n_lag)

        for _, row in best.iterrows():
            upstream = row['alarm_1'] if row['alarm_2'] == prec_alarm else row['alarm_2']
            chains.append({
                'root':         upstream,
                'intermediate': prec_alarm,
                'terminal':     hs_category,
                'correlation':  round(row['abs_corr'], 4),
                'lag_hours':    abs(row['lag']),
            })

    return chains


# ---------------------------------------------------------------------------
# 3. Site-Level RCA
# ---------------------------------------------------------------------------

def compute_site_risk(
    df: pd.DataFrame,
    site_col:     str = 'Alarmsource',
    hs_col:       str = 'is_HS',
    alarm_col:    str = 'Alarmname',
    time_col:     str = 'occurrencetime_reg',
    top_n_sites:  int = 20,
) -> pd.DataFrame:
    """
    Rank sites by their HS alarm load, diversity, and recency.

    Returns a DataFrame with one row per site and columns:
      total_alarms, hs_count, hs_rate, unique_hs_types, risk_score
    """
    grp = df.groupby(site_col).agg(
        total_alarms   = (alarm_col, 'count'),
        hs_count       = (hs_col,    'sum'),
        unique_alarms  = (alarm_col, 'nunique'),
    ).reset_index()

    grp['hs_rate'] = grp['hs_count'] / grp['total_alarms'].clip(lower=1)

    # unique HS alarm types per site
    hs_types = (df[df[hs_col]]
                .groupby(site_col)[alarm_col]
                .nunique()
                .reset_index()
                .rename(columns={alarm_col: 'unique_hs_types'}))
    grp = grp.merge(hs_types, on=site_col, how='left')
    grp['unique_hs_types'] = grp['unique_hs_types'].fillna(0).astype(int)

    # composite risk score
    def _mm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else 0.0

    grp['risk_score'] = (
        0.50 * _mm(grp['hs_count']) +
        0.30 * _mm(grp['hs_rate'])  +
        0.20 * _mm(grp['unique_hs_types'])
    )

    return (grp.sort_values('risk_score', ascending=False)
               .head(top_n_sites)
               .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 4. Temporal Pattern Analysis
# ---------------------------------------------------------------------------

def compute_temporal_patterns(
    df: pd.DataFrame,
    hs_category_col: str = 'hs_category',
    time_col:        str = 'occurrencetime_reg',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute hour-of-day and day-of-week failure distributions for each HS category.

    Returns
    -------
    hourly_df  : DataFrame [hs_category, hour, count]
    daily_df   : DataFrame [hs_category, dayofweek, count]
    """
    hs_df = df[df[hs_category_col] != 'Non-HS'].copy()

    hs_df['hour']      = hs_df[time_col].dt.hour
    hs_df['dayofweek'] = hs_df[time_col].dt.dayofweek

    hourly_df = (hs_df.groupby([hs_category_col, 'hour'])
                       .size()
                       .reset_index(name='count'))

    daily_df  = (hs_df.groupby([hs_category_col, 'dayofweek'])
                       .size()
                       .reset_index(name='count'))

    return hourly_df, daily_df


# ---------------------------------------------------------------------------
# 5. Markdown Report Generator
# ---------------------------------------------------------------------------

def generate_rca_report(
    evidence_df:   pd.DataFrame,
    chains:        list[dict],
    site_risk_df:  pd.DataFrame,
    output_path:   str,
) -> str:
    """
    Write a structured Markdown RCA report and return the content as a string.

    Parameters
    ----------
    evidence_df  : output of build_rca_evidence()
    chains       : output of build_causal_chains() (any HS category)
    site_risk_df : output of compute_site_risk()
    output_path  : where to save the .md file
    """
    lines = []
    lines.append("# Root Cause Analysis Report\n")
    lines.append("**Project:** AI-based 5G Alarm Analysis — Djezzy  \n")
    lines.append("**Phase:** 03 — Root Cause Analysis  \n\n")
    lines.append("---\n")

    # Top causes per HS category
    lines.append("## 1. Top Root Causes per HS Category\n")
    for cat in evidence_df['hs_category'].unique():
        sub = evidence_df[evidence_df['hs_category'] == cat].head(5)
        lines.append(f"\n### {cat}\n")
        lines.append("| Rank | Alarm | Composite Score |\n")
        lines.append("|------|-------|-----------------|\n")
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            lines.append(f"| {rank} | {row['alarm']} | {row['composite_score']:.3f} |\n")

    # Causal chains
    lines.append("\n---\n## 2. Causal Chains\n")
    if chains:
        lines.append("| Root Alarm | Intermediate | Terminal (HS) | Corr | Lag (h) |\n")
        lines.append("|------------|-------------|--------------|------|----------|\n")
        for c in chains[:20]:
            root = c.get('root') or '—'
            corr = f"{c['correlation']:.3f}" if c.get('correlation') else '—'
            lag  = str(c.get('lag_hours', '—'))
            lines.append(f"| {root} | {c['intermediate']} | {c['terminal']} | {corr} | {lag} |\n")
    else:
        lines.append("No chains found.\n")

    # High-risk sites
    lines.append("\n---\n## 3. High-Risk Sites\n")
    lines.append("| Rank | Site | HS Count | HS Rate | Risk Score |\n")
    lines.append("|------|------|----------|---------|------------|\n")
    site_col = site_risk_df.columns[0]
    for rank, (_, row) in enumerate(site_risk_df.head(10).iterrows(), 1):
        lines.append(
            f"| {rank} | {row[site_col]} | {int(row['hs_count'])} "
            f"| {row['hs_rate']:.2%} | {row['risk_score']:.3f} |\n"
        )

    report = "".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report
