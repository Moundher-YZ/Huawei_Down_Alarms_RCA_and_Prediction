# Phase 2 — Correlation Analysis Walkthrough

## What Was Built

Phase 2 implements **6 distinct correlation methods** for analyzing relationships between alarms in Djezzy's Huawei 5G network alarm data (4.6M records, 892 alarm types).

---

## Changes Made

### 1. [requirements.txt](file:///c:/Users/hp/Desktop/PFE%20Project/requirements.txt)
Added new dependencies: `scipy`, `mlxtend`, `networkx`, `pyarrow` (alongside existing ones).

```diff:requirements.txt
===
pandas
numpy
matplotlib
seaborn
plotly
scipy
mlxtend
networkx
pyarrow
```

---

### 2. [correlation.py](file:///c:/Users/hp/Desktop/PFE%20Project/src/correlation.py)
Reusable Python module with 8 functions covering 6 correlation methods:

| Function | Purpose |
|---|---|
| `build_cooccurrence_matrix()` | Binary (site×window) × alarm_type matrix |
| `compute_jaccard_similarity()` | Pairwise Jaccard similarity between alarm types |
| `compute_cooccurrence_counts()` | Raw co-occurrence counts via matrix dot product |
| `build_alarm_timeseries()` | Aggregate alarm counts into regular time series |
| `compute_temporal_crosscorrelation()` | Zero-lag + lagged Pearson correlations |
| `find_precursors()` | Non-HS → HS precursor detection with sampling |
| `build_transactions()` | Transactional format for FP-Growth |
| `compute_cramers_v()` / `compute_cramers_v_matrix()` | Cramér's V categorical correlation |
| `build_alarm_graph()` | NetworkX graph from similarity matrix |
| `compute_graph_metrics()` | Degree/betweenness/PageRank + community detection |

> [!NOTE]
> Performance-optimized for the 4.6M-row dataset: uses top-N filtering (default 30–50 alarms), sampling (5000 HS events for precursor detection), and 1M row sample for Cramér's V.

```diff:correlation.py
===
"""
src/correlation.py
==================
Reusable functions for Phase 2 — Alarm Correlation Analysis.

Provides 6 correlation methods:
  1. Co-occurrence matrix (site × time-window)
  2. Jaccard similarity between alarm pairs
  3. Temporal cross-correlation (time-series lag analysis)
  4. Sequential precursor detection (Non-HS → HS)
  5. Cramér's V / Chi-squared categorical correlation
  6. Alarm network graph construction & metrics
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from itertools import combinations
import networkx as nx


# ---------------------------------------------------------------------------
#  1. Co-occurrence
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(df, site_col='Alarmsource', alarm_col='Alarmname',
                              time_col='occurrencetime_reg', window_minutes=30,
                              top_n_alarms=50):
    """
    Build a binary (site×window) × alarm_type co-occurrence matrix.

    Parameters
    ----------
    df : DataFrame  – preprocessed alarms
    site_col : str  – column identifying the network element / site
    alarm_col : str – column with the alarm name
    time_col : str  – datetime column
    window_minutes : int – width of the time bin in minutes
    top_n_alarms : int  – keep only the N most frequent alarm types

    Returns
    -------
    pivot : DataFrame – binary matrix (rows = site×window, cols = alarm types)
    """
    tmp = df[[time_col, site_col, alarm_col]].copy()
    tmp['time_bin'] = tmp[time_col].dt.floor(f'{window_minutes}min')

    # Restrict to Top-N alarms by overall frequency
    top = tmp[alarm_col].value_counts().head(top_n_alarms).index
    tmp = tmp[tmp[alarm_col].isin(top)]

    # one row per (time_bin, site, alarm) → binary flag
    grp = (tmp.groupby(['time_bin', site_col, alarm_col])
              .size()
              .reset_index(name='_cnt'))
    grp['occurred'] = 1

    pivot = grp.pivot_table(index=['time_bin', site_col],
                            columns=alarm_col,
                            values='occurred',
                            fill_value=0)
    return pivot


def compute_jaccard_similarity(binary_matrix):
    """
    Pair-wise Jaccard similarity between the columns of a binary matrix.

    J(A,B) = |A∩B| / |A∪B|

    Returns a symmetric DataFrame.
    """
    cols = list(binary_matrix.columns)
    n = len(cols)
    mat = binary_matrix.values.astype(bool)

    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            intersection = np.sum(mat[:, i] & mat[:, j])
            union = np.sum(mat[:, i] | mat[:, j])
            sim = intersection / union if union > 0 else 0.0
            result[i, j] = sim
            result[j, i] = sim

    return pd.DataFrame(result, index=cols, columns=cols)


def compute_cooccurrence_counts(binary_matrix):
    """
    Raw co-occurrence counts: for each pair of alarm types count how many
    (site, window) rows have *both* alarms present.

    Returns a symmetric DataFrame of counts.
    """
    mat = binary_matrix.values.astype(bool)
    counts = mat.T @ mat  # dot product of boolean → co-occurrence
    return pd.DataFrame(counts,
                        index=binary_matrix.columns,
                        columns=binary_matrix.columns)


# ---------------------------------------------------------------------------
#  2. Temporal Cross-Correlation
# ---------------------------------------------------------------------------

def build_alarm_timeseries(df, alarm_col='Alarmname',
                           time_col='occurrencetime_reg',
                           freq='1h', top_n=30):
    """
    Aggregate alarm counts per type into a regular time-series DataFrame.

    Returns
    -------
    ts : DataFrame – columns = alarm types, index = time_bin, values = counts
    """
    tmp = df[[time_col, alarm_col]].copy()
    top = tmp[alarm_col].value_counts().head(top_n).index
    tmp = tmp[tmp[alarm_col].isin(top)]

    tmp['time_bin'] = tmp[time_col].dt.floor(freq)
    ts = (tmp.groupby(['time_bin', alarm_col])
             .size()
             .unstack(fill_value=0))
    return ts


def compute_temporal_crosscorrelation(ts, max_lag=24):
    """
    Compute zero-lag correlation matrix *and* per-pair lagged Pearson
    correlations for up to ±max_lag time steps.

    Parameters
    ----------
    ts : DataFrame – output of build_alarm_timeseries()
    max_lag : int  – maximum lag (in time-series index units)

    Returns
    -------
    corr_matrix : DataFrame – zero-lag Pearson correlation
    lag_df : DataFrame – columns [alarm_1, alarm_2, lag, correlation]
    """
    corr_matrix = ts.corr()

    alarms = ts.columns.tolist()
    records = []
    for i, a1 in enumerate(alarms):
        for j, a2 in enumerate(alarms):
            if i >= j:
                continue
            s1 = ts[a1].values.astype(float)
            s2 = ts[a2].values.astype(float)
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    r = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
                elif lag > 0:
                    r = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
                else:
                    r = np.corrcoef(s1, s2)[0, 1]
                if not np.isnan(r):
                    records.append({'alarm_1': a1, 'alarm_2': a2,
                                    'lag': lag, 'correlation': r})

    lag_df = pd.DataFrame(records)
    return corr_matrix, lag_df


# ---------------------------------------------------------------------------
#  3. Sequential Precursor Detection
# ---------------------------------------------------------------------------

def find_precursors(df, hs_col='is_HS', alarm_col='Alarmname',
                    site_col='Alarmsource', time_col='occurrencetime_reg',
                    hs_category_col='hs_category',
                    lookback_minutes=60, sample_hs=5000):
    """
    For each HS alarm event, look back on the same site to find Non-HS alarms
    that preceded it in the lookback window.

    To keep computation feasible on large datasets the function samples at
    most *sample_hs* HS events (randomly).

    Returns
    -------
    result : DataFrame – columns [precursor_alarm, hs_category, count, frequency]
    n_sampled : int    – number of HS events actually processed
    """
    tmp = df[[time_col, site_col, alarm_col, hs_col, hs_category_col]].copy()
    tmp = tmp.sort_values(time_col)

    hs_events = tmp[tmp[hs_col] == True]
    non_hs    = tmp[tmp[hs_col] == False]

    if len(hs_events) > sample_hs:
        hs_events = hs_events.sample(n=sample_hs, random_state=42)

    lookback = pd.Timedelta(minutes=lookback_minutes)
    precursor_counts = {}
    n_sampled = 0

    for _, row in hs_events.iterrows():
        site = row[site_col]
        t    = row[time_col]
        cat  = row[hs_category_col]

        mask = ((non_hs[site_col] == site) &
                (non_hs[time_col] >= t - lookback) &
                (non_hs[time_col] < t))
        precursors = non_hs.loc[mask, alarm_col].unique()

        for p in precursors:
            key = (p, cat)
            precursor_counts[key] = precursor_counts.get(key, 0) + 1

        n_sampled += 1

    records = [{'precursor_alarm': k[0], 'hs_category': k[1],
                'count': v, 'frequency': v / n_sampled}
               for k, v in precursor_counts.items()]

    result = pd.DataFrame(records)
    if len(result) > 0:
        result = result.sort_values('count', ascending=False)
    return result, n_sampled


# ---------------------------------------------------------------------------
#  4. Association-Rule helpers (for use with mlxtend)
# ---------------------------------------------------------------------------

def build_transactions(df, site_col='Alarmsource', alarm_col='Alarmname',
                       time_col='occurrencetime_reg', window_minutes=30,
                       top_n_alarms=50, min_items=2):
    """
    Convert alarm data into transactional format suitable for FP-Growth.

    Each transaction = unique alarm types on a single site within one time bin.
    Transactions with fewer than *min_items* alarms are dropped.

    Returns
    -------
    transactions_df : DataFrame – one-hot encoded (mlxtend format)
    """
    pivot = build_cooccurrence_matrix(df, site_col, alarm_col,
                                      time_col, window_minutes,
                                      top_n_alarms)
    # keep only rows with ≥ min_items alarm types
    row_sums = pivot.sum(axis=1)
    pivot = pivot[row_sums >= min_items]

    # ensure boolean for mlxtend
    return pivot.astype(bool)


# ---------------------------------------------------------------------------
#  5. Statistical Correlation  (Cramér's V / Chi-squared)
# ---------------------------------------------------------------------------

def compute_cramers_v(x, y):
    """
    Cramér's V  –  effect-size measure of association between two
    categorical variables (based on chi-squared).
    """
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def compute_cramers_v_matrix(df, columns):
    """
    Pair-wise Cramér's V for a list of categorical columns.

    Returns a symmetric DataFrame.
    """
    n = len(columns)
    mat = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)
    for i in range(n):
        for j in range(i, n):
            v = compute_cramers_v(df[columns[i]], df[columns[j]])
            mat.iloc[i, j] = v
            mat.iloc[j, i] = v
    return mat


# ---------------------------------------------------------------------------
#  6. Network / Graph Analysis
# ---------------------------------------------------------------------------

def build_alarm_graph(similarity_matrix, threshold=0.05):
    """
    Build a weighted undirected NetworkX graph from a similarity matrix.
    Only edges above *threshold* are included.
    """
    G = nx.Graph()
    cols = list(similarity_matrix.columns)
    for c in cols:
        G.add_node(c)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            w = similarity_matrix.iloc[i, j]
            if abs(w) >= threshold:
                G.add_edge(cols[i], cols[j], weight=float(w))
    return G


def compute_graph_metrics(G):
    """
    Compute degree centrality, betweenness centrality, PageRank, and
    community labels for an alarm graph.

    Returns a DataFrame indexed by alarm type.
    """
    if len(G.nodes()) == 0:
        return pd.DataFrame()

    data = {
        'degree_centrality':     nx.degree_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G, weight='weight'),
        'pagerank':              nx.pagerank(G, weight='weight'),
    }

    # community detection (greedy modularity)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G, weight='weight'))
        comm_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                comm_map[node] = idx
        data['community'] = comm_map
    except Exception:
        data['community'] = {n: 0 for n in G.nodes()}

    return pd.DataFrame(data)
```

---

### 3. [02_correlation.ipynb](file:///c:/Users/hp/Desktop/PFE%20Project/notebooks/02_correlation.ipynb)
Comprehensive Jupyter notebook with **36 cells** across 8 sections:

| Section | Content |
|---|---|
| **0. Setup** | Imports, configuration, data loading with HS label reconstruction |
| **1. Co-occurrence** | Jaccard similarity heatmap, top co-occurring alarm pairs |
| **2. Temporal** | Hourly time series correlation matrix + lag analysis with visualization |
| **3. Precursors** | Non-HS → HS precursor rankings per HS category, bar charts |
| **4. Association Rules** | FP-Growth frequent itemsets, rules with support/confidence/lift |
| **5. Cramér's V** | Categorical attribute correlation matrix + Chi-squared tests |
| **6. Network Graph** | Alarm graph centrality metrics, community detection, interactive Plotly network |
| **7. Summary** | Feature export summary, file listing |

**Output files** (saved to `data/features/`):
- `jaccard_similarity.csv`
- `cooccurrence_pairs.csv`
- `temporal_correlation_matrix.csv`
- `temporal_lag_correlations.csv`
- `precursor_rankings.csv`
- `association_rules.csv`
- `cramers_v_matrix.csv`
- `chi2_tests.csv`
- `network_metrics.csv`

**Figures** (saved to `reports/figures/`):
- `corr_jaccard_heatmap.png`
- `corr_temporal_heatmap.png`
- `corr_lag_top_pair.png`
- `corr_precursors_per_hs.png`
- `corr_association_rules_lift.png`
- `corr_cramers_v.png`
- `corr_network_centrality.png`

---

## Verification

| Check | Result |
|---|---|
| `02_correlation.ipynb` valid JSON | ✅ 36 cells, nbformat 4 |
| `src/correlation.py` compiles | ✅ No syntax errors |
| All packages installed | ✅ networkx, mlxtend, pyarrow installed |
| All imports succeed | ✅ Core packages + correlation module |

> [!IMPORTANT]
> The notebook must be **run** against the actual dataset to generate the output files. Open `notebooks/02_correlation.ipynb` in Jupyter and execute all cells. The preprocessed parquet at `data/processed/alarms_processed.parquet` is loaded automatically.
