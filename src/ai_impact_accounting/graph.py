"""Lineage DAG construction and family rollup.

Two correctness rules drive everything here:

1. Every node reports an INCREMENTAL footprint (its own delta). The family total
   is therefore the sum over the subtree, NOT a per-node cumulative.
2. Lineage is a DAG, not a tree: a merge has multiple parents and shares
   ancestors. Summing UNIQUE nodes once (visited-set traversal) dedups merges and
   shared ancestors automatically.

We never report a bare total. The headline is ``(lower-bound interval, coverage%)``
because real-world disclosure starts at ~10-40% (paper Table A6).
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from .models import Interval, Node, Report
from .parse import impute_from_method


def build_graph(nodes: dict[str, Node]) -> nx.DiGraph:
    """Build the lineage graph with edges pointing parent -> child.

    Parameters
    ----------
    nodes : dict of str to Node
        Tracked nodes keyed by model id.

    Returns
    -------
    networkx.DiGraph
        A directed graph where ``nx.descendants(g, base)`` yields the family.
    """
    g = nx.DiGraph()
    for mid in nodes:
        g.add_node(mid)
    for mid, n in nodes.items():
        for parent in n.lineage:
            pm = parent.get("model")
            if pm:
                g.add_node(pm)
                g.add_edge(pm, mid, relation=parent.get("relation"))
    return g


def family_members(g: nx.DiGraph, base: str) -> set[str]:
    """Return the base model plus all of its descendants.

    Parameters
    ----------
    g : networkx.DiGraph
        The lineage graph from :func:`build_graph`.
    base : str
        The base model id.

    Returns
    -------
    set of str
        ``{base}`` plus every descendant; just ``{base}`` if it is absent.
    """
    if base not in g:
        return {base}
    return {base} | nx.descendants(g, base)


def rollup(nodes: dict[str, Node], base: str, impute: bool = False) -> dict:
    """Aggregate a model family's footprint, deduping the DAG.

    Parameters
    ----------
    nodes : dict of str to Node
        All tracked nodes keyed by model id.
    base : str
        The base model whose family is rolled up.
    impute : bool, optional
        When ``True``, nodes with no report are filled from compute priors and
        labelled ``imputed``; when ``False`` they are listed but contribute zero.

    Returns
    -------
    dict
        Aggregate results: ``base_footprint``, ``deriv_footprint``,
        ``total_footprint`` (each a dict of energy/carbon/water
        :class:`~ai_impact_accounting.models.Interval`), counts, ``coverage``,
        ``carbon_by_quality`` provenance subtotals, the derivative-over-base
        carbon ``ratio``, and per-node ``rows``.
    """
    g = build_graph(nodes)
    members = family_members(g, base)

    base_fp = {"energy": Interval(), "carbon": Interval(), "water": Interval()}
    deriv_fp = {"energy": Interval(), "carbon": Interval(), "water": Interval()}
    carbon_by_quality: dict[str, Interval] = {}  # tier -> carbon subtotal
    rows = []
    n_with, n_without = 0, 0

    for mid in members:  # set => each node counted once
        node = nodes.get(mid)
        rep: Optional[Report] = node.report if node else None

        if rep is None:
            n_without += 1
            if impute:
                rep = impute_from_method(Report(method=_rel_to(g, base, mid)))
            else:
                rows.append(_row(mid, None, mid == base))
                continue
        else:
            n_with += 1

        bucket = base_fp if mid == base else deriv_fp
        for k, iv in (("energy", rep.energy), ("carbon", rep.carbon), ("water", rep.water)):
            bucket[k] = bucket[k] + iv
        # Segregate the carbon subtotal by provenance so a measured run and a pile
        # of priors aren't laundered into one number.
        tier = _carbon_tier(rep.quality.get("carbon", "unavailable"))
        carbon_by_quality[tier] = carbon_by_quality.get(tier, Interval()) + rep.carbon
        rows.append(_row(mid, rep, mid == base))

    total = {k: base_fp[k] + deriv_fp[k] for k in base_fp}
    n_total = len(members)
    coverage = (n_with / n_total) if n_total else 0.0

    # the paper's headline ratio: do derivatives exceed the base?
    ratio = None
    if base_fp["carbon"].hi > 0:
        ratio = (
            deriv_fp["carbon"].lo / base_fp["carbon"].hi,
            deriv_fp["carbon"].hi / max(base_fp["carbon"].lo, 1e-9),
        )

    return {
        "base": base,
        "n_models": n_total,
        "n_with_report": n_with,
        "n_without_report": n_without,
        "coverage": coverage,
        "base_footprint": base_fp,
        "deriv_footprint": deriv_fp,
        "total_footprint": total,
        "carbon_by_quality": carbon_by_quality,
        "deriv_over_base_ratio": ratio,
        "rows": sorted(rows, key=lambda r: r["carbon_hi"], reverse=True),
        "imputed": impute,
    }


def _rel_to(g: nx.DiGraph, base: str, mid: str) -> str:
    """Infer the relation label on the shortest path from ``base`` to ``mid``."""
    try:
        path = nx.shortest_path(g, base, mid)
        return g.edges[path[-2], path[-1]].get("relation", "finetune")
    except Exception:
        return "finetune"


def _carbon_tier(q: str) -> str:
    """Collapse the per-field quality enum into provenance buckets for the subtotal."""
    if q == "measured":
        return "measured"
    if q == "imputed":
        return "imputed"
    if isinstance(q, str) and q.startswith("estimated"):
        return "estimated"
    return "unavailable"


def _row(mid: str, rep: Optional[Report], is_base: bool) -> dict:
    """Build one dashboard table row from a node report (or a missing-report stub)."""
    if rep is None:
        return {
            "model": mid,
            "role": "base" if is_base else "derivative",
            "method": "",
            "carbon": "—",
            "water": "—",
            "energy": "—",
            "quality": "no report",
            "carbon_hi": -1.0,
        }
    return {
        "model": mid,
        "role": "base" if is_base else "derivative",
        "method": rep.method or "",
        "energy": rep.energy.fmt(" kWh"),
        "carbon": rep.carbon.fmt(" kg"),
        "water": rep.water.fmt(" L"),
        "quality": rep.quality.get("carbon", ""),
        "carbon_hi": rep.carbon.hi,
    }
