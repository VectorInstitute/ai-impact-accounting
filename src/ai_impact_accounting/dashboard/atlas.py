"""Interactive lineage graph for the impact dashboard.

Renders the lineage DAG as a Plotly network. Default view shows every model in
the ingested dataset; optional family scope highlights one base subtree.
"""

from __future__ import annotations

import json
import math
from collections import deque
from typing import Any, Literal, Optional

import networkx as nx
import plotly.graph_objects as go

from ..graph import build_graph, family_members, rollup
from ..models import Node, Report


GraphView = Literal["all", "family"]

GREEN = "#2e7d52"
AMBER = "#c9a017"
GREY = "#9ca3af"
NONE = "#cbd5e1"
BASE_RING = "#1a5c35"
HIGHLIGHT = "#1e6a8a"
CHART_INK = "#1a1a1a"
EDGE_COLOR = "rgba(100,100,100,0.45)"
LABEL_LIMIT = 25


def _quality_color(quality: str) -> str:
    if quality == "no report":
        return NONE
    if quality == "measured":
        return GREEN
    if quality == "imputed":
        return GREY
    if quality.startswith("estimated"):
        return AMBER
    return NONE


def _node_size(carbon_hi: float, highlighted: bool) -> float:
    if highlighted:
        return 28.0
    if carbon_hi <= 0:
        return 16.0
    return min(16.0 + carbon_hi * 8000.0, 26.0)


def _carbon_hi_from_label(carbon: str) -> float:
    if carbon == "—":
        return -1.0
    try:
        return float(str(carbon).split()[0])
    except ValueError:
        return 0.0


def _row_from_store(mid: str, node: Optional[Node], highlight: str) -> dict[str, Any]:
    """Build display fields for one graph node from store state."""
    if mid == "scratch":
        return {
            "model": mid,
            "role": "placeholder",
            "method": "",
            "carbon": "—",
            "water": "—",
            "energy": "—",
            "quality": "placeholder",
        }
    if node is None:
        return {
            "model": mid,
            "role": "lineage parent",
            "method": "",
            "carbon": "—",
            "water": "—",
            "energy": "—",
            "quality": "not in dataset",
        }
    rep: Optional[Report] = node.report
    if rep is None:
        return {
            "model": mid,
            "role": "in dataset",
            "method": "",
            "carbon": "—",
            "water": "—",
            "energy": "—",
            "quality": "no report",
        }
    return {
        "model": mid,
        "role": "base" if mid == highlight else "derivative",
        "method": rep.method or "",
        "carbon": rep.carbon.fmt(" kg"),
        "water": rep.water.fmt(" L"),
        "energy": rep.energy.fmt(" kWh"),
        "quality": rep.quality.get("carbon", ""),
    }


MIN_NODE_SEP = 92.0


def _resolve_collisions(
    pos: dict[str, tuple[float, float]],
    *,
    min_dist: float = MIN_NODE_SEP,
    iterations: int = 96,
) -> dict[str, tuple[float, float]]:
    """Iteratively separate nodes that sit closer than *min_dist*."""
    if len(pos) < 2:
        return pos

    nodes = list(pos.keys())
    out: dict[str, list[float]] = {n: [float(pos[n][0]), float(pos[n][1])] for n in nodes}

    for _ in range(iterations):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                ax, ay = out[a]
                bx, by = out[b]
                dx = bx - ax
                dy = by - ay
                dist = math.hypot(dx, dy)
                if dist < 1e-6:
                    angle = ((i + j) * 0.618033988) % (2.0 * math.pi)
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                    dist = 1e-3
                if dist < min_dist:
                    push = (min_dist - dist) / 2.0
                    ux, uy = dx / dist, dy / dist
                    out[a][0] -= ux * push
                    out[a][1] -= uy * push
                    out[b][0] += ux * push
                    out[b][1] += uy * push
                    moved = True
        if not moved:
            break

    return {n: (out[n][0], out[n][1]) for n in nodes}


def _radial_layout_component(
    g: nx.DiGraph,
    *,
    radius_step: float = 150.0,
) -> dict[str, tuple[float, float]]:
    """Radial tree: each parent's children fan on an arc in the outward direction.

    Unlike global depth-rings, a lone child continues straight from its parent
    instead of snapping to a fixed compass point on a shared circle.
    """
    if not g.nodes:
        return {}

    roots = sorted(n for n in g.nodes if g.in_degree(n) == 0)
    if not roots:
        roots = [min(g.nodes)]

    pos: dict[str, tuple[float, float]] = {}
    if len(roots) == 1:
        pos[roots[0]] = (0.0, 0.0)
    else:
        inner = radius_step * 0.32
        for i, root in enumerate(roots):
            angle = (2.0 * math.pi * i / len(roots)) - math.pi / 2
            pos[root] = (inner * math.cos(angle), inner * math.sin(angle))

    q: deque[str] = deque(roots)
    while q:
        parent = q.popleft()
        children = sorted(g.successors(parent))
        if not children:
            continue

        px, py = pos[parent]
        preds = [p for p in g.predecessors(parent) if p in pos]
        if preds:
            gx, gy = pos[preds[0]]
            outward = math.atan2(py - gy, px - gx)
        else:
            outward = math.pi / 2  # roots branch downward

        n = len(children)
        if n == 1:
            angles = [outward]
        else:
            step = radius_step * (1.0 + 0.06 * max(n - 3, 0))
            max_span = math.pi * 1.35
            chord = MIN_NODE_SEP
            half = math.asin(min(chord / (2.0 * step), 1.0))
            min_span = 2.0 * half * (n - 1)
            while min_span > max_span and step < radius_step * 2.8:
                step *= 1.1
                half = math.asin(min(chord / (2.0 * step), 1.0))
                min_span = 2.0 * half * (n - 1)
            span = min(
                max_span,
                max(
                    min_span,
                    math.pi * 0.4,
                    math.pi * 0.2 * (n - 1) + math.pi * 0.25,
                ),
            )
            start = outward - span / 2
            angles = [start + span * i / (n - 1) for i in range(n)]

        for child, angle in zip(children, angles):
            if child in pos:
                continue
            if n == 1:
                step = radius_step
            pos[child] = (
                px + step * math.cos(angle),
                py + step * math.sin(angle),
            )
            q.append(child)

    for node in g.nodes:
        pos.setdefault(node, (0.0, 0.0))
    return pos


def _layered_layout_component(
    g: nx.DiGraph,
    *,
    x_gap: float = 240.0,
    y_gap: float = 100.0,
) -> dict[str, tuple[float, float]]:
    """Lay one weakly-connected DAG out left-to-right by depth from roots."""
    if not g.nodes:
        return {}

    depth: dict[str, int] = {}
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    if not roots:
        roots = [min(g.nodes)]

    q: deque[str] = deque(roots)
    for root in roots:
        depth[root] = 0
    while q:
        u = q.popleft()
        for v in g.successors(u):
            depth[v] = max(depth.get(v, 0), depth[u] + 1)
            q.append(v)

    for node in g.nodes:
        depth.setdefault(node, 0)

    levels: dict[int, list[str]] = {}
    for node, level in depth.items():
        levels.setdefault(level, []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    # Per-level vertical spacing (not global max) — one busy column must not
    # squeeze siblings elsewhere. Coordinates are used as-is by vis-network;
    # do not rely on the front-end to compress them.
    min_y_step = 95.0

    for level in sorted(levels):
        members = sorted(levels[level])
        n = len(members)
        level_y_step = max(min_y_step, min(y_gap, 1300.0 / max(n, 1)))
        # Small fan-outs: vertical stack (readable lineage). Large: loose grid.
        cols = 1 if n <= 10 else max(1, math.ceil(math.sqrt(n * 1.35)))
        rows = math.ceil(n / cols)
        x_step = x_gap * 0.5 if cols > 1 else 0.0
        for i, node in enumerate(members):
            row = i // cols
            col = i % cols
            x = level * x_gap + col * x_step
            y = (row - (rows - 1) / 2.0) * level_y_step
            pos[node] = (x, y)
    return pos


def _layered_layout(
    g: nx.DiGraph,
    *,
    x_gap: float = 240.0,
    y_gap: float = 100.0,
    max_cols: int = 3,
    radius_step: float = 150.0,
) -> dict[str, tuple[float, float]]:
    """Lay out disconnected trees in a grid so clusters use width and height."""
    components = [g.subgraph(c).copy() for c in nx.weakly_connected_components(g)]
    components.sort(key=lambda sg: min(sg.nodes) if sg.nodes else "")
    comp_sets = [set(sub.nodes) for sub in components]

    boxes: list[tuple[dict[str, tuple[float, float]], float, float]] = []
    # Vertical 2-node chains have width≈0; reserve a slot so neighbors are not flush.
    min_cell = radius_step * 0.34
    col_pad = max(radius_step * 0.52, x_gap * 0.35)
    row_pad = max(radius_step * 0.42, y_gap * 1.05)

    for sub in components:
        if not sub.nodes:
            continue
        sub_pos = _radial_layout_component(sub, radius_step=radius_step)
        sub_pos = _resolve_collisions(sub_pos, min_dist=max(MIN_NODE_SEP, radius_step * 0.38))
        min_x = min(x for x, _ in sub_pos.values())
        min_y = min(y for _, y in sub_pos.values())
        normalized = {n: (x - min_x, y - min_y) for n, (x, y) in sub_pos.items()}
        width = max(x for x, _ in normalized.values()) if normalized else 0.0
        height = max(y for _, y in normalized.values()) if normalized else 0.0
        boxes.append((normalized, width, height))

    if not boxes:
        return {}

    pos: dict[str, tuple[float, float]] = {}
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0
    col = 0
    for normalized, width, height in boxes:
        if col >= max_cols:
            col = 0
            x_cursor = 0.0
            y_cursor += row_height + row_pad
            row_height = 0.0
        for node, (x, y) in normalized.items():
            pos[node] = (x + x_cursor, y + y_cursor)
        x_cursor += max(width, min_cell) + col_pad
        row_height = max(row_height, max(height, min_cell))
        col += 1

    out = dict(pos)
    min_dist = max(MIN_NODE_SEP, radius_step * 0.36)
    for members in comp_sets:
        sub_pos = {n: out[n] for n in members}
        sub_pos = _resolve_collisions(sub_pos, min_dist=min_dist, iterations=64)
        out.update(sub_pos)
    return out


def graph_payload(
    nodes: dict[str, Node],
    base: str,
    view: GraphView = "all",
    impute: bool = False,
    rollup_res: Optional[dict] = None,
) -> dict[str, Any]:
    """Export nodes/edges/positions for the lineage graph."""
    g = build_graph(nodes)
    if view == "family":
        members = family_members(g, base)
        sub = g.subgraph(members).copy()
        res = rollup_res if rollup_res is not None else rollup(nodes, base, impute=impute)
        rows = {r["model"]: r for r in res["rows"]}
        coverage = res["coverage"]
        scope_label = "family"
    else:
        sub = g.copy()
        rows = None
        in_store = sum(1 for mid in sub.nodes() if mid in nodes and nodes[mid].has_report)
        coverage = (in_store / len(sub)) if sub.nodes else 0.0
        scope_label = "dataset"

    if not sub.nodes:
        return {
            "nodes": [],
            "edges": [],
            "base": base,
            "view": view,
            "scope_label": scope_label,
            "coverage": 0.0,
        }

    n_nodes = len(sub.nodes)
    radius_step = 165.0 + min(n_nodes, 80) * 1.8
    n_comp = nx.number_weakly_connected_components(sub)
    max_cols = min(6, max(3, math.ceil(math.sqrt(n_comp * 2.0))))
    pos = _layered_layout(
        sub,
        x_gap=200.0 + min(n_nodes, 120) * 1.2,
        y_gap=120.0 + min(n_nodes, 120) * 0.8,
        max_cols=max_cols,
        radius_step=radius_step,
    )

    out_nodes = []
    for mid in sub.nodes():
        row = rows.get(mid, {}) if rows is not None else _row_from_store(mid, nodes.get(mid), base)
        x, y = pos[mid]
        highlighted = mid == base and (view == "family" or mid in nodes)
        out_nodes.append(
            {
                "id": mid,
                "label": "From scratch" if mid == "scratch" else mid.rsplit("/", 1)[-1],
                "x": float(x),
                "y": float(y),
                "role": row.get("role", "derivative"),
                "carbon": row.get("carbon", "—"),
                "water": row.get("water", "—"),
                "energy": row.get("energy", "—"),
                "quality": row.get("quality", "no report"),
                "method": row.get("method", ""),
                "highlighted": highlighted,
            }
        )

    out_edges = []
    for src, dst, data in sub.edges(data=True):
        out_edges.append(
            {
                "source": src,
                "target": dst,
                "relation": data.get("relation") or "finetune",
            }
        )

    return {
        "base": base,
        "view": view,
        "scope_label": scope_label,
        "n_models": len(out_nodes),
        "n_edges": len(out_edges),
        "coverage": coverage,
        "nodes": out_nodes,
        "edges": out_edges,
    }


def family_graph_payload(
    nodes: dict[str, Node],
    base: str,
    impute: bool = False,
    rollup_res: Optional[dict] = None,
) -> dict[str, Any]:
    """Export the family subgraph (backward-compatible wrapper)."""
    return graph_payload(nodes, base, view="family", impute=impute, rollup_res=rollup_res)


def impact_graph_figure(
    nodes: dict[str, Node],
    base: str,
    view: GraphView = "all",
    impute: bool = False,
    rollup_res: Optional[dict] = None,
) -> Optional[go.Figure]:
    """Return a Plotly force-layout graph coloured by disclosure quality."""
    payload = graph_payload(nodes, base, view=view, impute=impute, rollup_res=rollup_res)
    if not payload["nodes"]:
        return None

    node_by_id = {n["id"]: n for n in payload["nodes"]}
    edge_x: list[Optional[float]] = []
    edge_y: list[Optional[float]] = []
    for edge in payload["edges"]:
        src = node_by_id[edge["source"]]
        dst = node_by_id[edge["target"]]
        edge_x.extend([src["x"], dst["x"], None])
        edge_y.extend([src["y"], dst["y"], None])

    node_x = [n["x"] for n in payload["nodes"]]
    node_y = [n["y"] for n in payload["nodes"]]
    labels = [n["label"] for n in payload["nodes"]]
    colors = [_quality_color(n["quality"]) for n in payload["nodes"]]
    sizes = [_node_size(_carbon_hi_from_label(n["carbon"]), n["highlighted"]) for n in payload["nodes"]]

    hovers = []
    for n in payload["nodes"]:
        hovers.append(
            f"<b>{n['id']}</b><br>"
            f"Role: {n['role']}<br>"
            f"Carbon: {n['carbon']}<br>"
            f"Water: {n['water']} · Energy: {n['energy']}<br>"
            f"Quality: {n['quality']}" + (f"<br>Method: {n['method']}" if n["method"] else "")
        )

    line_widths = [3 if n["highlighted"] else 1 for n in payload["nodes"]]
    line_colors = [
        (BASE_RING if payload["view"] == "family" else HIGHLIGHT) if n["highlighted"] else "rgba(0,0,0,0.25)"
        for n in payload["nodes"]
    ]

    show_labels = len(payload["nodes"]) <= LABEL_LIMIT
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 1.5, "color": EDGE_COLOR},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if show_labels else "markers",
            text=labels if show_labels else None,
            textposition="top center",
            textfont={"size": 10, "color": CHART_INK},
            marker={
                "size": sizes,
                "color": colors,
                "line": {"width": line_widths, "color": line_colors},
            },
            hovertext=hovers,
            hoverinfo="text",
            showlegend=False,
        )
    )

    cov = payload["coverage"] * 100
    if payload["view"] == "family":
        title = f"{base.rsplit('/', 1)[-1]} family · {payload['n_models']} models · {cov:.0f}% disclosed"
    else:
        title = f"Full dataset · {payload['n_models']} models · {payload['n_edges']} edges · {cov:.0f}% with report"
        if base in node_by_id:
            title += f" · highlight: {base.rsplit('/', 1)[-1]}"

    fig.update_layout(
        title={"text": title, "font": {"size": 13, "color": CHART_INK}},
        margin={"l": 20, "r": 20, "t": 44, "b": 20},
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        dragmode="pan",
        font={"color": CHART_INK},
    )
    return fig


def payload_to_json(payload: dict[str, Any]) -> str:
    """Serialize graph payload (for tests or future Sigma/vis-network front-end)."""
    return json.dumps(payload, indent=2)
