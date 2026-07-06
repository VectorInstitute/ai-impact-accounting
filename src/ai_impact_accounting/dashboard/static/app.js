/* DIA web dashboard — vis-network graph + Chart.js */

let fullGraph = { nodes: [], edges: [] };
let network = null;
let selectedNodeId = null;
let graphFocused = false;
let hoverActive = false;
let hoveredNodeId = null;
let barChart = null;
let defaultBase = "";
let lastRollup = null;
let tableRowsByModel = {};
let lastTableRows = [];
let lastTableRollup = null;
let tableSortKey = "carbon";
let tableSortDir = "desc";
let tableSearchQuery = "";
let appliedControlsSnapshot = "";
let loadDebounceTimer = null;
let graphColorBy = "quality";
let graphLegendData = null;

const ROLE_NODE_COLORS = {
  base: "#1e6a8a",
  derivative: "#3d9a40",
  "lineage parent": "#d4cfc4",
  placeholder: "#94a3b8",
  "in dataset": "#cbd5e1",
  default: "#cbd5e1",
};

const ROLE_LEGEND_NODES = [
  { label: "Base model", color: "#1e6a8a" },
  { label: "Derivative", color: "#3d9a40" },
  { label: "Lineage parent only", color: "#d4cfc4" },
  { label: "From scratch (placeholder)", color: "#94a3b8" },
  { label: "In dataset (no report)", color: "#cbd5e1" },
];

const GRAPH_ZOOM_MIN_RATIO = 0.55;
const GRAPH_ZOOM_MAX_RATIO = 3;
const GRAPH_ZOOM_ABSOLUTE_MIN = 0.12;
const GRAPH_ZOOM_ABSOLUTE_MAX = 2.5;
let graphFitScale = 1;
let graphZoomAnchor = null;

const $ = (id) => document.getElementById(id);

function isEmbedMode() {
  return new URLSearchParams(window.location.search).get("embed") === "1";
}

function readUrlState() {
  const p = new URLSearchParams(window.location.search);
  return {
    base: p.get("base") || "",
    compare: p.get("compare") || "",
    impute: p.get("impute") === "1",
    rowFilter: p.get("row_filter") || "all",
    graphView: p.get("graph_view") || "all",
  };
}

function applyEmbedMode() {
  if (isEmbedMode()) document.body.classList.add("embed-mode");
  syncTopbarHeight();
}

function syncTopbarHeight() {
  const topbar = document.querySelector(".topbar");
  const height = topbar && !isEmbedMode() && topbar.offsetParent !== null
    ? topbar.getBoundingClientRect().height
    : 0;
  document.documentElement.style.setProperty("--topbar-height", `${height}px`);
}

function syncUrlState() {
  const p = queryParams();
  if (isEmbedMode()) p.set("embed", "1");
  const qs = p.toString();
  const next = qs ? `?${qs}` : window.location.pathname;
  if (`${window.location.pathname}${window.location.search}` !== next) {
    history.replaceState(null, "", next);
  }
}

function applyControlsFromUrl(state) {
  if (!state) return;
  if (state.rowFilter) $("row-filter").value = state.rowFilter;
  if (state.graphView) $("graph-view").value = state.graphView;
}

// Method-based imputation is wired through the API (`?impute=1`) for future exploration;
// the dashboard UI does not expose it yet — totals default to reported footprints only.
function isImputeEnabled() {
  return new URLSearchParams(window.location.search).get("impute") === "1";
}

function controlsSignature() {
  return JSON.stringify({
    base: $("base-select").value,
    compare: $("compare-select").value,
    impute: isImputeEnabled(),
    rowFilter: $("row-filter").value,
    graphView: $("graph-view").value,
  });
}

function updateStaleControlsHint() {
  const stale = appliedControlsSnapshot && controlsSignature() !== appliedControlsSnapshot;
  $("controls-stale-hint").classList.toggle("hidden", !stale);
  $("apply-btn").classList.toggle("needs-apply", stale);
}

function markControlsApplied() {
  appliedControlsSnapshot = controlsSignature();
  updateStaleControlsHint();
}

function setDashboardLoading(loading) {
  document.body.classList.toggle("is-loading", loading);
  $("loading-indicator").classList.toggle("hidden", !loading);
  $("main").setAttribute("aria-busy", loading ? "true" : "false");
  $("apply-btn").disabled = loading;
}

function scheduleLoadDashboard() {
  clearTimeout(loadDebounceTimer);
  updateStaleControlsHint();
  loadDebounceTimer = setTimeout(() => loadDashboard(), 400);
}

function parseNumericField(value) {
  if (value == null || value === "—") return -1;
  const m = String(value).replace(/,/g, "").match(/-?\d+(\.\d+)?/);
  return m ? parseFloat(m[0]) : -1;
}

function compareTableRows(a, b, key) {
  if (key === "carbon") return (a.carbon_hi ?? 0) - (b.carbon_hi ?? 0);
  if (key === "water" || key === "energy") {
    return parseNumericField(a[key]) - parseNumericField(b[key]);
  }
  const av = String(a[key] ?? "").toLowerCase();
  const bv = String(b[key] ?? "").toLowerCase();
  return av.localeCompare(bv);
}

function updateTableSortHeaders() {
  $("footprint-table").querySelectorAll("th.sortable").forEach((th) => {
    const key = th.getAttribute("data-sort");
    th.classList.remove("sort-asc", "sort-desc");
    if (key === tableSortKey) {
      th.classList.add(tableSortDir === "asc" ? "sort-asc" : "sort-desc");
      th.setAttribute("aria-sort", tableSortDir === "asc" ? "ascending" : "descending");
    } else {
      th.setAttribute("aria-sort", "none");
    }
  });
}

function renderTableBody(rows) {
  const tbody = $("footprint-table").querySelector("tbody");
  tableRowsByModel = {};
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty">No models match this filter.</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map((r) => {
      tableRowsByModel[r.model] = r;
      const qk = r.quality_key || "none";
      return `<tr class="q-${qk}" data-model="${r.model}">` +
        `<td>${r.model_url
          ? `<a href="${r.model_url}" target="_blank" rel="noopener" onclick="event.stopPropagation()">${r.model}</a>`
          : r.model}</td>` +
        `<td>${r.role}</td><td>${r.method || "—"}</td>` +
        `<td>${r.carbon}</td><td>${r.water}</td><td>${r.energy}</td>` +
        `<td>${r.quality}</td><td>${r.gpu}</td><td>${r.region}</td></tr>`;
    })
    .join("");
  tbody.querySelectorAll("tr[data-model]").forEach((tr) => {
    tr.addEventListener("click", () => focusGraphNode(tr.getAttribute("data-model")));
  });
}

function applyTableView() {
  let rows = [...lastTableRows];
  const q = tableSearchQuery.trim().toLowerCase();
  if (q) {
    rows = rows.filter((r) =>
      [r.model, r.role, r.method, r.quality, r.gpu, r.region].join(" ").toLowerCase().includes(q)
    );
  }
  rows.sort((a, b) => {
    const cmp = compareTableRows(a, b, tableSortKey);
    return tableSortDir === "asc" ? cmp : -cmp;
  });
  updateTableSortHeaders();
  renderTableBody(rows);
  const rollup = lastTableRollup || { n_models: 0, n_with_report: 0 };
  const disclosed = rollup.n_models ? ((rollup.n_with_report / rollup.n_models) * 100).toFixed(1) : "0";
  const filterNote = q ? ` (search: "${tableSearchQuery.trim()}")` : "";
  $("table-note").textContent =
    `Showing ${rows.length} of ${lastTableRows.length} models${filterNote} — ${disclosed}% disclosed. Click a row to focus that node in the graph.`;
}

function updateKpiFootnote(rollup) {
  const el = $("kpi-footnote");
  if (!rollup?.base_card_disclosure) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }
  const d = rollup.base_card_disclosure;
  el.classList.remove("hidden");
  el.innerHTML =
    `Family KPI totals sum <code>dia_report</code> footprints only. Publisher pretraining on the base model card` +
    ` (${d.carbon}${d.gpu_hours ? ` · ${d.gpu_hours} GPU h` : ""}) appears in the summary and node pane` +
    ` but is <strong>not</strong> included in rollup totals unless ingested as a report.`;
}

function showError(msg) {
  const el = $("error-banner");
  el.textContent = msg;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 6000);
}

function queryParams() {
  const base = $("base-select").value;
  const compare = $("compare-select").value;
  const impute = isImputeEnabled() ? "1" : "0";
  const rowFilter = $("row-filter").value;
  const graphView = $("graph-view").value;
  const p = new URLSearchParams({ base, impute, row_filter: rowFilter, graph_view: graphView });
  if (compare) p.set("compare", compare);
  return p;
}

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok && data.error) throw new Error(data.error);
  return data;
}

function fillSelect(el, choices, value) {
  el.innerHTML = "";
  for (const c of choices) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    el.appendChild(opt);
  }
  if (value && choices.includes(value)) el.value = value;
  else if (choices.length) el.value = choices[0];
}

async function loadMeta(urlState) {
  const meta = await fetchJson("/api/meta");
  defaultBase = meta.default_base || "";
  $("dataset-meta").innerHTML =
    `<strong>Dataset:</strong> <a href="${meta.dataset_url}" target="_blank" rel="noopener">${meta.dataset}</a>` +
    ` · <strong>${meta.n_nodes}</strong> model(s) · <strong>${meta.n_with_report}</strong> with footprint data`;

  const bases = await fetchJson("/api/bases");
  const preferredBase = urlState?.base || bases.default_base;
  fillSelect($("base-select"), bases.bases, preferredBase);
  const cmp = $("compare-select");
  cmp.innerHTML = '<option value="">— none —</option>';
  for (const b of bases.bases) {
    const opt = document.createElement("option");
    opt.value = b;
    opt.textContent = b;
    cmp.appendChild(opt);
  }
  if (urlState?.compare && bases.bases.includes(urlState.compare)) {
    cmp.value = urlState.compare;
  }
}

function qualityChipClass(quality) {
  const q = (quality || "").toLowerCase();
  if (q === "measured") return "measured";
  if (q === "imputed") return "imputed";
  if (q === "disclosed-on-card") return "disclosed-on-card";
  if (q.startsWith("estimated")) return "estimated";
  return "";
}

function renderKpi(cards) {
  $("kpi-row").innerHTML = cards
    .map(
      (c) =>
        `<div class="kpi${c.primary ? " kpi-primary" : ""}">` +
        `<div class="kpi-label">${c.label}</div>` +
        `<div class="kpi-value">${c.value}</div>` +
        `<div class="kpi-sub">${c.sub || ""}</div></div>`
    )
    .join("");
}

function renderConfidenceBanner(rollup) {
  const el = $("confidence-banner");
  if (!rollup || rollup.n_models <= 1) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }
  const cov = rollup.coverage * 100;
  const missing = rollup.n_without_report;
  if (missing === 0 && cov >= 99) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }
  el.classList.remove("hidden");
  if (missing > 0 && cov < 60) {
    el.innerHTML =
      `<strong>Lower-bound estimate.</strong> Only ${rollup.n_with_report} of ${rollup.n_models} models ` +
      `in this family disclosed footprint data (${cov.toFixed(0)}% coverage). ` +
      `Totals omit <strong>${missing}</strong> model(s) — ` +
      `<a href="#" id="banner-hub-link">check the base on Hub</a> to ingest reports.`;
    const link = $("banner-hub-link");
    if (link) {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        checkBaseOnHub();
      });
    }
    return;
  }
  if (missing > 0) {
    el.innerHTML =
      `<strong>Partial coverage (${cov.toFixed(0)}%).</strong> ` +
      `${missing} model(s) have no footprint data yet — family totals may increase as more models report.`;
    return;
  }
  el.classList.add("hidden");
  el.innerHTML = "";
}

function renderFamilyHeading(_rollup) {
  /* family name is already in the base-model selector */
}

function renderSummary(rollup) {
  const cov = (rollup.coverage * 100).toFixed(0);
  let html = "";
  if (rollup.n_with_report >= 2) {
    const t = rollup.total_footprint;
    const ratio = rollup.deriv_over_base_ratio;
    const ratioTxt = ratio ? `${ratio[0].toFixed(1)}–${ratio[1].toFixed(1)}× base` : "n/a";
    html +=
      `<p><strong>Models with data:</strong> ${rollup.n_with_report} / ${rollup.n_models} (${cov}%)</p>` +
      `<table><tbody>` +
      `<tr><td>Carbon</td><td>${t.carbon.fmt} kgCO₂eq</td></tr>` +
      `<tr><td>Water</td><td>${t.water.fmt} L</td></tr>` +
      `<tr><td>Energy</td><td>${t.energy.fmt} kWh</td></tr>` +
      `</tbody></table>` +
      `<p><strong>Derivative footprint vs base:</strong> ${ratioTxt}</p>`;
  } else {
    html += `<p><strong>Models in family:</strong> ${rollup.n_models} ` +
      `(${rollup.n_with_report} with footprint data)</p>` +
      `<p><strong>Coverage:</strong> ${cov}%</p>`;
  }
  if (rollup.base_card_disclosure) {
    const d = rollup.base_card_disclosure;
    html +=
      `<p class="publisher-note"><strong>Publisher pretraining estimate:</strong> ${d.carbon}` +
      (d.gpu_hours ? ` · ${d.gpu_hours} GPU h` : "") +
      (d.hardware ? ` · ${d.hardware}` : "") +
      ` <span class="hub-status-warn">(model card only — not in family rollup)</span></p>`;
  }
  $("summary").innerHTML = `<h2 class="section-label">Family summary</h2>` + html;
}

function renderTable(rows, shown, total, rollup) {
  lastTableRows = rows;
  lastTableRollup = rollup;
  applyTableView();
}

function focusGraphNode(modelId) {
  if (!modelId || !network) return;
  const node = fullGraph.nodes.find((n) => n.id === modelId);
  if (!node) return;
  selectedNodeId = modelId;
  graphFocused = true;
  network.setData(buildNetworkData(fullGraph, modelId));
  scheduleFit();
  showNodePane(modelId);
  updateGraphToolbarState();
  document.querySelector(".graph-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderHardware(items) {
  if (!items.length) {
    $("hardware").innerHTML = "<p><em>No disclosed compute metadata for this family.</em></p>";
    return;
  }
  $("hardware").innerHTML =
    "<h3>Hardware &amp; region</h3><ul>" +
    items.map((i) => `<li><strong>${i.short}</strong> — <code>${i.gpu}</code>, region <code>${i.region}</code></li>`).join("") +
    "</ul>";
}

function renderCompare(cmp) {
  const sec = $("compare-section");
  if (!cmp) {
    sec.classList.add("hidden");
    return;
  }
  sec.classList.remove("hidden");
  sec.innerHTML =
    `<h3>Compare families</h3>` +
    `<table><thead><tr><th></th><th>${cmp.primary_base}</th><th>${cmp.secondary_base}</th></tr></thead><tbody>` +
    cmp.rows.map((r) => `<tr><td>${r.label}</td><td>${r.primary}</td><td>${r.secondary}</td></tr>`).join("") +
    `</tbody></table>`;
}

function renderBarChart(spec) {
  const row = $("charts-row");
  const canvas = $("bar-chart");
  if (barChart) {
    barChart.destroy();
    barChart = null;
  }
  if (!spec) {
    row.classList.add("hidden");
    return;
  }
  row.classList.remove("hidden");
  barChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: spec.labels,
      datasets: [{
        label: "kgCO₂eq",
        data: spec.values,
        backgroundColor: spec.colors,
        borderColor: "#333",
        borderWidth: 0.6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2.2,
      plugins: { title: { display: false }, legend: { display: false } },
      scales: { y: { title: { display: true, text: "kgCO₂eq" } } },
    },
  });
}

const LABEL_LIMIT = 18;

function renderLegend(legend) {
  graphLegendData = legend;
  renderLegendForMode();
}

function renderLegendForMode() {
  const legend = graphLegendData;
  if (!legend) return;
  const heading = $("legend-node-heading");
  if (graphColorBy === "role") {
    heading.textContent = "Node · lineage role";
    $("legend-nodes").innerHTML = ROLE_LEGEND_NODES.map(
      (item) =>
        `<li><span class="legend-swatch" style="background:${item.color}"></span>${item.label}</li>`
    ).join("");
  } else {
    heading.textContent = "Node · data quality";
    $("legend-nodes").innerHTML = (legend.nodes || [])
      .map(
        (item) =>
          `<li><span class="legend-swatch" style="background:${item.color}"></span>${item.label}</li>`
      )
      .join("");
  }
  $("legend-edges").innerHTML = (legend.edges || [])
    .map(
      (item) =>
        `<li><span class="legend-swatch edge" style="background:${item.color}"></span>${item.label}</li>`
    )
    .join("");
}

function nodeColorForMode(n) {
  if (graphColorBy !== "role") {
    if (n.color) {
      return typeof n.color === "string"
        ? { background: n.color, border: n.color, highlight: { background: n.color, border: "#1e6a8a" } }
        : { ...n.color };
    }
    return { background: "#cbd5e1", border: "#94a3b8", highlight: { background: "#cbd5e1", border: "#1e6a8a" } };
  }
  const role = (n.role || "").toLowerCase();
  const bg = ROLE_NODE_COLORS[role] || ROLE_NODE_COLORS.default;
  return {
    background: bg,
    border: n.is_placeholder ? "#64748b" : "#555",
    highlight: { background: bg, border: "#1e6a8a" },
  };
}

function refreshGraphColors() {
  if (!network) return;
  const focus = graphFocused ? selectedNodeId : null;
  network.setData(buildNetworkData(fullGraph, focus));
  scheduleFit();
  renderLegendForMode();
}

function applyLabelVisibility(nodes, show) {
  return nodes.map((n) => ({
    ...n,
    label: show ? n.label || n.id.split("/").pop() : undefined,
  }));
}

function graphTargetSpan(nodeCount) {
  if (nodeCount <= 20) return 500;
  if (nodeCount <= 50) return 900;
  return Math.min(3200, 600 + nodeCount * 18);
}

function graphNodeSize(nodeCount, baseSize = 22) {
  if (nodeCount > 70) return 10;
  if (nodeCount > 35) return 14;
  if (nodeCount > 20) return 18;
  return baseSize;
}

function syncGraphViewport(nodeCount) {
  const height = nodeCount > 70 ? 520 : nodeCount > 35 ? 440 : 360;
  document.documentElement.style.setProperty("--graph-height", `${height}px`);
}

function normalizeGraphPositions(nodes, targetSpan) {
  if (!nodes.length) return nodes;
  const xs = nodes.map((n) => n.x ?? 0);
  const ys = nodes.map((n) => n.y ?? 0);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  const span = targetSpan ?? graphTargetSpan(nodes.length);
  const scale = span / Math.max(spanX, spanY);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return nodes.map((n) => ({
    ...n,
    x: (n.x - cx) * scale,
    y: (n.y - cy) * scale,
  }));
}

function withEdgeIds(edges) {
  return edges.map((e, i) => ({ ...e, id: e.id || `${e.from}|${e.to}|${i}` }));
}

function egoNetwork(nodeId) {
  const active = new Set([nodeId]);
  for (const e of fullGraph.edges) {
    if (e.from === nodeId) active.add(e.to);
    if (e.to === nodeId) active.add(e.from);
  }
  return active;
}

function storedNodeColor(nodeId) {
  const src = fullGraph.nodes.find((n) => n.id === nodeId);
  if (!src) return { background: "#cbd5e1", border: "#94a3b8" };
  return nodeColorForMode(src);
}

const DIM_NODE_COLOR = { background: "#b8bcc4", border: "#8b93a1" };
const DIM_EDGE_COLOR = { color: "#aeb4bc" };

function hideHoverLabel() {
  hoveredNodeId = null;
  $("graph-hover-label").classList.add("hidden");
}

function positionHoverLabel(nodeId) {
  const el = $("graph-hover-label");
  const container = $("graph");
  if (!network || !nodeId || graphFocused) {
    hideHoverLabel();
    return;
  }

  const positions = network.getPositions([nodeId]);
  const pos = positions[nodeId];
  if (!pos) {
    hideHoverLabel();
    return;
  }

  hoveredNodeId = nodeId;
  const dom = network.canvasToDOM({ x: pos.x, y: pos.y });
  const node = network.body.data.nodes.get(nodeId);
  const scale = network.getScale();
  const nodeRadius = ((node?.size || 22) / 2) * scale;
  const w = container.clientWidth;
  const h = container.clientHeight;
  const pad = 10;
  const labelH = 30;

  el.textContent = nodeDisplayName(nodeId);
  el.classList.remove("hidden", "above", "below");

  const left = Math.max(pad, Math.min(dom.x, w - pad));
  el.style.left = `${left}px`;

  const belowTop = dom.y + nodeRadius + 8;
  const aboveTop = dom.y - nodeRadius - 8;
  if (belowTop + labelH > h - pad) {
    el.classList.add("above");
    el.style.top = `${Math.max(pad, aboveTop)}px`;
  } else {
    el.classList.add("below");
    el.style.top = `${Math.min(h - pad - labelH, belowTop)}px`;
  }
}

function applyHoverFocus(nodeId) {
  if (!network || graphFocused) return;
  hoverActive = true;
  const active = egoNetwork(nodeId);

  const nodeUpdates = network.body.data.nodes.get().map((n) => {
    const isCenter = n.id === nodeId;
    const inEgo = active.has(n.id);
    if (inEgo) {
      const baseColor = storedNodeColor(n.id);
      return {
        id: n.id,
        opacity: 1,
        color: {
          ...baseColor,
          highlight: baseColor.highlight || { background: baseColor.background, border: baseColor.border },
        },
        label: isCenter ? undefined : n.label || undefined,
        font: isCenter
          ? undefined
          : {
              size: 12,
              color: "#1a1a1a",
              strokeWidth: 4,
              strokeColor: "#ffffff",
              face: "Segoe UI, system-ui, sans-serif",
            },
        size: isCenter ? (n.size || 22) + 6 : n.size,
        borderWidth: isCenter ? 3 : 2,
      };
    }
    return {
      id: n.id,
      opacity: 0.8,
      color: {
        background: DIM_NODE_COLOR.background,
        border: DIM_NODE_COLOR.border,
        highlight: DIM_NODE_COLOR,
      },
      label: undefined,
      borderWidth: 2,
    };
  });

  const edgeUpdates = network.body.data.edges.get().map((e) => {
    const lit = e.from === nodeId || e.to === nodeId;
    const src = fullGraph.edges.find((x) => x.from === e.from && x.to === e.to);
    const litColor = src?.color || e.color;
    return {
      id: e.id,
      opacity: lit ? 1 : 0.35,
      color: lit ? litColor : DIM_EDGE_COLOR,
      width: lit ? 2.5 : 1.2,
    };
  });

  network.body.data.nodes.update(nodeUpdates);
  network.body.data.edges.update(edgeUpdates);
  positionHoverLabel(nodeId);
}

function clearHoverFocus() {
  if (!network || !hoverActive || graphFocused) return;
  hoverActive = false;
  hideHoverLabel();
  network.setData(buildNetworkData(fullGraph, null));
}

function buildNetworkData(graph, filterNodeId) {
  const decorate = (n, overrides = {}) => ({
    ...n,
    color: nodeColorForMode(n),
    ...overrides,
    font: {
      size: 16,
      color: "#1a1a1a",
      strokeWidth: 5,
      strokeColor: "#ffffff",
      face: "Segoe UI, system-ui, sans-serif",
      ...(n.font || {}),
    },
  });

  if (!filterNodeId) {
    const n = graph.nodes.length;
    const showLabels = n <= LABEL_LIMIT;
    syncGraphViewport(n);
    const dotSize = graphNodeSize(n);
    const positioned = normalizeGraphPositions(graph.nodes);
    return {
      nodes: new vis.DataSet(
        applyLabelVisibility(
          positioned.map((n) =>
            decorate(n, {
              fixed: { x: true, y: true },
              size: n.size ? Math.min(n.size, dotSize + 4) : dotSize,
            })
          ),
          showLabels
        )
      ),
      edges: new vis.DataSet(withEdgeIds(graph.edges)),
    };
  }

  const center = graph.nodes.find((n) => n.id === filterNodeId);
  if (!center) {
    return buildNetworkData(graph, null);
  }

  const parents = [];
  const children = [];
  for (const e of graph.edges) {
    if (e.to === filterNodeId) parents.push(e.from);
    if (e.from === filterNodeId) children.push(e.to);
  }
  const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));

  const nodes = applyLabelVisibility(
    [
      decorate(
        { ...center, x: 0, y: 0, fixed: { x: true, y: true } },
        { size: (center.size || 28) + 8, borderWidth: 4 }
      ),
      ...parents.flatMap((id, i) => {
        const src = byId[id];
        if (!src) return [];
        return [
          decorate(
            { ...src, x: -320, y: (i - (parents.length - 1) / 2) * 120, fixed: { x: true, y: true } },
            { borderWidth: 2 }
          ),
        ];
      }),
      ...children.flatMap((id, i) => {
        const src = byId[id];
        if (!src) return [];
        return [
          decorate(
            { ...src, x: 320, y: (i - (children.length - 1) / 2) * 120, fixed: { x: true, y: true } },
            { borderWidth: 2 }
          ),
        ];
      }),
    ],
    true
  );

  const edges = graph.edges.filter(
    (e) => e.from === filterNodeId || e.to === filterNodeId
  );
  return { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(withEdgeIds(edges)) };
}

function graphOptions() {
  return {
    physics: { enabled: false },
    interaction: {
      hover: true,
      hoverConnectedEdges: false,
      multiselect: false,
      tooltipDelay: 120,
      zoomView: true,
      dragView: true,
    },
    edges: {
      smooth: { type: "cubicBezier", forceDirection: "horizontal", roundness: 0.35 },
      width: 2,
      color: { inherit: false },
    },
    nodes: {
      shape: "dot",
      margin: 12,
      borderWidth: 2,
      size: 22,
      font: {
        size: 14,
        color: "#1a1a1a",
        strokeWidth: 4,
        strokeColor: "#ffffff",
        face: "Segoe UI, system-ui, sans-serif",
      },
    },
  };
}

function graphZoomLimits() {
  const fit = graphFitScale || 1;
  return {
    min: Math.max(GRAPH_ZOOM_ABSOLUTE_MIN, fit * GRAPH_ZOOM_MIN_RATIO),
    max: Math.min(GRAPH_ZOOM_ABSOLUTE_MAX, fit * GRAPH_ZOOM_MAX_RATIO),
  };
}

function clampGraphScale(scale) {
  const { min, max } = graphZoomLimits();
  return Math.min(max, Math.max(min, scale));
}

function rememberGraphZoomAnchor() {
  if (!network) return;
  graphZoomAnchor = network.getViewPosition();
}

function attachGraphZoomLimits(net) {
  rememberGraphZoomAnchor();
  net.on("zoom", () => {
    const scale = net.getScale();
    const clamped = clampGraphScale(scale);
    if (scale !== clamped) {
      net.moveTo({
        position: graphZoomAnchor || net.getViewPosition(),
        scale: clamped,
        animation: false,
      });
    } else {
      rememberGraphZoomAnchor();
    }
  });
  net.on("dragEnd", rememberGraphZoomAnchor);
}

function fitGraph(padding = 28) {
  if (!network) return;
  network.fit({
    animation: false,
    padding,
  });
  graphFitScale = network.getScale();
  rememberGraphZoomAnchor();
}

function scheduleFit() {
  requestAnimationFrame(() => {
    fitGraph(28);
    setTimeout(() => fitGraph(28), 120);
  });
}

function updateGraphToolbarState(graphView) {
  const scope = graphView ?? $("graph-view").value;
  $("clear-focus-btn").disabled = !graphFocused;
  $("focus-family-btn").disabled = !graphFocused;
  const fullBtn = $("full-dataset-btn");
  if (scope === "family") {
    fullBtn.classList.remove("hidden");
    fullBtn.disabled = false;
  } else {
    fullBtn.classList.add("hidden");
    fullBtn.disabled = true;
  }
}

function initNetwork(graph) {
  fullGraph = graph;
  graphFocused = false;
  selectedNodeId = null;
  renderLegend(graph.legend);

  const container = $("graph");
  const data = buildNetworkData(graph, null);
  const options = graphOptions();

  if (network) network.destroy();
  network = new vis.Network(container, data, options);
  attachGraphZoomLimits(network);

  network.on("hoverNode", (params) => {
    applyHoverFocus(params.node);
  });

  network.on("blurNode", () => {
    clearHoverFocus();
  });

  network.on("afterDrawing", () => {
    if (hoverActive && hoveredNodeId && !graphFocused) {
      positionHoverLabel(hoveredNodeId);
    }
  });

  network.on("click", (params) => {
    hideHoverLabel();
    if (params.event) params.event.preventDefault();
    clearHoverFocus();
    if (!params.nodes.length) return;
    const nodeId = params.nodes[0];
    selectedNodeId = nodeId;
    graphFocused = true;
    network.setData(buildNetworkData(fullGraph, nodeId));
    scheduleFit();
    showNodePane(nodeId);
    updateGraphToolbarState();
  });

  scheduleFit();

  const cov = (graph.coverage * 100).toFixed(0);
  const baseLabel = graph.base ? graph.base.split("/").pop() : "family";
  const title =
    graph.view === "family"
      ? `Family: ${baseLabel} · ${graph.n_models} models · ${graph.n_edges} edges · ${cov}% disclosed`
      : `Full dataset · ${graph.n_models} models · ${graph.n_edges} edges · ${cov}% with report`;
  $("graph-title").textContent = title;
  updateGraphToolbarState(graph.view);
}

function nodeDisplayName(nodeId) {
  const node = fullGraph.nodes.find((n) => n.id === nodeId);
  return node?.display_label || node?.label || nodeId;
}

function renderNodeDetail(node) {
  const qClass = qualityChipClass(node.quality || node.quality_key);
  const qualityLabel = node.quality || node.quality_key || "no report";
  let html = "";
  if (node.is_placeholder) {
    html += `<p><em>Placeholder — groups models trained from scratch (not a Hugging Face repo).</em></p>`;
  } else if (node.hub_url) {
    html += `<p><a href="${node.hub_url}" target="_blank" rel="noopener">Open on Hugging Face ↗</a></p>`;
  }
  html += `<table class="node-detail-table"><tbody>` +
    `<tr><th>Role</th><td>${node.role || "—"}</td></tr>` +
    `<tr><th>Carbon</th><td>${node.carbon || "—"}</td></tr>` +
    `<tr><th>Water</th><td>${node.water || "—"}</td></tr>` +
    `<tr><th>Quality</th><td><span class="quality-chip ${qClass}">${qualityLabel}</span></td></tr>` +
    `</tbody></table>`;
  if (node.card_disclosure) html += renderCardDisclosure(node.card_disclosure);
  const row = tableRowsByModel[node.id];
  if (row && row.method) {
    html += `<table class="node-detail-table"><tbody>` +
      `<tr><th>Method</th><td>${row.method}</td></tr>` +
      `<tr><th>GPU</th><td>${row.gpu || "—"}</td></tr>` +
      `<tr><th>Region</th><td>${row.region || "—"}</td></tr>` +
      `</tbody></table>`;
  }
  return html;
}

function showNodePane(nodeId) {
  const node = fullGraph.nodes.find((n) => n.id === nodeId);
  if (!node) return;
  $("pane-name").textContent = node.display_label || node.label || nodeId;
  $("pane-data").innerHTML = renderNodeDetail(node);

  const links = [];
  for (const e of fullGraph.edges) {
    if (e.from === nodeId) links.push({ id: e.to, dir: "→" });
    if (e.to === nodeId) links.push({ id: e.from, dir: "←" });
  }
  $("pane-links").innerHTML = links.length
    ? links
        .map(
          (l) =>
            `<li><button type="button" data-node="${l.id}">${l.dir} ${nodeDisplayName(l.id)}</button></li>`
        )
        .join("")
    : "<li><em>No connections</em></li>";

  $("pane-links").querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-node");
      selectedNodeId = id;
      graphFocused = true;
      network.setData(buildNetworkData(fullGraph, id));
      scheduleFit();
      showNodePane(id);
      updateGraphToolbarState();
    });
  });

  $("node-pane").classList.remove("hidden");
  $("pane-close").focus();
}

function closeNodePane() {
  $("node-pane").classList.add("hidden");
}

function clearGraphFocus() {
  selectedNodeId = null;
  graphFocused = false;
  hoverActive = false;
  hideHoverLabel();
  $("node-pane").classList.add("hidden");
  if (network && fullGraph.nodes.length) {
    network.setData(buildNetworkData(fullGraph, null));
    scheduleFit();
  }
  updateGraphToolbarState();
}

async function showFullDataset() {
  $("graph-view").value = "all";
  selectedNodeId = null;
  graphFocused = false;
  $("node-pane").classList.add("hidden");
  await loadDashboard();
}

async function loadDashboard() {
  const params = queryParams();
  $("csv-link").href = `/api/export.csv?${params.toString()}`;
  setDashboardLoading(true);
  try {
    const data = await fetchJson(`/api/dashboard?${params.toString()}`);
    if (!data.ok) {
      showError(data.error);
      renderConfidenceBanner(null);
      renderFamilyHeading(null);
      updateKpiFootnote(null);
      return;
    }
    lastRollup = data.rollup;
    renderFamilyHeading(data.rollup);
    renderConfidenceBanner(data.rollup);
    renderKpi(data.kpi);
    updateKpiFootnote(data.rollup);
    renderSummary(data.rollup);
    renderTable(data.table_rows, data.table_shown, data.table_total, data.rollup);
    renderHardware(data.hardware);
    renderCompare(data.compare);
    renderBarChart(data.bar_chart);
    initNetwork(data.graph);
    syncUrlState();
    markControlsApplied();
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    setDashboardLoading(false);
  }
}

async function copyShareLink() {
  const p = queryParams();
  if (isEmbedMode()) p.set("embed", "1");
  const url = `${window.location.origin}${window.location.pathname}?${p.toString()}`;
  try {
    await navigator.clipboard.writeText(url);
    const btn = $("copy-link-btn");
    const prev = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = prev; }, 2000);
  } catch {
    showError("Could not copy link — copy from the address bar after clicking View family.");
    syncUrlState();
  }
}

async function focusFamilyFromSelection() {
  const target = selectedNodeId || $("base-select").value;
  if (!target) return;
  $("base-select").value = target;
  $("graph-view").value = "family";
  await loadDashboard();
}

function renderFootprintTable(fp, label) {
  if (!fp) return `<p class="hub-status-miss"><em>No ${label} footprint.</em></p>`;
  return (
    `<table><caption>${label}</caption><tbody>` +
    `<tr><th>Carbon</th><td>${fp.carbon}</td></tr>` +
    `<tr><th>Water</th><td>${fp.water}</td></tr>` +
    `<tr><th>Energy</th><td>${fp.energy}</td></tr>` +
    `<tr><th>Quality</th><td>${fp.quality || "—"}</td></tr>` +
    `<tr><th>Method</th><td>${fp.method || "—"}</td></tr>` +
    `<tr><th>GPU</th><td>${fp.gpu || "—"}</td></tr>` +
    `<tr><th>Region</th><td>${fp.region || "—"}</td></tr>` +
    `</tbody></table>`
  );
}

function renderCardDisclosure(d) {
  if (!d) return "";
  let rows =
    `<tr><th>Scope</th><td>${d.scope || "—"} (${d.source || "model card"})</td></tr>` +
    `<tr><th>Carbon</th><td>${d.carbon || "—"}</td></tr>`;
  if (d.variant) rows += `<tr><th>Variant</th><td>${d.variant}</td></tr>`;
  if (d.gpu_hours) rows += `<tr><th>GPU hours</th><td>${d.gpu_hours}</td></tr>`;
  if (d.hardware) rows += `<tr><th>Hardware</th><td>${d.hardware}</td></tr>`;
  if (d.power_w) rows += `<tr><th>Power (W)</th><td>${d.power_w}</td></tr>`;
  if (d.carbon_market != null && d.carbon_market !== undefined) {
    rows += `<tr><th>Market-based GHG</th><td>${d.carbon_market}</td></tr>`;
  } else if (d.carbon_market_kg != null && d.carbon_market_kg !== undefined) {
    rows += `<tr><th>Market-based GHG</th><td>${d.carbon_market_kg} kgCO₂eq</td></tr>`;
  }
  rows += `<tr><th>Quality</th><td>${d.quality || "—"}</td></tr>`;
  return (
    `<table><caption>Published on model card</caption><tbody>${rows}</tbody></table>` +
    (d.notes ? `<p class="legend-note">${d.notes}</p>` : "")
  );
}

function renderHubLookup(data) {
  $("hub-panel").classList.remove("hidden");
  $("hub-model-link").innerHTML = data.model_url
    ? `<a href="${data.model_url}" target="_blank" rel="noopener"><strong>${data.model}</strong></a>`
    : `<strong>${data.model}</strong>`;

  const ingestBtn = $("hub-ingest-btn");
  ingestBtn.classList.add("hidden");
  ingestBtn.disabled = true;

  if (!data.ok) {
    $("hub-result").innerHTML = `<p class="hub-status-warn">${data.message || data.error}</p>`;
    return;
  }

  let html = "<ul>";
  html += `<li><strong>In dataset:</strong> ${data.in_store ? "yes" : "no"}`;
  if (data.in_store) {
    html += data.in_store_has_report
      ? ' <span class="hub-status-ok">(has dia_report)</span>'
      : ' <span class="hub-status-warn">(no dia_report)</span>';
  }
  html += "</li>";
  html += `<li><strong>On Hub card:</strong> `;
  html += data.hub_has_report
    ? '<span class="hub-status-ok">dia_report found</span>'
    : '<span class="hub-status-miss">no dia_report on model card</span>';
  html += "</li>";
  html += `<li><strong>Publisher carbon table:</strong> `;
  html += data.card_disclosure
    ? '<span class="hub-status-ok">found on model card</span>'
    : '<span class="hub-status-miss">not on this card</span>';
  html += "</li>";
  if (data.lineage && data.lineage.length) {
    html += `<li><strong>Lineage on card:</strong> ${data.lineage.map((e) => e.model).join(", ")}</li>`;
  }
  if (data.parse_errors && data.parse_errors.length) {
    html += `<li class="hub-status-warn"><strong>Parse notes:</strong> ${data.parse_errors.join("; ")}</li>`;
  }
  html += "</ul>";

  html += '<div class="hub-footprints">';
  html += renderFootprintTable(data.hub_footprint, "Hub dia_report");
  if (data.card_disclosure) {
    html += renderCardDisclosure(data.card_disclosure);
  }
  if (data.in_store) {
    html += renderFootprintTable(data.store_footprint, "Ingested copy");
  }
  html += "</div>";

  if (!data.hub_has_report && !data.in_store_has_report && !data.card_disclosure) {
    html +=
      "<p class=\"hub-status-miss\">Base carbon is unavailable until this model publishes a " +
      "<code>dia_report</code> on its Hugging Face card (or you enable imputation).</p>";
  } else if (!data.hub_has_report && !data.in_store_has_report && data.card_disclosure) {
    html +=
      "<p class=\"hub-status-warn\">No <code>dia_report</code> yet — showing publisher disclosure from the model card. " +
      "Family rollups still need DIA-format reports or imputation.</p>";
  }

  if (!data.writable) {
    html += "<p class=\"hub-status-warn\">Read-only mode — set <code>HF_TOKEN</code> to ingest.</p>";
  }

  $("hub-result").innerHTML = html;

  ingestBtn.textContent = "Ingest into dataset";
  ingestBtn.classList.add("hidden");
  ingestBtn.disabled = true;
  if (data.can_ingest && data.hub_has_report) {
    ingestBtn.classList.remove("hidden");
    ingestBtn.disabled = false;
    if (data.in_store && !data.in_store_has_report) {
      ingestBtn.textContent = "Refresh ingest from Hub";
    }
  }
}

async function checkBaseOnHub() {
  const model = $("base-select").value;
  if (!model) {
    showError("Pick a base model first.");
    return;
  }
  $("hub-check-btn").disabled = true;
  try {
    const data = await fetchJson(`/api/hub-lookup?model=${encodeURIComponent(model)}`);
    renderHubLookup(data);
  } catch (err) {
    renderHubLookup({ ok: false, model, model_url: null, message: err.message });
  } finally {
    $("hub-check-btn").disabled = false;
  }
}

async function ingestBaseFromHub() {
  const model = $("base-select").value;
  $("hub-ingest-btn").disabled = true;
  try {
    const res = await fetchJson(`/api/hub-ingest?model=${encodeURIComponent(model)}`, { method: "POST" });
    if (!res.ok) {
      showError(res.error || "Ingest failed");
      return;
    }
    await loadMeta();
    await loadDashboard();
    await checkBaseOnHub();
    const note = res.has_report ? "Ingested — dashboard refreshed." : "Ingested (no dia_report on card).";
    $("hub-result").insertAdjacentHTML("beforeend", `<p class="hub-status-ok">${note}</p>`);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    $("hub-ingest-btn").disabled = false;
  }
}

$("apply-btn").addEventListener("click", () => {
  clearTimeout(loadDebounceTimer);
  loadDashboard();
});
$("copy-link-btn").addEventListener("click", copyShareLink);
["base-select", "compare-select", "graph-view", "row-filter"].forEach((id) => {
  $(id).addEventListener("change", scheduleLoadDashboard);
});
$("graph-color-by").addEventListener("change", () => {
  graphColorBy = $("graph-color-by").value;
  refreshGraphColors();
});
$("table-search").addEventListener("input", () => {
  tableSearchQuery = $("table-search").value;
  applyTableView();
});
$("footprint-table").querySelector("thead").addEventListener("click", (e) => {
  const th = e.target.closest("th.sortable");
  if (!th) return;
  const key = th.getAttribute("data-sort");
  if (tableSortKey === key) {
    tableSortDir = tableSortDir === "asc" ? "desc" : "asc";
  } else {
    tableSortKey = key;
    tableSortDir = key === "carbon" ? "desc" : "asc";
  }
  applyTableView();
});
$("refresh-btn").addEventListener("click", async () => {
  await fetch("/api/refresh", { method: "POST" });
  await loadMeta();
  await loadDashboard();
});
$("clear-focus-btn").addEventListener("click", clearGraphFocus);
$("full-dataset-btn").addEventListener("click", showFullDataset);
$("focus-family-btn").addEventListener("click", focusFamilyFromSelection);
$("zoom-in-btn").addEventListener("click", () => {
  if (!network) return;
  network.moveTo({ scale: clampGraphScale(network.getScale() * 1.25), animation: true });
});
$("zoom-out-btn").addEventListener("click", () => {
  if (!network) return;
  network.moveTo({ scale: clampGraphScale(network.getScale() * 0.8), animation: true });
});
$("zoom-fit-btn").addEventListener("click", () => fitGraph(28));
window.addEventListener("resize", () => {
  syncTopbarHeight();
  scheduleFit();
});
$("pane-close").addEventListener("click", closeNodePane);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("node-pane").classList.contains("hidden")) {
    $("node-pane").classList.add("hidden");
    e.preventDefault();
    return;
  }
  if (!$("hub-panel").classList.contains("hidden")) {
    $("hub-panel").classList.add("hidden");
    e.preventDefault();
  }
});
$("hub-check-btn").addEventListener("click", checkBaseOnHub);
$("hub-ingest-btn").addEventListener("click", ingestBaseFromHub);
$("hub-panel-close").addEventListener("click", () => {
  $("hub-panel").classList.add("hidden");
});

(async function init() {
  try {
    applyEmbedMode();
    const urlState = readUrlState();
    applyControlsFromUrl(urlState);
    await loadMeta(urlState);
    markControlsApplied();
    await loadDashboard();
  } catch (err) {
    showError(err.message || String(err));
  }
})();
