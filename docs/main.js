/* Sustainable Open-Source AI site – interactive table + scatter (no external libs) */
const $ = (sel) => document.querySelector(sel);

const DATA = [
  { model:"GPT-3", year:2020, params:"175B", open:false, tco2:{min:552,max:552,raw:"552"}, trees:{min:22080,max:22080,raw:"22,080"}, water:{min:2.5,max:5.6,raw:"2.5–5.6"}, status:"Est." },
  { model:"BLOOM⋆", year:2022, params:"176B", open:true,  tco2:{min:37.6,max:37.6,raw:"37.6"}, trees:{min:1504,max:1504,raw:"1,504"}, water:{min:0.43,max:0.43,raw:"0.43"}, status:"R" },
  { model:"OPT⋆", year:2022, params:"175B", open:true,  tco2:{min:75,max:75,raw:"75"}, trees:{min:3000,max:3000,raw:"3,000"}, water:{min:0.6,max:1.3,raw:"0.6–1.3"}, status:"Est." },
  { model:"Falcon 180B⋆", year:2023, params:"180B", open:true, tco2:{min:1200,max:1200,raw:"∼1,200"}, trees:{min:48000,max:48000,raw:"∼48,000"}, water:{min:5.0,max:11.0,raw:"5.0–11.0"}, status:"Est." },
  { model:"Llama 2⋆", year:2023, params:"70B", open:true, tco2:{min:539,max:539,raw:"539"}, trees:{min:21560,max:21560,raw:"21,560"}, water:{min:2.4,max:5.3,raw:"2.4–5.3"}, status:"R" },
  { model:"Mistral 7B⋆", year:2023, params:"7.3B", open:true, tco2:null, trees:null, water:null, status:"N/D" },
  { model:"GPT-4", year:2023, params:"N/D", open:false, tco2:{min:4240,max:18870,raw:"4,240–18,870"}, trees:{min:169600,max:754800,raw:"169,600–754,800"}, water:{min:76,max:170,raw:"76–170"}, status:"Est." },
  { model:"Llama 3⋆", year:2024, params:"8B / 70B", open:true, tco2:{min:2290,max:2290,raw:"2,290"}, trees:{min:91600,max:91600,raw:"91,600"}, water:{min:10.2,max:22.6,raw:"10.2–22.6"}, status:"R" },
  { model:"Llama 3.1⋆", year:2024, params:"405B", open:true, tco2:{min:8930,max:8930,raw:"8,930"}, trees:{min:357200,max:357200,raw:"357,200"}, water:{min:40,max:88,raw:"40–88"}, status:"Est." },
  { model:"DeepSeek-V3⋆", year:2024, params:"671B", open:true, tco2:{min:545,max:545,raw:"∼545"}, trees:{min:21800,max:21800,raw:"∼21,800"}, water:{min:1.9,max:4.3,raw:"1.9–4.3"}, status:"Est." },
];

// State
let sortKey = "year";
let sortDir = "desc";
let filtered = [...DATA];

function midpoint(r){
  if(!r) return null;
  const a = Number(r.min);
  const b = Number(r.max);
  if(Number.isFinite(a) && Number.isFinite(b)) return (a + b) / 2;
  return null;
}

function fmtNum(x){
  if(x === null || x === undefined) return "—";
  if(!Number.isFinite(x)) return "—";
  // compact-ish but readable
  if(x >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if(x >= 10) return x.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return x.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function compare(a, b, key){
  const va = a[key];
  const vb = b[key];
  // Special cases for range objects
  if(key === "tco2" || key === "trees" || key === "water"){
    const ma = midpoint(va);
    const mb = midpoint(vb);
    if(ma === null && mb === null) return 0;
    if(ma === null) return 1;
    if(mb === null) return -1;
    return ma - mb;
  }
  if(typeof va === "number" && typeof vb === "number") return va - vb;
  if(typeof va === "boolean" && typeof vb === "boolean") return (va === vb) ? 0 : (va ? -1 : 1);
  return String(va ?? "").localeCompare(String(vb ?? ""));
}

function applyFilters(){
  const q = ($("#searchBox").value || "").trim().toLowerCase();
  const openOnly = $("#openOnly").checked;
  
  filtered = DATA.filter(d => {
    if(openOnly && !d.open) return false;
    if(q && !d.model.toLowerCase().includes(q) && !d.params.toLowerCase().includes(q)) return false;
    return true;
  });
  
  filtered.sort((a,b) => {
    const c = compare(a,b,sortKey);
    return sortDir === "asc" ? c : -c;
  });
  
  renderTable();
  renderScatter();
}

function renderTable(){
  const tbody = $("#modelsTbody");
  tbody.innerHTML = "";
  
  for(const d of filtered){
    const tr = document.createElement("tr");
    
    const tdModel = document.createElement("td");
    tdModel.innerHTML = `<strong>${escapeHtml(d.model)}</strong>`;
    tr.appendChild(tdModel);
    
    const tdYear = document.createElement("td");
    tdYear.className = "num";
    tdYear.textContent = d.year ?? "—";
    tr.appendChild(tdYear);
    
    const tdParams = document.createElement("td");
    tdParams.textContent = d.params ?? "—";
    tr.appendChild(tdParams);
    
    const tdOpen = document.createElement("td");
    tdOpen.className = "center";
    tdOpen.textContent = d.open ? "⋆" : "—";
    tr.appendChild(tdOpen);
    
    const tdT = document.createElement("td");
    tdT.className = "num";
    tdT.textContent = d.tco2 ? d.tco2.raw : "—";
    tr.appendChild(tdT);
    
    const tdTrees = document.createElement("td");
    tdTrees.className = "num";
    tdTrees.textContent = d.trees ? d.trees.raw : "—";
    tr.appendChild(tdTrees);
    
    const tdW = document.createElement("td");
    tdW.className = "num";
    tdW.textContent = d.water ? d.water.raw : "—";
    tr.appendChild(tdW);
    
    const tdS = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = d.status ?? "—";
    tdS.appendChild(badge);
    tr.appendChild(tdS);
    
    tbody.appendChild(tr);
  }
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
  }[c]));
}

/* Scatter plot */
function renderScatter(){
  const canvas = $("#scatter");
  const ctx = canvas.getContext("2d");
  const tooltip = $("#tooltip");
  const useMid = $("#showMidpoints").checked;
  const scale = $("#scaleSelect").value; // log|linear
  
  // Collect points
  const pts = filtered
    .map(d => {
      const x = d.water ? (useMid ? midpoint(d.water) : d.water.max) : null;
      const y = d.tco2 ? (useMid ? midpoint(d.tco2) : d.tco2.max) : null;
      if(x === null || y === null) return null;
      return { x, y, d };
    })
    .filter(Boolean);
  
  // Clear
  ctx.clearRect(0,0,canvas.width,canvas.height);
  
  // Layout
  const pad = { l:100, r:140, t:50, b:80 };
  const W = canvas.width, H = canvas.height;
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  
  // Determine bounds
  const xs = pts.map(p => p.x);
  const ys = pts.map(p => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  
  // Helpers
  const log10 = (v) => Math.log(v) / Math.log(10);
  const sx = (v) => {
    if(scale === "log"){
      const a = log10(Math.max(xMin, 0.1));
      const b = log10(xMax);
      return pad.l + (log10(Math.max(v, 0.1)) - a) / (b - a) * iw;
    }
    return pad.l + (v - xMin) / (xMax - xMin) * iw;
  };
  const sy = (v) => {
    if(scale === "log"){
      const a = log10(Math.max(yMin, 1));
      const b = log10(yMax);
      return pad.t + (1 - (log10(Math.max(v, 1)) - a) / (b - a)) * ih;
    }
    return pad.t + (1 - (v - yMin) / (yMax - yMin)) * ih;
  };
  
  // Grid
  ctx.strokeStyle = "#e4e7ec";
  ctx.lineWidth = 1;
  const {xTicks, yTicks} = drawGrid(ctx, {pad, iw, ih, xMin, xMax, yMin, yMax, scale, sx, sy});
  
  // Axes
  ctx.strokeStyle = "#475467";
  ctx.lineWidth = 2;
  // x axis
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + ih);
  ctx.lineTo(pad.l + iw, pad.t + ih);
  ctx.stroke();
  // y axis
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, pad.t + ih);
  ctx.stroke();
  
  // Axis labels
  ctx.fillStyle = "#1e293b";
  ctx.font = "bold 15px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Water Consumption (ML)", W / 2, H - 15);
  
  ctx.save();
  ctx.translate(25, H / 2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText("Carbon Emissions (tCO₂eq)", 0, 0);
  ctx.restore();
  
  // Points with labels
  const hit = []; // for hover
  const labelPositions = [];
  
  for(const p of pts){
    const cx = sx(p.x);
    const cy = sy(p.y);
    
    // Draw point
    ctx.beginPath();
    ctx.fillStyle = p.d.open ? "#0EA5E9" : "#64748B";
    ctx.arc(cx, cy, p.d.open ? 8 : 6, 0, Math.PI*2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
    
    hit.push({ cx, cy, r:10, p });
  }
  
  // Draw labels with smart positioning
  for(const h of hit){
    const {cx, cy, p} = h;
    ctx.font = "bold 11px sans-serif";
    const labelWidth = ctx.measureText(p.d.model).width;
    
    const spacing = 15;
    const positions = [
      { x: cx + spacing, y: cy + 4, align: 'left' },
      { x: cx - spacing, y: cy + 4, align: 'right' },
      { x: cx, y: cy - spacing, align: 'center' },
      { x: cx, y: cy + spacing + 4, align: 'center' },
      { x: cx + spacing, y: cy - 8, align: 'left' },
      { x: cx + spacing, y: cy + 16, align: 'left' },
      { x: cx - spacing, y: cy - 8, align: 'right' },
      { x: cx - spacing, y: cy + 16, align: 'right' },
      { x: cx + spacing * 1.5, y: cy + 4, align: 'left' },
      { x: cx - spacing * 1.5, y: cy + 4, align: 'right' },
    ];
    
    let bestPos = positions[0];
    let minOverlap = Infinity;
    
    for (const pos of positions) {
      let overlapScore = 0;
      
      for (const existing of labelPositions) {
        const dx = Math.abs(existing.x - pos.x);
        const dy = Math.abs(existing.y - pos.y);
        const minDx = (existing.width + labelWidth) / 2 + 15;
        const minDy = 20;
        
        if (dx < minDx && dy < minDy) {
          overlapScore += (minDx - dx) + (minDy - dy);
        }
      }
      
      if (pos.x < pad.l + 10 || pos.x > W - pad.r - 10) {
        overlapScore += 100;
      }
      if (pos.y < pad.t + 10 || pos.y > H - pad.b - 10) {
        overlapScore += 100;
      }
      
      if (overlapScore < minOverlap) {
        minOverlap = overlapScore;
        bestPos = pos;
      }
    }
    
    const labelX = bestPos.x;
    const labelY = bestPos.y;
    const labelAlign = bestPos.align;
    
    // Draw connector line if far
    const dist = Math.sqrt((labelX - cx) ** 2 + (labelY - cy) ** 2);
    if (dist > 20) {
      ctx.strokeStyle = p.d.open ? '#0EA5E9' : '#64748B';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(labelX - (labelAlign === 'left' ? 5 : labelAlign === 'right' ? -5 : 0), labelY - 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    labelPositions.push({ x: labelX, y: labelY, width: labelWidth });
    
    ctx.fillStyle = p.d.open ? '#0c4a6e' : '#334155';
    ctx.textAlign = labelAlign;
    ctx.fillText(p.d.model, labelX, labelY);
  }
  
  // Hover handling
  let raf = null;
  const onMove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (ev.clientX - rect.left) * scaleX;
    const my = (ev.clientY - rect.top) * scaleY;
    
    let found = null;
    let minDist = Infinity;
    
    for(const h of hit){
      const dx = mx - h.cx, dy = my - h.cy;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if(dist <= h.r && dist < minDist){
        found = h;
        minDist = dist;
      }
    }
    
    if(!found){
      tooltip.style.opacity = 0;
      canvas.style.cursor = "crosshair";
      return;
    }
    
    const d = found.p.d;
    const waterEquiv = Math.round((d.water ? midpoint(d.water) : 0) * 2000000);
    const treesYears = Math.round((d.trees ? midpoint(d.trees) : 0) / 50);
    
    tooltip.innerHTML =
      `<b>${escapeHtml(d.model)}</b> (${d.year})<br/>` +
      `<strong>Parameters:</strong> ${escapeHtml(d.params)}<br/>` +
      `<strong>Carbon:</strong> ${d.tco2 ? escapeHtml(d.tco2.raw) : "—"} tCO₂eq<br/>` +
      `<strong>Water:</strong> ${d.water ? escapeHtml(d.water.raw) : "—"} ML (~${waterEquiv.toLocaleString()} 500ml bottles)<br/>` +
      `<strong>Tree offset:</strong> ${d.trees ? escapeHtml(d.trees.raw) : "—"} trees (~${treesYears} years)<br/>` +
      `<strong>Type:</strong> ${d.open ? 'Open-source ⋆' : 'Proprietary'}<br/>` +
      `<strong>Source:</strong> ${escapeHtml(d.status)}`;
    
    tooltip.style.left = `${ev.clientX + 15}px`;
    tooltip.style.top = `${ev.clientY + 15}px`;
    tooltip.style.opacity = 1;
    tooltip.style.transform = "translateY(0)";
    canvas.style.cursor = "pointer";
  };
  
  const onLeave = () => {
    tooltip.style.opacity = 0;
    canvas.style.cursor = "crosshair";
  };
  
  canvas.onmousemove = (ev) => {
    if(raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => onMove(ev));
  };
  canvas.onmouseleave = onLeave;
}

function drawGrid(ctx, cfg){
  const {pad, iw, ih, xMin, xMax, yMin, yMax, scale, sx, sy} = cfg;
  
  ctx.fillStyle = "#64748b";
  ctx.strokeStyle = "#e4e7ec";
  ctx.lineWidth = 1;
  ctx.font = "13px sans-serif";
  
  const log10 = (v) => Math.log(v) / Math.log(10);
  
  let xTicks, yTicks;
  
  if(scale === "log"){
    // Logarithmic ticks - use powers of 10
    xTicks = generateLogTicks(xMin, xMax);
    yTicks = generateLogTicks(yMin, yMax);
  } else {
    // Linear ticks - use nice round numbers
    xTicks = generateLinearTicks(xMin, xMax, 5);
    yTicks = generateLinearTicks(yMin, yMax, 5);
  }
  
  // Draw x-axis grid and labels
  ctx.textAlign = "center";
  for(const v of xTicks){
    const x = sx(v);
    
    // Grid line
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ih);
    ctx.stroke();
    
    // Label
    const label = formatTickLabel(v);
    ctx.fillText(label, x, pad.t + ih + 25);
  }
  
  // Draw y-axis grid and labels
  ctx.textAlign = "right";
  for(const v of yTicks){
    const y = sy(v);
    
    // Grid line
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + iw, y);
    ctx.stroke();
    
    // Label
    const label = formatTickLabel(v);
    ctx.fillText(label, pad.l - 12, y + 5);
  }
  
  return {xTicks, yTicks};
}

function generateLogTicks(min, max){
  // Generate clean logarithmic ticks
  const minLog = Math.floor(Math.log10(Math.max(min, 0.1)));
  const maxLog = Math.ceil(Math.log10(max));
  
  const ticks = [];
  
  // Always include powers of 10
  for(let exp = minLog; exp <= maxLog; exp++){
    const val = Math.pow(10, exp);
    if(val >= min * 0.9 && val <= max * 1.1){
      ticks.push(val);
    }
  }
  
  // If we have too few ticks, add intermediate values (2, 5)
  if(ticks.length < 4){
    const intermediate = [];
    for(let exp = minLog; exp <= maxLog; exp++){
      const base = Math.pow(10, exp);
      for(const mult of [2, 5]){
        const val = base * mult;
        if(val >= min && val <= max && !ticks.includes(val)){
          intermediate.push(val);
        }
      }
    }
    ticks.push(...intermediate);
    ticks.sort((a, b) => a - b);
  }
  
  return ticks;
}

function generateLinearTicks(min, max, targetCount){
  // Generate nice round linear ticks
  const range = max - min;
  const roughStep = range / (targetCount - 1);
  
  // Find a nice round step size
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const normalized = roughStep / magnitude;
  
  let niceStep;
  if(normalized <= 1) niceStep = magnitude;
  else if(normalized <= 2) niceStep = 2 * magnitude;
  else if(normalized <= 5) niceStep = 5 * magnitude;
  else niceStep = 10 * magnitude;
  
  // Generate ticks
  const ticks = [];
  const start = Math.ceil(min / niceStep) * niceStep;
  
  for(let v = start; v <= max; v += niceStep){
    ticks.push(v);
  }
  
  // Ensure we have at least min and max represented
  if(ticks.length === 0 || ticks[0] > min * 1.1){
    ticks.unshift(min);
  }
  if(ticks[ticks.length - 1] < max * 0.9){
    ticks.push(max);
  }
  
  return ticks;
}

function formatTickLabel(val){
  if(val === 0) return "0";
  
  // For very small values
  if(val < 0.01) return val.toExponential(0);
  
  // For values < 1, show 1-2 decimal places
  if(val < 1) return val.toFixed(2).replace(/\.?0+$/, '');
  
  // For values 1-999, show as-is or with 1 decimal if needed
  if(val < 1000){
    if(val % 1 === 0) return val.toString();
    return val.toFixed(1).replace(/\.0$/, '');
  }
  
  // For thousands, use K notation
  if(val < 1000000){
    const k = val / 1000;
    if(k % 1 === 0) return k + "K";
    return k.toFixed(1).replace(/\.0$/, '') + "K";
  }
  
  // For millions, use M notation
  const m = val / 1000000;
  if(m % 1 === 0) return m + "M";
  return m.toFixed(1).replace(/\.0$/, '') + "M";
}

// Remove the old ticks function since we have new ones

/* Modal + bib copy */
function setupModal(){
  const modal = $("#modal");
  const modalImg = $("#modalImg");
  const modalCaption = $("#modalCaption");
  
  document.querySelectorAll(".figure-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const src = btn.getAttribute("data-zoom-src");
      const fig = btn.closest("figure");
      const cap = fig ? fig.querySelector("figcaption")?.textContent : "";
      modalImg.src = src;
      modalImg.alt = cap || "Figure";
      modalCaption.textContent = cap || "";
      modal.setAttribute("aria-hidden", "false");
    });
  });
  
  modal.addEventListener("click", (e) => {
    if(e.target && e.target.getAttribute("data-close") === "1"){
      modal.setAttribute("aria-hidden", "true");
      modalImg.src = "";
    }
  });
  
  document.addEventListener("keydown", (e) => {
    if(e.key === "Escape" && modal.getAttribute("aria-hidden") === "false"){
      modal.setAttribute("aria-hidden", "true");
      modalImg.src = "";
    }
  });
}

function setupSorting(){
  document.querySelectorAll("#modelsTable thead th").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-key");
      if(!key) return;
      if(sortKey === key){
        sortDir = (sortDir === "asc") ? "desc" : "asc";
      }else{
        sortKey = key;
        sortDir = (key === "year") ? "desc" : "asc";
      }
      applyFilters();
    });
  });
}

function setupFilters(){
  $("#searchBox").addEventListener("input", applyFilters);
  $("#openOnly").addEventListener("change", applyFilters);
  $("#showMidpoints").addEventListener("change", applyFilters);
  $("#scaleSelect").addEventListener("change", applyFilters);
}

function setupBibCopy(){
  const btn = $("#copyBibBtn");
  const bib = $("#bibBlock");
  btn.addEventListener("click", async () => {
    const txt = bib.innerText;
    try{
      await navigator.clipboard.writeText(txt);
      btn.textContent = "✓ Copied!";
      setTimeout(() => btn.textContent = "📋 Copy BibTeX", 2000);
    }catch{
      // fallback: select
      const range = document.createRange();
      range.selectNodeContents(bib);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      btn.textContent = "Select & copy";
      setTimeout(() => btn.textContent = "📋 Copy BibTeX", 2000);
    }
  });
}

function setupActiveNav(){
  const links = Array.from(document.querySelectorAll(".top-nav a"));
  const sections = links
    .map(a => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);

  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if(e.isIntersecting){
        links.forEach(a => a.classList.remove("active"));
        const id = "#" + e.target.id;
        const hit = links.find(a => a.getAttribute("href") === id);
        if(hit) hit.classList.add("active");
      }
    });
  }, { rootMargin: "-30% 0px -60% 0px", threshold: 0.01 });

  sections.forEach(s => io.observe(s));
}

function init(){
  setupModal();
  setupSorting();
  setupFilters();
  setupBibCopy();
  setupActiveNav();
  applyFilters();
}

init();