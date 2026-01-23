/* Beyond Efficiency site – interactive table + scatter (no external libs) */

const $ = (sel) => document.querySelector(sel);

const DATA = [
  { model:"GPT-3", year:2020, params:"175B", open:false, tco2:{min:552,max:552,raw:"552"}, trees:{min:22080,max:22080,raw:"22,080"}, water:{min:2.5,max:5.6,raw:"2.5–5.6"}, status:"Est." },
  { model:"BLOOM", year:2022, params:"176B", open:true,  tco2:{min:24.7,max:50.5,raw:"24.7–50.5"}, trees:{min:988,max:2020,raw:"988–2,020"}, water:{min:0.43,max:0.43,raw:"0.43"}, status:"R" },
  { model:"OPT", year:2022, params:"175B", open:true,  tco2:{min:75,max:75,raw:"75"}, trees:{min:3000,max:3000,raw:"3,000"}, water:{min:0.6,max:1.3,raw:"0.6–1.3"}, status:"Est." },
  { model:"Falcon 180B", year:2023, params:"180B", open:true, tco2:{min:1200,max:1200,raw:"∼1,200"}, trees:{min:48000,max:48000,raw:"∼48,000"}, water:{min:5.0,max:11.0,raw:"5.0–11.0"}, status:"Est." },
  { model:"Llama 2", year:2023, params:"70B", open:true, tco2:{min:539,max:539,raw:"539"}, trees:{min:21560,max:21560,raw:"21,560"}, water:{min:2.4,max:5.3,raw:"2.4–5.3"}, status:"R" },
  { model:"Mistral 7B", year:2023, params:"7.3B", open:true, tco2:null, trees:null, water:null, status:"N/D" },
  { model:"GPT-4", year:2023, params:"N/D", open:false, tco2:{min:4240,max:18870,raw:"4,240–18,870"}, trees:{min:169600,max:754800,raw:"169,600–754,800"}, water:{min:52,max:185,raw:"52–185"}, status:"Est." },
  { model:"Llama 3", year:2024, params:"8B / 70B", open:true, tco2:{min:2290,max:2290,raw:"2,290"}, trees:{min:91600,max:91600,raw:"91,600"}, water:{min:10.2,max:22.6,raw:"10.2–22.6"}, status:"R" },
  { model:"Llama 3.1", year:2024, params:"405B", open:true, tco2:{min:8930,max:8930,raw:"8,930"}, trees:{min:357200,max:357200,raw:"357,200"}, water:{min:40,max:88,raw:"40–88"}, status:"Est." },
  { model:"DeepSeek-V3", year:2024, params:"671B", open:true, tco2:{min:545,max:545,raw:"∼545"}, trees:{min:21800,max:21800,raw:"∼21,800"}, water:{min:1.9,max:4.3,raw:"1.9–4.3"}, status:"Est." },
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
    if(q && !d.model.toLowerCase().includes(q)) return false;
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
    tdModel.innerHTML = `<b>${escapeHtml(d.model)}</b>`;
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
    tdOpen.innerHTML = d.open ? `<span class="badge">⋆ open</span>` : `<span class="badge">closed</span>`;
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
    tdS.textContent = d.status ?? "—";
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
  const pad = { l:58, r:20, t:18, b:48 };
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
      const a = log10(xMin), b = log10(xMax);
      return pad.l + (log10(v) - a) / (b - a) * iw;
    }
    return pad.l + (v - xMin) / (xMax - xMin) * iw;
  };
  const sy = (v) => {
    if(scale === "log"){
      const a = log10(yMin), b = log10(yMax);
      return pad.t + (1 - (log10(v) - a) / (b - a)) * ih;
    }
    return pad.t + (1 - (v - yMin) / (yMax - yMin)) * ih;
  };

  // Axes
  ctx.strokeStyle = "#d0d5dd";
  ctx.lineWidth = 1;

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

  // Labels
  ctx.fillStyle = "#475467";
  ctx.font = "12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
  ctx.fillText("Water (ML, training)", pad.l, H - 16);
  ctx.save();
  ctx.translate(16, pad.t + ih/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText("Carbon (tCO₂e, training)", 0, 0);
  ctx.restore();

  // Simple ticks (5)
  drawTicks(ctx, {pad, iw, ih, xMin, xMax, yMin, yMax, scale});

  // Points
  const hit = []; // for hover
  for(const p of pts){
    const cx = sx(p.x);
    const cy = sy(p.y);

    ctx.beginPath();
    ctx.fillStyle = p.d.open ? "#0b5fff" : "#101828";
    ctx.globalAlpha = 0.85;
    ctx.arc(cx, cy, 5, 0, Math.PI*2);
    ctx.fill();
    ctx.globalAlpha = 1;

    hit.push({ cx, cy, r:7, p });
  }

  // Hover handling
  let raf = null;
  const onMove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (ev.clientX - rect.left) * (canvas.width / rect.width);
    const my = (ev.clientY - rect.top) * (canvas.height / rect.height);

    let found = null;
    for(const h of hit){
      const dx = mx - h.cx, dy = my - h.cy;
      if(dx*dx + dy*dy <= h.r*h.r){ found = h; break; }
    }

    if(!found){
      tooltip.style.opacity = 0;
      return;
    }

    const d = found.p.d;
    tooltip.innerHTML =
      `<div><b>${escapeHtml(d.model)}</b> ${d.open ? "⋆" : ""}</div>` +
      `<div>tCO₂e: ${d.tco2 ? escapeHtml(d.tco2.raw) : "—"}</div>` +
      `<div>Water (ML): ${d.water ? escapeHtml(d.water.raw) : "—"}</div>` +
      `<div class="muted">(${escapeHtml(d.status)})</div>`;

    // position tooltip near cursor
    const tx = ev.clientX + 14;
    const ty = ev.clientY - 10;
    tooltip.style.left = `${tx}px`;
    tooltip.style.top = `${ty}px`;
    tooltip.style.opacity = 1;
    tooltip.style.transform = "translateY(-6px)";
  };

  const onLeave = () => { tooltip.style.opacity = 0; };

  canvas.onmousemove = (ev) => {
    if(raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => onMove(ev));
  };
  canvas.onmouseleave = onLeave;
}

function drawTicks(ctx, cfg){
  const {pad, iw, ih, xMin, xMax, yMin, yMax, scale} = cfg;
  const tickN = 5;

  ctx.fillStyle = "#667085";
  ctx.strokeStyle = "#e4e7ec";
  ctx.lineWidth = 1;
  ctx.font = "12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";

  const log10 = (v) => Math.log(v) / Math.log(10);

  const xVals = ticks(xMin, xMax, tickN, scale);
  const yVals = ticks(yMin, yMax, tickN, scale);

  // x ticks
  for(const v of xVals){
    const x = (scale === "log")
      ? pad.l + (log10(v) - log10(xMin)) / (log10(xMax) - log10(xMin)) * iw
      : pad.l + (v - xMin) / (xMax - xMin) * iw;

    ctx.beginPath();
    ctx.moveTo(x, pad.t + ih);
    ctx.lineTo(x, pad.t);
    ctx.stroke();

    ctx.fillText(fmtNum(v), x - 10, pad.t + ih + 18);
  }

  // y ticks
  for(const v of yVals){
    const y = (scale === "log")
      ? pad.t + (1 - (log10(v) - log10(yMin)) / (log10(yMax) - log10(yMin))) * ih
      : pad.t + (1 - (v - yMin) / (yMax - yMin)) * ih;

    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + iw, y);
    ctx.stroke();

    ctx.fillText(fmtNum(v), 10, y + 4);
  }
}

function ticks(min, max, n, scale){
  if(scale === "log"){
    const a = Math.floor(Math.log10(min));
    const b = Math.ceil(Math.log10(max));
    const out = [];
    for(let e=a; e<=b; e++){
      out.push(Math.pow(10,e));
    }
    // keep it short-ish
    if(out.length > n){
      // sample
      const step = Math.ceil(out.length / n);
      return out.filter((_,i)=> i%step===0);
    }
    return out;
  }

  // linear ticks
  const span = max - min;
  const step = span / (n-1);
  const out = [];
  for(let i=0;i<n;i++){
    out.push(min + step*i);
  }
  return out;
}

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
      btn.textContent = "Copied!";
      setTimeout(() => btn.textContent = "Copy BibTeX", 1100);
    }catch{
      // fallback: select
      const range = document.createRange();
      range.selectNodeContents(bib);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      btn.textContent = "Select & copy";
      setTimeout(() => btn.textContent = "Copy BibTeX", 1300);
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
