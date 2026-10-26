/* SQL Explanation Grader: RAG dashboard (no build step, no framework). */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (r) => (r && r.rate != null ? `${(r.rate * 100).toFixed(1)}%` : "n/a");
const frac = (r) => (r && r.n ? `${r.k}/${r.n}` : "");
const num = (d, digits = 2) => (d && d.mean != null ? Number(d.mean).toFixed(digits) : "n/a");
const fmt = (v, digits = 2) => (v == null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(digits));

const state = { mode: "rag", examples: [], experiments: [], currentExp: null, items: [], selectedItem: null, label: null };

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json", ...(opts.headers || {}) }, ...opts });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(`${res.status}: ${detail}`);
  }
  return data;
}

/* ------------------------------------------------------------------ theme */
(function theme() {
  const saved = (() => { try { return localStorage.getItem("theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* storage unavailable */ }
  });
})();

/* ------------------------------------------------------------------- tabs */
function showTab(name) {
  $$("nav.tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  try { localStorage.setItem("tab", name); } catch { /* ignore */ }
  if (name === "knowledge") loadKnowledge();
  if (name === "reports") loadReports();
  if (name === "evaluation") { if (!state.experiments.length) loadExperiments(); else requestAnimationFrame(rerenderCharts); }
}
$$("nav.tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

/* ----------------------------------------------------------------- health */
async function loadHealth() {
  const el = $("#health");
  try {
    const h = await api("/api/health");
    const llm = h.llm || {};
    const idx = h.index || {};
    const ok = h.status === "ok" && llm.generator_available;
    el.className = `badge ${ok ? "good" : "warn"}`;
    el.innerHTML = `<span class="dot"></span>${ok ? "Ready" : "Degraded"}: ${esc(idx.chunks ?? "?")} chunks indexed, LLM ${llm.reachable ? "reachable" : "busy or unreachable"}`;
    el.title = llm.reachable
      ? `Generator ${llm.generator_model} ${llm.generator_available ? "available" : "NOT pulled"}; judge ${llm.judge_model} ${llm.judge_available ? "available" : "NOT pulled"}`
      : `Ollama not reachable: ${llm.error || ""}`;
  } catch (e) {
    el.className = "badge bad";
    el.innerHTML = `<span class="dot"></span>API unreachable`;
    el.title = String(e);
  }
}

/* ------------------------------------------------------------- playground */
$$("#mode-group button").forEach((b) => b.addEventListener("click", () => {
  state.mode = b.dataset.mode;
  $$("#mode-group button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
}));

async function loadExamples() {
  try {
    const data = await api("/api/examples");
    state.examples = data.examples;
    const sel = $("#example-select");
    const groups = { dev: "Development set", abstention_probe: "Abstention probes (uncovered features)" };
    for (const [split, label] of Object.entries(groups)) {
      const og = document.createElement("optgroup");
      og.label = label;
      data.examples.filter((x) => x.split === split).forEach((x, i) => {
        const o = document.createElement("option");
        o.value = `${split}:${x.id}`;
        o.textContent = `${x.id} · ${x.feature || x.expected || ""}`.slice(0, 70);
        og.appendChild(o);
      });
      if (og.children.length) sel.appendChild(og);
    }
  } catch { /* examples are optional */ }
}
$("#example-select").addEventListener("change", (e) => {
  const [split, id] = e.target.value.split(":");
  const ex = state.examples.find((x) => x.split === split && x.id === id);
  if (ex) { $("#sql-input").value = ex.sql; $("#schema-input").value = ex.schema; }
});

const STATUS = {
  grounded: ["good", "Grounded", "Every SQL feature was covered by retrieved documentation and the answer cites valid sources."],
  weakly_grounded: ["warn", "Weakly grounded", "Evidence was retrieved, but the answer has no valid citation or cites a source that was not provided."],
  abstained: ["bad", "Abstained", "The system refused because the retrieved evidence was insufficient under the grounding policy."],
  fallback_ungrounded: ["warn", "Fallback: not grounded", "Evidence was insufficient; this answer was written without sources and is NOT evidence-grounded."],
  ungrounded_by_policy: ["warn", "Not grounded (policy off)", "Evidence was insufficient but the policy allowed an answer anyway."],
  not_applicable: ["info", "Baseline: no retrieval", "Prompt-only explanation. No documentation was retrieved."],
};

function statusBadge(status) {
  const [cls, label, title] = STATUS[status] || ["", status, ""];
  const icon = { good: "✓", warn: "!", bad: "⊘", info: "i" }[cls] || "";
  return `<span class="badge ${cls}" title="${esc(title)}"><span aria-hidden="true">${icon}</span>${esc(label)}</span>`;
}

function renderExplanationText(text, validLabels) {
  // Turn [S1] / [S1, S2] markers into buttons that highlight the source card.
  return esc(text).replace(/\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]/g, (_, group) => group.split(/[,;]/).map((raw) => {
    let lab = raw.trim().toUpperCase();
    if (!lab.startsWith("S")) lab = `S${lab}`;
    const ok = validLabels.has(lab);
    return `<button type="button" class="cite${ok ? "" : " invalid"}" data-src="${lab}" title="${ok ? "Show source " + lab : "Cites a source that was not provided"}">${lab}</button>`;
  }).join(""));
}

function dimsHtml(j) {
  if (!j || !j.dimensions) return "";
  return `<div class="dims">${Object.entries(j.dimensions).map(([k, v]) => `<div class="dim"><span>${esc(k)}</span><strong>${v}</strong>/5</div>`).join("")}</div>`;
}

function checksHtml(c) {
  if (!c) return "";
  const items = Object.entries(c.checks).map(([k, v]) => `<span class="chip ${v ? "ok" : "miss"}">${v ? "✓" : "✗"} ${esc(k.replace(/_/g, " "))}</span>`);
  if (c.invented_numbers?.length) items.push(`<span class="chip miss">✗ invented numbers: ${esc(c.invented_numbers.join(", "))}</span>`);
  if (c.sql_keyword_leaks?.length) items.push(`<span class="chip miss">SQL keywords in text: ${esc(c.sql_keyword_leaks.join(", "))}</span>`);
  return items.length ? `<div class="row" style="margin-top:8px">${items.join("")}</div>` : `<span class="muted small">No applicable checks.</span>`;
}

function resultCard(mode, r) {
  const valid = new Set(((r.retrieval || {}).chunks || []).map((c) => c.source_label));
  const m = r.metrics || {};
  let body;
  if (r.abstained) {
    body = `<div class="error-box"><b>No explanation given.</b> ${esc(r.abstention_reason)}</div>
      ${r.draft_explanation ? `<details style="margin-top:8px"><summary>Show the model's withheld draft (not grounded)</summary><p class="small">${esc(r.draft_explanation)}</p></details>` : ""}`;
  } else {
    body = `<div class="explanation">${renderExplanationText(r.explanation || "", valid)}</div>`;
    if (r.grounding_status === "fallback_ungrounded") body = `<div class="provenance small" style="margin-bottom:8px">Fallback answer: written without sources (${esc(r.fallback_reason || "")}).</div>` + body;
  }
  const cr = r.citation_report;
  const citeLine = cr ? `<div class="metrics-row"><span>Citations: <b>${cr.markers.length}</b></span><span>valid: <b>${cr.citation_validity == null ? "n/a" : (cr.citation_validity * 100).toFixed(0) + "%"}</b></span><span>cited sentences: <b>${cr.n_cited_sentences}/${cr.n_sentences}</b></span><span title="Share of cited sentences whose content words appear in the cited passage (heuristic)">lexical support: <b>${cr.support_rate == null ? "n/a" : (cr.support_rate * 100).toFixed(0) + "%"}</b></span></div>` : "";
  const judge = r.judge ? `<div style="margin-top:12px"><div class="row"><h4>LLM judge</h4><span class="badge ${r.judge.grade === "good" ? "good" : "bad"}">${r.judge.grade === "good" ? "✓" : "✗"} ${esc(r.judge.grade)}</span><span class="muted small">unverified automated grade (${esc(r.judge.judge_model)})</span></div>
      ${dimsHtml(r.judge)}<p class="small" style="margin-top:6px">${esc(r.judge.reason)}</p>
      ${r.judge.unsupported_claims?.length ? `<p class="small"><b>Unsupported claims:</b> ${r.judge.unsupported_claims.map(esc).join("; ")}</p>` : ""}</div>` : "";
  const checks = r.checks ? `<div style="margin-top:10px"><h4>Deterministic checks</h4>${checksHtml(r.checks)}</div>` : "";
  return `<div class="card result-card ${mode}">
    <header><h3>${mode === "rag" ? "RAG explanation" : "Baseline explanation"}</h3>${statusBadge(r.grounding_status)}<span class="spacer"></span><span class="muted small">${esc(r.model)} · ${esc(r.prompt_file)} (${esc(r.prompt_hash)})</span></header>
    ${body}${citeLine}
    <div class="metrics-row"><span>retrieval <b>${fmt(m.retrieval_s, 2)} s</b></span><span>generation <b>${fmt(m.generation_s, 1)} s</b></span><span>total <b>${fmt(m.total_s, 1)} s</b></span><span>tokens in/out <b>${m.in_tok ?? 0}/${m.out_tok ?? 0}</b></span></div>
    ${judge}${checks}</div>`;
}

function renderAnalysis(retrieval, analysis) {
  const a = analysis || (retrieval && retrieval.analysis);
  if (!a) return;
  const required = new Set(retrieval?.required_features || []);
  const covered = new Set(retrieval?.covered_features || []);
  const chips = a.features.map((f) => {
    if (!required.has(f.key)) return `<span class="chip" title="Basic feature: no documentation required">${esc(f.label)}</span>`;
    const ok = covered.has(f.key);
    return `<span class="chip ${ok ? "ok" : "miss"}" title="${ok ? "Covered by a retrieved passage" : "No retrieved passage explains this feature"}">${ok ? "✓" : "✗"} ${esc(f.label)}</span>`;
  }).join(" ");
  const cov = retrieval ? `<p class="small" style="margin-top:8px">Evidence coverage: <b>${(retrieval.coverage * 100).toFixed(0)}%</b> of features that need documentation · ${retrieval.sufficient ? '<span class="badge good">✓ sufficient</span>' : '<span class="badge bad">⊘ insufficient</span>'}</p>` : "";
  const tables = a.tables?.length ? `<p class="small muted">Tables: ${a.tables.map(esc).join(", ")}${a.limit ? ` · LIMIT ${esc(a.limit)}` : ""}${a.order_by?.length ? ` · ORDER BY ${a.order_by.map((o) => esc(o.expr) + (o.desc ? " DESC" : " ASC")).join(", ")}` : ""}</p>` : "";
  const queries = retrieval?.queries ? `<details style="margin-top:6px"><summary>Retrieval queries built from the SQL (${retrieval.queries.length})</summary><ul class="small">${retrieval.queries.map((q) => `<li><b>${esc(q.feature || "whole query")}</b>: ${esc(q.text)}</li>`).join("")}</ul></details>` : "";
  $("#analysis-out").innerHTML = `<div class="row">${chips}</div>${tables}${cov}${queries}`;
}

function renderSources(retrieval) {
  const out = $("#sources-out");
  if (!retrieval || !retrieval.chunks?.length) {
    out.innerHTML = `<div class="muted small">No passages were selected.</div>`;
    $("#context-meta").textContent = "";
    return;
  }
  const tok = retrieval.chunks.reduce((s, c) => s + (c.n_tokens || 0), 0);
  $("#context-meta").textContent = `${retrieval.chunks.length} passages · ~${tok} tokens · ${retrieval.config.mode} · ${(retrieval.timings.total_s * 1000).toFixed(0)} ms`;
  out.innerHTML = retrieval.chunks.map((c) => `<article class="source" id="src-${c.source_label}">
      <header><span class="label">${c.source_label}</span><b>${esc(c.title)}</b><span class="muted small">› ${esc(c.section)}</span><span class="spacer"></span>
        ${c.source_url ? `<a class="small" href="${esc(c.source_url)}" target="_blank" rel="noopener">source</a>` : ""}</header>
      <div class="row small" style="margin-top:4px">${(c.covers || []).map((f) => `<span class="chip ok">covers ${esc(f)}</span>`).join("")}<span class="chip">${esc(c.doc_type)}</span><span class="chip">${esc(c.dialect)}</span></div>
      <div class="scores">dense sim ${fmt(c.dense_score, 3)} · BM25 ${fmt(c.bm25_score, 2)} · fused ${fmt(c.fused_score, 4)} · rank ${c.rank}</div>
      <div class="text">${esc(c.text)}</div></article>`).join("");
}

document.addEventListener("click", (e) => {
  const b = e.target.closest(".cite");
  if (!b) return;
  const card = document.getElementById(`src-${b.dataset.src}`);
  if (card) {
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1600);
  }
});

function startTimer(label) {
  const t0 = performance.now();
  const el = $("#explain-status");
  el.innerHTML = `<span class="spinner"></span> ${label} 0 s`;
  const id = setInterval(() => { el.innerHTML = `<span class="spinner"></span> ${label} ${((performance.now() - t0) / 1000).toFixed(0)} s`; }, 500);
  return () => { clearInterval(id); el.textContent = `done in ${((performance.now() - t0) / 1000).toFixed(1)} s`; };
}

$("#explain-btn").addEventListener("click", async () => {
  const btn = $("#explain-btn");
  btn.disabled = true;
  const stop = startTimer("generating");
  const out = $("#results-out");
  out.innerHTML = `<div class="empty"><span class="spinner"></span> Retrieving evidence and generating...</div>`;
  try {
    const data = await api("/api/explain", {
      method: "POST",
      body: JSON.stringify({ sql: $("#sql-input").value, schema: $("#schema-input").value, mode: state.mode, policy: $("#policy-select").value, judge: $("#judge-check").checked }),
    });
    const modes = Object.keys(data.results);
    out.innerHTML = modes.map((m) => resultCard(m, data.results[m])).join("");
    const rag = data.results.rag;
    renderAnalysis(rag?.retrieval, (rag || data.results.baseline).analysis);
    renderSources(rag?.retrieval);
    if (!rag) $("#sources-out").innerHTML = `<div class="muted small">Baseline mode does not retrieve documentation.</div>`;
  } catch (e) {
    out.innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
  } finally {
    stop();
    btn.disabled = false;
  }
});

$("#retrieve-btn").addEventListener("click", async () => {
  const stop = startTimer("retrieving");
  try {
    const r = await api("/api/retrieve", { method: "POST", body: JSON.stringify({ sql: $("#sql-input").value }) });
    renderAnalysis(r, r.analysis);
    renderSources(r);
  } catch (e) {
    $("#sources-out").innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
  } finally { stop(); }
});

/* ---------------------------------------------------------------- charts */
const tip = $("#tooltip");
function showTip(e, html) { tip.innerHTML = html; tip.classList.remove("hidden"); moveTip(e); }
function moveTip(e) {
  const pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tip.style.left = `${x}px`; tip.style.top = `${y}px`;
}
function hideTip() { tip.classList.add("hidden"); }

/**
 * Horizontal grouped bars: two series (baseline, RAG) per category.
 * rows: [{label, a, b, aText, bText, tip}] ; max: domain maximum.
 */
const charts = new Map();  // element id -> [rows, options]; re-rendered when visible or resized
function rerenderCharts() {
  for (const [id, [rows, opts]] of charts) {
    const el = document.getElementById(id);
    if (el && el.clientWidth) groupedBars(el, rows, opts);
  }
}
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(rerenderCharts, 150); });

function groupedBars(el, rows, opts = {}) {
  charts.set(el.id, [rows, opts]);
  if (!el.clientWidth) return;  // hidden tab: drawn by rerenderCharts() when it is shown
  const { max = 1, format = (v) => v, ticks = [0, 0.25, 0.5, 0.75, 1] } = opts;
  const narrow = el.clientWidth < 520;
  const labelWidth = narrow ? Math.min(opts.labelWidth || 190, 120) : (opts.labelWidth || 190);
  const barH = 9, gap = 2, groupGap = 14, padTop = 8, axisH = 20;
  const W = el.clientWidth;
  const plotW = W - labelWidth - (opts.rightPad || 56);
  const H = padTop + rows.length * (barH * 2 + gap + groupGap) + axisH;
  const x = (v) => labelWidth + (Math.max(0, Math.min(v ?? 0, max)) / max) * plotW;
  const rounded = (x0, y, w, h, cls) => {
    if (w <= 0.5) return `<rect class="${cls}" x="${x0}" y="${y}" width="1" height="${h}" />`;
    const r = Math.min(4, h / 2, w);
    return `<path class="${cls}" d="M${x0},${y} H${x0 + w - r} Q${x0 + w},${y} ${x0 + w},${y + r} V${y + h - r} Q${x0 + w},${y + h} ${x0 + w - r},${y + h} H${x0} Z" />`;
  };
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Grouped bar chart"><g class="grid">`;
  for (const t of ticks) svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${padTop - 4}" y2="${H - axisH}" />`;
  svg += `</g>`;
  rows.forEach((r, i) => {
    const y0 = padTop + i * (barH * 2 + gap + groupGap);
    svg += `<g class="grp" data-i="${i}"><rect class="row-bg" x="0" y="${y0 - 5}" width="${W}" height="${barH * 2 + gap + 10}" fill="transparent" />`;
    const maxChars = Math.floor(labelWidth / 7);
    svg += `<text class="cat-label" x="${labelWidth - 10}" y="${y0 + barH + 4}" text-anchor="end">${esc(r.label.length > maxChars ? r.label.slice(0, maxChars - 1) + "…" : r.label)}</text>`;
    svg += rounded(labelWidth, y0, x(r.a) - labelWidth, barH, "bar-baseline");
    svg += rounded(labelWidth, y0 + barH + gap, x(r.b) - labelWidth, barH, "bar-rag");
    svg += `<text class="value-label" x="${x(r.a) + 5}" y="${y0 + barH - 1}">${esc(r.aText ?? format(r.a))}</text>`;
    svg += `<text class="value-label" x="${x(r.b) + 5}" y="${y0 + barH * 2 + gap - 1}">${esc(r.bText ?? format(r.b))}</text></g>`;
  });
  for (const t of ticks) svg += `<text class="axis-label" x="${x(t)}" y="${H - 4}" text-anchor="middle">${esc(format(t))}</text>`;
  svg += `</svg>`;
  el.innerHTML = svg;
  $$("g.grp", el).forEach((g) => {
    const r = rows[Number(g.dataset.i)];
    g.addEventListener("mouseenter", (e) => { g.querySelector(".row-bg").setAttribute("fill", "var(--surface-2)"); showTip(e, r.tip); });
    g.addEventListener("mousemove", moveTip);
    g.addEventListener("mouseleave", () => { g.querySelector(".row-bg").setAttribute("fill", "transparent"); hideTip(); });
  });
}

function table(headers, rows, opts = {}) {
  const th = headers.map((h, i) => `<th class="${opts.num?.includes(i) ? "num" : ""}">${esc(h)}</th>`).join("");
  const tr = rows.map((r, ri) => `<tr class="${opts.best?.includes(ri) ? "best" : ""}">${r.map((c, i) => `<td class="${opts.num?.includes(i) ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

/* ------------------------------------------------------------- evaluation */
async function loadExperiments() {
  try {
    const data = await api("/api/experiments");
    state.experiments = data.experiments.filter((e) => Object.keys(e.summaries).length);
    const sel = $("#exp-select");
    sel.innerHTML = state.experiments.map((e) => `<option value="${esc(e.exp_id)}">${esc(e.exp_id)} (${esc(e.metadata?.dataset?.split || "?")}, n=${esc(e.metadata?.dataset?.n_items ?? "?")})</option>`).join("");
    $("#label-exp").innerHTML = sel.innerHTML;
    if (!state.experiments.length) {
      $("#eval-empty").classList.remove("hidden");
      $("#eval-body").classList.add("hidden");
    } else {
      const preferred = state.experiments.find((e) => e.exp_id === "rag_vs_baseline_dev")
        || state.experiments.find((e) => /dev/.test(e.exp_id)) || state.experiments[0];
      sel.value = preferred.exp_id;
      selectExperiment(preferred.exp_id);
    }
  } catch (e) {
    $("#eval-empty").textContent = `Could not load experiments: ${e.message}`;
    $("#eval-empty").classList.remove("hidden");
  }
  loadRetrievalEval();
  loadAgreement();
  loadHistorical();
}
$("#exp-select").addEventListener("change", (e) => selectExperiment(e.target.value));
$("#judge-select").addEventListener("change", () => renderSummary());

function selectExperiment(expId) {
  const exp = state.experiments.find((e) => e.exp_id === expId);
  state.currentExp = exp;
  const tags = Object.keys(exp.summaries);
  const preferredTag = tags.find((t) => /7b__rubric$/.test(t)) || tags.find((t) => /rubric$/.test(t)) || tags[0];
  $("#judge-select").innerHTML = tags.map((t) => `<option value="${esc(t)}" ${t === preferredTag ? "selected" : ""}>${esc(t)}</option>`).join("");
  const md = exp.metadata || {};
  $("#exp-meta").innerHTML = `Generator <b>${esc(md.generator?.model)}</b> · temp ${esc(md.generator?.options?.temperature)} · seed ${esc(md.seed)} · dataset sha256 ${esc((md.dataset?.sha256 || "").slice(0, 10))} · ${esc(md.created_at || "")}`;
  renderSummary();
  loadItems();
}

function currentSummary() {
  const exp = state.currentExp;
  return exp ? exp.summaries[$("#judge-select").value] : null;
}

function tile(k, a, b, s) {
  return `<div class="tile"><div class="k">${k}</div><div class="pair"><div><small><span class="legend-swatch" style="background:var(--series-1)"></span>Baseline</small><span class="v">${a}</span></div><div><small><span class="legend-swatch" style="background:var(--series-2)"></span>RAG</small><span class="v">${b}</span></div></div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
}

function renderSummary() {
  const s = currentSummary();
  if (!s) return;
  const B = s.conditions.baseline || {}, R = s.conditions.rag || {};
  const p = s.paired || {};
  const d = p.approval_delta_rag_minus_baseline || {};
  const delta = d.mean != null ? `${(d.mean * 100 >= 0 ? "+" : "")}${(d.mean * 100).toFixed(1)} pts (95% CI ${(d.ci_low * 100).toFixed(1)} to ${(d.ci_high * 100).toFixed(1)}), McNemar p=${p.mcnemar_exact_p == null ? "n/a" : p.mcnemar_exact_p.toFixed(3)}` : "";
  const refTiles = [];
  for (const [src, label] of [["human", "Human labels"], ["ai_reference", "Reference labels (AI-assisted, not human)"]]) {
    const rc = state.agreement?.reference_comparison?.[src]?.[state.currentExp?.exp_id];
    if (!rc || !rc.n_pairs) continue;
    const d = rc.delta_rag_minus_baseline || {};
    const r2 = (x) => ({ rate: x.rate, k: x.good, n: x.n });
    refTiles.push(tile(`${label}: good`, pct(r2(rc.conditions.baseline)), pct(r2(rc.conditions.rag)),
      `${(d.mean * 100 >= 0 ? "+" : "")}${(d.mean * 100).toFixed(1)} pts (95% CI ${(d.ci_low * 100).toFixed(1)} to ${(d.ci_high * 100).toFixed(1)}), McNemar p=${rc.mcnemar_exact_p == null ? "n/a" : rc.mcnemar_exact_p.toFixed(3)}`));
  }
  $("#kpi-tiles").innerHTML = [
    ...refTiles,
    tile("Judge approval (LLM, unverified)", pct(B.judge_approval), pct(R.judge_approval), delta),
    tile("Approval, abstentions count as fail", pct(B.judge_approval_itt), pct(R.judge_approval_itt), `RAG abstained on ${R.n_abstained ?? 0} of ${R.n_items ?? 0}`),
    tile("Explanations with unsupported claims", pct(B.explanations_with_unsupported_claims), pct(R.explanations_with_unsupported_claims), "as listed by the judge"),
    tile("Mean end-to-end latency", `${num(B.latency_s?.end_to_end, 1)} s`, `${num(R.latency_s?.end_to_end, 1)} s`, `input tokens ${num(B.tokens?.input, 0)} vs ${num(R.tokens?.input, 0)}`),
  ].join("");

  const dims = ["correctness", "completeness", "clarity", "groundedness"];
  const dimRows = dims.map((k) => {
    const a = B.dimensions?.[k]?.mean, b = R.dimensions?.[k]?.mean, dd = p.dimension_deltas?.[k];
    return { label: k, a, b, aText: fmt(a), bText: fmt(b), tip: `<div class="t">${k}</div>Baseline ${fmt(a)} · RAG ${fmt(b)}${dd ? `<br>Paired mean difference ${dd.mean >= 0 ? "+" : ""}${dd.mean.toFixed(2)} (95% CI ${dd.ci_low.toFixed(2)} to ${dd.ci_high.toFixed(2)}, n=${dd.n})` : ""}` };
  });
  if (dimRows.some((r) => r.a != null)) {
    groupedBars($("#chart-dims"), dimRows, { max: 5, ticks: [0, 1, 2, 3, 4, 5], format: (v) => String(v), labelWidth: 110 });
    $("#table-dims").innerHTML = table(["Dimension", "Baseline", "RAG", "Paired diff (95% CI)"], dims.map((k) => {
      const dd = p.dimension_deltas?.[k];
      return [k, fmt(B.dimensions?.[k]?.mean), fmt(R.dimensions?.[k]?.mean), dd ? `${dd.mean >= 0 ? "+" : ""}${dd.mean.toFixed(2)} (${dd.ci_low.toFixed(2)} to ${dd.ci_high.toFixed(2)})` : "n/a"];
    }), { num: [1, 2] });
  } else {
    $("#chart-dims").innerHTML = `<div class="muted small">This judge configuration has no rubric dimensions (legacy binary judge).</div>`;
    $("#table-dims").innerHTML = "";
  }

  const checkNames = Array.from(new Set([...Object.keys(B.deterministic_checks?.per_check_pass_rate || {}), ...Object.keys(R.deterministic_checks?.per_check_pass_rate || {})]));
  const crow = (name, a, b) => [esc(name), `${pct(a)} <span class="muted small">${frac(a)}</span>`, `${pct(b)} <span class="muted small">${frac(b)}</span>`];
  $("#table-checks").innerHTML = table(["Check", "Baseline", "RAG"], [
    [`<b>Mean coverage score</b>`, num(B.deterministic_checks?.coverage_score), num(R.deterministic_checks?.coverage_score)],
    ...checkNames.map((n) => crow(n.replace(/_/g, " "), B.deterministic_checks?.per_check_pass_rate?.[n], R.deterministic_checks?.per_check_pass_rate?.[n])),
    crow("invented numbers (lower is better)", B.deterministic_checks?.explanations_with_invented_numbers, R.deterministic_checks?.explanations_with_invented_numbers),
    crow("raw SQL keywords in text (lower is better)", B.deterministic_checks?.explanations_with_sql_keyword_leaks, R.deterministic_checks?.explanations_with_sql_keyword_leaks),
  ], { num: [1, 2] });

  const feats = Object.entries(s.by_feature || {});
  const featRows = feats.map(([f, e]) => ({
    label: f, a: e.baseline?.approval?.rate, b: e.rag?.approval?.rate,
    aText: `${pct(e.baseline?.approval)} (n=${e.baseline?.approval?.n ?? 0})`, bText: `${pct(e.rag?.approval)} (n=${e.rag?.approval?.n ?? 0})`,
    tip: `<div class="t">${esc(f)}</div>Baseline ${pct(e.baseline?.approval)} ${frac(e.baseline?.approval)}<br>RAG ${pct(e.rag?.approval)} ${frac(e.rag?.approval)}${e.rag?.abstained ? `<br>RAG abstained ${e.rag.abstained}` : ""}<br><span class="muted">small n: read as indicative only</span>`,
  }));
  groupedBars($("#chart-features"), featRows, { max: 1, format: (v) => `${Math.round(v * 100)}%`, labelWidth: 230, rightPad: 110 });
  $("#table-features").innerHTML = table(["Feature", "n", "Baseline approval", "RAG approval", "Baseline checks", "RAG checks"],
    feats.map(([f, e]) => [esc(f), e.baseline?.n ?? 0, `${pct(e.baseline?.approval)} ${frac(e.baseline?.approval)}`, `${pct(e.rag?.approval)} ${frac(e.rag?.approval)}`, fmt(e.baseline?.check_coverage), fmt(e.rag?.check_coverage)]), { num: [1, 4, 5] });

  const lat = (c, k) => num(c.latency_s?.[k], k === "retrieval" ? 2 : 1);
  $("#table-perf").innerHTML = table(["Metric (mean)", "Baseline", "RAG"], [
    ["Retrieval time (s)", lat(B, "retrieval"), lat(R, "retrieval")],
    ["Generation time (s)", lat(B, "generation"), lat(R, "generation")],
    ["End-to-end (s)", lat(B, "end_to_end"), lat(R, "end_to_end")],
    ["Judge time (s)", lat(B, "judge"), lat(R, "judge")],
    ["Input tokens", num(B.tokens?.input, 0), num(R.tokens?.input, 0)],
    ["Output tokens", num(B.tokens?.output, 0), num(R.tokens?.output, 0)],
    ["Retrieved context tokens", "n/a", num(R.tokens?.context, 0)],
    ["Explanation length (words)", num(B.explanation_words, 0), num(R.explanation_words, 0)],
    ["Generation errors", B.generation_errors ?? 0, R.generation_errors ?? 0],
  ], { num: [1, 2] });

  const g = R.grounding || {}, rr = R.retrieval || {};
  $("#table-grounding").innerHTML = table(["Metric", "Value"], [
    ["Answers with at least one citation", `${pct(g.answers_with_any_citation)} ${frac(g.answers_with_any_citation)}`],
    ["Citation validity (cited source was provided)", num(g.citation_validity, 3)],
    ["Answers citing a source that was not provided", `${pct(g.answers_with_invalid_citation)} ${frac(g.answers_with_invalid_citation)}`],
    ["Share of sentences carrying a citation", num(g.cited_sentence_share, 3)],
    ["Lexical support of cited sentences (heuristic)", num(g.lexical_claim_support_rate, 3)],
    ["Evidence sufficient before generation", `${pct(rr.evidence_sufficient)} ${frac(rr.evidence_sufficient)}`],
    ["Abstentions", `${pct(R.abstention_rate)} ${frac(R.abstention_rate)}`],
    ["Grounding status counts", esc(JSON.stringify(R.grounding_status || {}))],
  ], { num: [1] });
  updateReportLinks();
}

async function loadRetrievalEval() {
  try {
    const r = await api("/api/retrieval-eval");
    if (!r.available) { $("#table-retrieval").innerHTML = `<div class="muted small">Run <code>python -m sqlrag retrieval-eval</code> to populate this table.</div>`; return; }
    $("#retrieval-prov").textContent = `${r.n_items} SQL items · relevance map: ${r.qrels_provenance}`;
    const rows = r.runs.map((x) => [x.chunk_size, esc(x.strategy), esc(x.mode), fmt(x.mrr, 3), fmt(x.at_k["1"]?.feature_recall, 3), fmt(x.at_k["5"]?.feature_recall, 3), fmt(x.at_k["5"]?.precision, 3), fmt(x.at_k["5"]?.ndcg, 3), fmt(x.context_feature_recall, 3), fmt(x.latency_s.mean * 1000, 0), x.n_chunks]);
    const bestIdx = r.runs.reduce((bi, x, i, arr) => (x.context_feature_recall > arr[bi].context_feature_recall ? i : bi), 0);
    $("#table-retrieval").innerHTML = table(["Chunk size", "Query strategy", "Mode", "MRR", "Recall@1", "Recall@5", "Precision@5", "nDCG@5", "Context recall", "ms / query", "Chunks"], rows, { num: [0, 3, 4, 5, 6, 7, 8, 9, 10], best: [bestIdx] })
      + `<p class="muted small" style="margin-top:6px">Recall@k = share of the query's SQL features with a relevant passage in the top k. Context recall = the same for the passages actually given to the generator. Highlighted row: best context recall.</p>`;
  } catch (e) { $("#table-retrieval").textContent = e.message; }
}

function confusion(m) {
  const [[tp, fn], [fp, tn]] = m.confusion_matrix.rows_true_cols_pred;
  return `<table style="width:auto"><thead><tr><th></th><th>judge: good</th><th>judge: bad</th></tr></thead><tbody><tr><th>reference: good</th><td class="num">${tp}</td><td class="num">${fn}</td></tr><tr><th>reference: bad</th><td class="num">${fp}</td><td class="num">${tn}</td></tr></tbody></table>`;
}

async function loadAgreement() {
  try {
    const a = await api("/api/agreement");
    state.agreement = a;
    renderSummary();
    const names = { human: "Genuine human labels", ai_reference: "AI-assisted reference labels (not human)" };
    let html = "";
    for (const [src, rep] of Object.entries(a.sources)) {
      html += `<h4 style="margin-top:10px">${names[src] || src}: ${rep.n_labels} labels</h4>`;
      if (!rep.n_labels) { html += `<p class="muted small">${src === "human" ? "No human labels yet. Use the Human labelling tab." : "None."}</p>`; continue; }
      const rows = Object.entries(rep.judges).map(([tag, m]) => [esc(tag), m.n, fmt(m.accuracy, 3), fmt(m.precision_good, 3), fmt(m.recall_good, 3), fmt(m.f1_good, 3), fmt(m.recall_bad, 3), fmt(m.cohen_kappa, 3), `${fmt(m.reference_good_rate * 100, 0)}% / ${fmt(m.judge_good_rate * 100, 0)}%`]);
      html += table(["Experiment / judge", "n", "Accuracy", "Precision (good)", "Recall (good)", "F1 (good)", "Recall (bad)", "Cohen's κ", "Good rate ref / judge"], rows, { num: [1, 2, 3, 4, 5, 6, 7] });
      const first = Object.entries(rep.judges)[0];
      if (first) html += `<details style="margin-top:6px"><summary>Confusion matrices</summary>${Object.entries(rep.judges).map(([t, m]) => `<p class="small"><b>${esc(t)}</b></p>${confusion(m)}`).join("")}</details>`;
    }
    $("#agreement-out").innerHTML = html;
  } catch (e) { $("#agreement-out").textContent = e.message; }
}

async function loadHistorical() {
  try {
    const h = await api("/api/historical");
    const rows = h.runs.filter((r) => r.kind.startsWith("historical")).map((r) => [esc(r.file), r.n, r.good, `${((r.good / r.n) * 100).toFixed(1)}%`, r.judge_saw_schema ? "yes" : "no"]);
    $("#table-historical").innerHTML = table(["Results file", "Items", "Judged good", "Approval", "Judge saw schema"], rows, { num: [1, 2, 3] });
  } catch (e) { $("#table-historical").textContent = e.message; }
}

async function loadItems() {
  const exp = state.currentExp;
  if (!exp) return;
  try {
    const data = await api(`/api/experiments/${encodeURIComponent(exp.exp_id)}/items?judge=${encodeURIComponent($("#judge-select").value)}`);
    state.items = data.items;
    renderItemList();
  } catch (e) { $("#item-list").innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
}
$("#judge-select").addEventListener("change", loadItems);
$("#item-filter").addEventListener("change", renderItemList);
$("#item-search").addEventListener("input", renderItemList);

function itemOutcome(it) {
  const b = it.baseline?.judge?.score, r = it.rag?.judge?.score;
  return { b, r };
}

function renderItemList() {
  const f = $("#item-filter").value, q = $("#item-search").value.trim().toLowerCase();
  const items = state.items.filter((it) => {
    const { b, r } = itemOutcome(it);
    if (q && !(it.id.toLowerCase().includes(q) || it.sql.toLowerCase().includes(q))) return false;
    if (f === "disagree") return b != null && r != null && b !== r;
    if (f === "only_rag") return b === 0 && r === 1;
    if (f === "only_base") return b === 1 && r === 0;
    if (f === "both_bad") return b === 0 && r === 0;
    if (f === "abstained") return it.rag?.abstained;
    return true;
  });
  const mark = (v) => (v == null ? "–" : v ? "✓" : "✗");
  $("#item-list").innerHTML = items.length ? items.map((it) => {
    const { b, r } = itemOutcome(it);
    return `<button type="button" data-id="${esc(it.id)}" aria-current="${state.selectedItem === it.id}"><b>${esc(it.id)}</b> <span class="meta">B ${mark(b)} · R ${it.rag?.abstained ? "⊘" : mark(r)}</span><div class="meta">${esc(it.feature || "")}</div></button>`;
  }).join("") : `<div class="muted small" style="padding:10px">No items match.</div>`;
  $$("#item-list button").forEach((btn) => btn.addEventListener("click", () => { state.selectedItem = btn.dataset.id; renderItemList(); renderItemDetail(); }));
}

function conditionPanel(name, c) {
  if (!c) return `<div class="card"><div class="muted">No ${name} result.</div></div>`;
  const valid = new Set(((c.retrieval || {}).chunks || []).map((x) => x.source_label));
  const j = c.judge;
  const text = c.abstained ? `<div class="error-box">${esc(c.abstention_reason)}</div>` : `<div class="explanation small">${renderExplanationText(c.explanation || "", valid)}</div>`;
  return `<div class="card result-card ${name}"><header><h3>${name === "rag" ? "RAG" : "Baseline"}</h3>${statusBadge(c.grounding_status)}${j ? `<span class="badge ${j.grade === "good" ? "good" : "bad"}">${j.grade === "good" ? "✓" : "✗"} judge: ${esc(j.grade)}</span>` : ""}</header>
    ${text}${j ? `${dimsHtml(j)}<p class="small" style="margin-top:6px"><b>Judge:</b> ${esc(j.reason)}</p>${j.unsupported_claims?.length ? `<p class="small"><b>Unsupported:</b> ${j.unsupported_claims.map(esc).join("; ")}</p>` : ""}` : ""}
    ${checksHtml(c.checks)}
    ${c.retrieval?.chunks?.length ? `<details style="margin-top:8px"><summary>Sources given to the model (${c.retrieval.chunks.length})</summary>${c.retrieval.chunks.map((s) => `<p class="small"><b>${s.source_label}</b> ${esc(s.title)} › ${esc(s.section)} <span class="muted">(dense ${fmt(s.dense_score, 3)})</span></p>`).join("")}</details>` : ""}</div>`;
}

function renderItemDetail() {
  const it = state.items.find((x) => x.id === state.selectedItem);
  if (!it) return;
  $("#item-detail").innerHTML = `<div class="stack"><div><b>${esc(it.id)}</b> <span class="muted small">${esc(it.feature)} · ${esc(it.domain)}</span><pre class="sql">${esc(it.sql)}</pre></div><div class="side-by-side">${conditionPanel("baseline", it.baseline)}${conditionPanel("rag", it.rag)}</div></div>`;
}

/* --------------------------------------------------------- knowledge base */
async function loadKnowledge() {
  try {
    const d = await api("/api/documents");
    const s = d.stats;
    const t = (k, v, sub = "") => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div>${sub ? `<div class="s">${sub}</div>` : ""}</div>`;
    $("#kb-tiles").innerHTML = [
      t("Documents indexed", s.documents_indexed, `${s.documents_failed} failed · ${s.documents_duplicate} duplicates`),
      t("Chunks", s.chunks, `avg ${s.avg_chunk_tokens} tokens · max ${s.max_chunk_tokens}`),
      t("Embedding model", `<span style="font-size:15px">${esc(s.embedding_model)}</span>`, `${s.embedding_dimension} dimensions · window ${s.embedding_max_seq_length ?? "n/a"} tokens`),
      t("Chunking", `${s.chunking.chunk_size}/${s.chunking.overlap}`, `size/overlap · ${s.chunks_over_embedding_window} chunks exceed the embedding window`),
    ].join("");
    const indexed = new Map(d.documents.map((x) => [x.doc_id, x]));
    const rows = d.manifest.map((m) => {
      const r = indexed.get(m.doc_id) || {};
      const st = m.status === "indexed" ? `<span class="badge good">✓ indexed</span>` : m.status === "failed" ? `<span class="badge bad">✗ failed</span>` : `<span class="badge">not indexed</span>`;
      const act = m.status === "indexed"
        ? `<button class="btn small" data-chunks="${esc(m.doc_id)}">View chunks</button>`
        : `<button class="btn small" data-ingest="${esc(m.doc_id)}">Ingest</button>`;
      return [`<a href="${esc(m.url)}" target="_blank" rel="noopener">${esc(m.title)}</a>${m.note ? `<div class="muted small">${esc(m.note)}</div>` : ""}`, esc(m.doc_type), esc(m.license), r.n_chunks ?? "", r.n_tokens ?? "", st, act];
    });
    d.documents.filter((x) => x.doc_type === "user_upload").forEach((r) => rows.push([`${esc(r.title || r.doc_id)} <span class="chip">upload</span>`, "user_upload", esc(r.license), r.n_chunks ?? "", r.n_tokens ?? "",
      r.status === "indexed" ? `<span class="badge good">✓ indexed</span>` : `<span class="badge">${esc(r.status)}</span>`,
      `<button class="btn small" data-chunks="${esc(r.doc_id)}">View chunks</button> <button class="btn small bad" data-remove="${esc(r.doc_id)}">Remove</button>`]));
    $("#kb-docs").innerHTML = table(["Document", "Type", "License", "Chunks", "Tokens", "Status", ""], rows, { num: [3, 4] });
    $$("#kb-docs [data-chunks]").forEach((b) => b.addEventListener("click", () => showChunks(b.dataset.chunks)));
    $$("#kb-docs [data-ingest]").forEach((b) => b.addEventListener("click", () => ingestManifestDoc(b.dataset.ingest, b)));
    $$("#kb-docs [data-remove]").forEach((b) => b.addEventListener("click", () => removeDoc(b.dataset.remove)));
  } catch (e) { $("#kb-docs").innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
}
$("#kb-refresh").addEventListener("click", loadKnowledge);

function adminHeaders() {
  const t = $("#admin-token").value.trim();
  return t ? { "X-Admin-Token": t } : {};
}

async function ingestManifestDoc(docId, btn) {
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>`;
  try {
    await api("/api/ingest/manifest", { method: "POST", headers: adminHeaders(), body: JSON.stringify({ doc_ids: [docId] }) });
    loadKnowledge(); loadHealth();
  } catch (e) { alert(e.message); btn.disabled = false; btn.textContent = "Ingest"; }
}

async function removeDoc(docId) {
  if (!confirm(`Remove ${docId} and its chunks from the index?`)) return;
  try { await api(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE", headers: adminHeaders() }); loadKnowledge(); }
  catch (e) { alert(e.message); }
}

async function showChunks(docId) {
  const dlg = $("#chunk-dialog");
  $("#dlg-title").textContent = docId;
  $("#dlg-body").innerHTML = `<span class="spinner"></span>`;
  dlg.showModal();
  try {
    const d = await api(`/api/documents/${encodeURIComponent(docId)}/chunks`);
    $("#dlg-body").innerHTML = d.chunks.map((c) => `<div class="source" style="margin-bottom:8px"><header><span class="label">#${c.chunk_index}</span><b>${esc(c.section)}</b><span class="spacer"></span><span class="muted small">${c.n_tokens} tokens${c.page_start > 0 ? ` · p.${c.page_start}` : ""}</span></header><div class="text">${esc(c.text)}</div></div>`).join("");
  } catch (e) { $("#dlg-body").innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
}

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const status = $("#upload-status");
  status.innerHTML = `<span class="spinner"></span> parsing, chunking and embedding...`;
  try {
    const res = await fetch("/api/ingest", { method: "POST", body: fd, headers: adminHeaders() });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    const r = data.report;
    status.textContent = r.unchanged.length ? "Unchanged: this exact document is already indexed." : r.duplicates.length ? `Duplicate of ${r.duplicates[0].duplicate_of}; not indexed again.` : `Indexed ${r.chunks_written} chunks in ${r.seconds}s.`;
    loadKnowledge(); loadHealth();
  } catch (err) { status.textContent = err.message; }
});

$("#kb-search").addEventListener("click", async () => {
  const out = $("#kb-results");
  out.innerHTML = `<span class="spinner"></span>`;
  try {
    const d = await api("/api/retrieve", { method: "POST", body: JSON.stringify({ query: $("#kb-query").value, mode: $("#kb-mode").value, top_k: 5 }) });
    out.innerHTML = d.chunks.map((c) => `<div class="source"><header><span class="label">#${c.rank}</span><b>${esc(c.title)}</b><span class="muted small">› ${esc(c.section)}</span></header><div class="scores">dense ${fmt(c.dense_score, 3)} · BM25 ${fmt(c.bm25_score, 2)} · fused ${fmt(c.fused_score, 4)}</div><div class="text">${esc(c.text.slice(0, 700))}${c.text.length > 700 ? "…" : ""}</div></div>`).join("") || `<div class="muted">No results.</div>`;
  } catch (e) { out.innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
});

/* ---------------------------------------------------------------- reports */
async function loadReports() {
  try {
    const d = await api("/api/reports");
    $("#reports-list").innerHTML = table(["Report", "Folder", "Size", "Modified"], d.reports.map((r) => [`<a href="/api/reports/${esc(r.folder)}/${esc(r.name)}" target="_blank" rel="noopener">${esc(r.name)}</a>`, esc(r.folder), `${(r.bytes / 1024).toFixed(0)} KB`, esc(r.modified)]), { num: [2] });
  } catch (e) { $("#reports-list").textContent = e.message; }
  if (!state.experiments.length) await loadExperiments();
  updateReportLinks();
}
$("#reports-refresh").addEventListener("click", loadReports);

function updateReportLinks() {
  const exp = state.currentExp, tag = $("#judge-select").value;
  if (!exp || !tag) return;
  $("#dl-csv").href = `/api/experiments/${encodeURIComponent(exp.exp_id)}/export?judge=${encodeURIComponent(tag)}&fmt=csv`;
  $("#dl-json").href = `/api/experiments/${encodeURIComponent(exp.exp_id)}/export?judge=${encodeURIComponent(tag)}&fmt=json`;
}

$("#build-pdf").addEventListener("click", async () => {
  const exp = state.currentExp, tag = $("#judge-select").value;
  if (!exp) { $("#pdf-status").textContent = "Select an experiment on the Evaluation tab first."; return; }
  $("#pdf-status").innerHTML = `<span class="spinner"></span> building...`;
  try {
    const res = await fetch(`/api/experiments/${encodeURIComponent(exp.exp_id)}/report.pdf?judge=${encodeURIComponent(tag)}`);
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `rag_comparison_${exp.exp_id}__${tag}.pdf`;
    a.click();
    $("#pdf-status").textContent = "Downloaded.";
    loadReports();
  } catch (e) { $("#pdf-status").textContent = e.message; }
});

/* --------------------------------------------------------------- labelling */
$("#label-start").addEventListener("click", nextLabel);
async function nextLabel() {
  const annotator = $("#annotator").value.trim();
  const exp = $("#label-exp").value;
  if (annotator.length < 2) { $("#label-progress").textContent = "Enter your name or initials first."; return; }
  if (!exp) { $("#label-progress").textContent = "No experiment available."; return; }
  try { localStorage.setItem("annotator", annotator); } catch { /* ignore */ }
  const card = $("#label-card");
  card.innerHTML = `<span class="spinner"></span>`;
  try {
    const d = await api(`/api/labels/next?exp_id=${encodeURIComponent(exp)}&annotator=${encodeURIComponent(annotator)}`);
    if (d.done) { card.innerHTML = `<div class="empty">All explanations of this experiment are labelled by ${esc(annotator)} (${d.labelled}).</div>`; return; }
    state.label = d;
    $("#label-progress").textContent = `${d.labelled} labelled · ${d.remaining} remaining`;
    card.innerHTML = `<div class="stack"><div><h4>Schema</h4><pre class="sql">${esc(d.schema)}</pre></div><div><h4>SQL</h4><pre class="sql">${esc(d.sql)}</pre></div>
      <div><h4>Explanation</h4><p class="explanation">${esc(d.explanation)}</p></div>
      <label class="field">Notes (optional) <input type="text" id="label-notes" placeholder="e.g. says students but COUNT counts courses" /></label>
      <div class="row"><button class="btn good" type="button" data-label="good">✓ Good: correct, complete and clear</button><button class="btn bad" type="button" data-label="bad">✗ Bad</button><button class="btn" type="button" id="label-skip">Skip</button></div></div>`;
    $$("#label-card [data-label]").forEach((b) => b.addEventListener("click", () => submitLabel(b.dataset.label)));
    $("#label-skip").addEventListener("click", nextLabel);
  } catch (e) { card.innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
}
async function submitLabel(label) {
  const d = state.label;
  try {
    await api("/api/labels", { method: "POST", headers: adminHeaders(), body: JSON.stringify({ exp_id: d.exp_id, item_id: d.item_id, condition_token: d.condition_token, explanation_sha256_12: d.explanation_sha256_12, label, annotator: $("#annotator").value.trim(), notes: $("#label-notes").value }) });
    nextLabel();
  } catch (e) { alert(e.message); }
}

/* -------------------------------------------------------------------- init */
(function init() {
  try { const a = localStorage.getItem("annotator"); if (a) $("#annotator").value = a; } catch { /* ignore */ }
  loadHealth();
  loadExamples();
  loadExperiments();
  let tab = null;
  try { tab = localStorage.getItem("tab"); } catch { /* ignore */ }
  if (tab && $(`#tab-${tab}`)) showTab(tab);
  setInterval(loadHealth, 30000);
})();
