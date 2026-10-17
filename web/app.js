/* HonestPrice UI. All inference runs in the browser via model.js (verified against Python in tests/parity.test.js). */
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = ["--c0", "--c1", "--c2", "--c3", "--c4"];
const col = i => css(COL[i % COL.length]);
let M, charts = {}, state = { car: null, pathNodes: new Set(), actual: null, level: "0.9" }, rendered = {};

const inr = v => v >= 1e7 ? `₹${(v / 1e7).toFixed(2)} crore` : v >= 1e5 ? `₹${(v / 1e5).toFixed(2)} lakh` : `₹${Math.round(v / 1e3)}k`;
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function mk(id, cfg) {
  if (charts[id]) charts[id].destroy();
  Chart.defaults.color = css("--mut"); Chart.defaults.borderColor = css("--line");
  cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false }, cfg.options || {});
  charts[id] = new Chart($(id), cfg);
}
const cv = (id, tall) => `<div class="cv ${tall ? "tall" : ""}"><canvas id="${id}"></canvas></div>`;
const card = (t, b, c = "") => `<div class="card ${c}"><h3>${t}</h3>${b}</div>`;

/* ---------------- valuation ---------------- */
function readCar() {
  const g = id => $(id).value;
  return { brand: g("brand"), model: g("model"), name: g("variant") || null, age: Math.max(0, M.meta.ref_year - (+g("year") || M.meta.ref_year)),
    km: Math.max(0, +g("km") || 0), fuel: g("fuel"), transmission: g("trans"), seller: g("seller"), owner: g("owner") };
}
function fillModels() {
  const b = $("brand").value, ms = Object.keys(M.catalog[b] || {});
  $("model").innerHTML = ms.map(m => `<option>${esc(m)}</option>`).join("");
  fillVariants();
}
function fillVariants() {
  const names = (M.catalog[$("brand").value] || {})[$("model").value] || [];
  $("variant").innerHTML = `<option value="">Not listed / any</option>` + names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
}
function update(keepActual) {
  if (!keepActual) state.actual = null;
  const car = readCar(); state.car = car;
  const x = HP.features(M, car);
  const cls = HP.classify(M, x), reg = HP.regress(M, x), staged = HP.gbmStaged(M, x);
  state.pathNodes = new Set(HP.pathOf(M.models.classification.cart, x));
  rendered.trees = false; if ($("trees").classList.contains("on")) renderTrees();
  renderResult(car, x, cls, reg, staged);
}
function renderResult(car, x, cls, reg, staged) {
  const T = M.models.tiers, C = M.models.classification, R = M.models.regression;
  const lv = state.level, [lo, hi] = reg.intervals[lv].map(Math.exp), mid = Math.exp(reg.mean);
  const dom = [Math.exp(reg.intervals["0.95"][0] - 0.15), Math.exp(reg.intervals["0.95"][1] + 0.15)];
  const pos = v => Math.min(100, Math.max(0, (Math.log(v) - Math.log(dom[0])) / (Math.log(dom[1]) - Math.log(dom[0])) * 100));
  const stack = cls["Stacking (softmax)"], tier = HP.argmax(stack);
  const votes = C.names.map(n => HP.argmax(cls[n])), unan = votes.every(v => v === votes[0]);
  const uncertain = !unan || Math.max(...stack) < 0.55;
  const tierLbl = ["Budget", "Mid-range", "Premium"];
  const stackBar = p => `<div class="stack">${p.map((v, i) => v > 0.02 ? `<span style="width:${v * 100}%;background:${col(i === 0 ? 1 : i === 1 ? 2 : 3)}">${i === 0 ? "B" : i === 1 ? "M" : "P"} ${Math.round(v * 100)}%</span>` : "").join("")}</div>`;
  const levels = ["0.8", "0.9", "0.95"];
  const members = R.names.map((n, i) => ({ n, p: Math.exp(reg.members[n]), w: R.weights.final[i] }));
  const actual = state.actual;
  $("result").innerHTML = `
  <div class="card">
    <div class="hero"><div><div class="note" style="margin:0">Estimated price</div><div class="price">${inr(mid)}</div></div>
      <div><div class="note" style="margin:0">${Math.round(lv * 100)}% interval</div><div style="font-size:22px;font-weight:600">${inr(lo)} &ndash; ${inr(hi)}</div></div></div>
    <div class="chips" role="group" aria-label="Interval level">${levels.map(l => `<button class="chip ${l === lv ? "on" : ""}" data-lv="${l}">${Math.round(l * 100)}%</button>`).join("")}</div>
    <div class="rangebar"><div class="iv" style="left:${pos(lo)}%;width:${pos(hi) - pos(lo)}%"></div><div class="pt" style="left:${pos(mid)}%"></div>${actual ? `<div class="act" style="left:${pos(actual)}%" title="actual listing price"></div>` : ""}</div>
    <div class="ticks"><span>${inr(dom[0])}</span><span>${actual ? `<span style="color:var(--acc2)">orange line = actual listing ${inr(actual)}${actual >= lo && actual <= hi ? " (inside interval)" : " (outside interval)"}</span>` : ""}</span><span>${inr(dom[1])}</span></div>
    <p class="note">Intervals use <b>normalised conformal prediction</b>: the width adapts to how much the models disagree about this car, and the multiplier was calibrated on ${M.regression.intervals.n_cal} held-out cars. On ${M.meta.split.test} unseen test cars the 90% interval contained the true price ${pct(M.regression.intervals.methods["normalised conformal (HonestPrice)"][4].coverage)} of the time. The interval is wide on purpose: listing prices are noisy.</p>
  </div>
  <div class="grid">
    ${card("Price tier council", `<p style="margin:0 0 8px"><span class="badge ${uncertain ? "warn" : "ok"}">${uncertain ? "Council split - treat the tier as uncertain" : "All members agree"}</span> Stacked verdict: <b>${tierLbl[tier]}</b> (${T.cuts_inr.map(inr).join(" / ")} cut-offs)</p>
      <table><tr><th>Member</th><th>Vote</th><th>Probability</th></tr>
      ${C.names.map(n => `<tr><td>${n}</td><td>${tierLbl[HP.argmax(cls[n])]}</td><td>${stackBar(cls[n])}</td></tr>`).join("")}
      <tr class="best"><td>Stacking</td><td>${tierLbl[tier]}</td><td>${stackBar(stack)}</td></tr></table>
      <p class="note">B = Budget, M = Mid-range, P = Premium. Members: pruned CART, bagging, random forest, AdaBoost.</p>`)}
    ${card("Who contributed to the price", `<table><tr><th>Model</th><th class="n">Estimate</th><th class="n">BMA weight</th></tr>
      ${members.map(m => `<tr><td>${m.n}</td><td class="n">${inr(m.p)}</td><td class="n">${pct(m.w, 0)}</td></tr>`).join("")}
      <tr class="best"><td>Weighted (BMA)</td><td class="n">${inr(mid)}</td><td class="n">100%</td></tr></table>
      <p class="note">Models disagree by &plusmn;${pct(Math.exp(reg.between) - 1, 0)} (weighted spread), forest trees by &plusmn;${pct(Math.exp(reg.rfStd) - 1, 0)}. Both feed the interval width.</p>`)}
    ${card("Gradient boosting builds the price stage by stage", cv("stagedChart") + `<p class="note">The additive model starts from the average log-price and each small tree adds a correction. Final value after ${R.gbm.stages} stages.</p>`)}
    ${card("Why the single CART says " + tierLbl[HP.argmax(cls["CART (pruned)"])], `<ol class="rules">${cartRules(x).map(r => `<li>${r}</li>`).join("")}</ol><p class="note">The exact path through the pruned classification tree; see the Tree explorer tab.</p>`)}
  </div>`;
  document.querySelectorAll(".chip").forEach(b => b.onclick = () => { state.level = b.dataset.lv; update(true); });
  mk("stagedChart", { type: "line", data: { labels: staged.map((_, i) => i), datasets: [{ data: staged.map(v => Math.exp(v) / 1e5), borderColor: col(1), pointRadius: 0, borderWidth: 2 }] },
    options: { plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: "Boosting stage" } }, y: { title: { display: true, text: "Estimate (₹ lakh)" } } } } });
}

/* ---------------- tree helpers ---------------- */
function cond(f, t, left) {
  const n = M.models.features[f];
  const yn = { fuel_diesel: "Diesel", fuel_petrol: "Petrol", seller_dealer: "Dealer seller", seller_trustmark: "Trustmark Dealer", automatic: "Automatic" };
  if (yn[n] || n.startsWith("brand_")) { const label = yn[n] || n.slice(6); return left ? `not ${label}` : `is ${label}`; }
  const op = left ? "≤" : ">";
  if (n === "log_km") return `odometer ${op} ${Math.round(Math.expm1(t)).toLocaleString()} km`;
  if (n === "age") return `age ${op} ${t.toFixed(1)} yrs`;
  if (n === "model_price_index") return `model's typical price ${op} ${inr(Math.exp(t))}`;
  if (n === "variant_price_index") return `variant's typical price ${op} ${inr(Math.exp(t))}`;
  if (n === "engine_litres") return `engine ${op} ${t.toFixed(2)} L`;
  if (n === "owner_rank") return left ? `owner count ≤ ${Math.floor(t) + 1}` : `owner count > ${Math.floor(t) + 1}`;
  return `${n} ${op} ${t.toFixed(2)}`;
}
function cartRules(x) {
  const t = M.models.classification.cart, p = HP.pathOf(t, x), out = [];
  for (let i = 0; i < p.length - 1; i++) { const n = p[i], left = p[i + 1] === t.l[n]; out.push(cond(t.f[n], t.t[n], left)); }
  const v = t.v[p[p.length - 1]], names = ["Budget", "Mid-range", "Premium"];
  out.push(`<b>${names[HP.argmax(v)]}</b> (${pct(Math.max(...v), 0)} of ${t.n[p[p.length - 1]]} training cars in this leaf)`);
  return out;
}

/* ---------------- boosting lab ---------------- */
function renderBoost() {
  const V = M.ada_view, keys = Object.keys(V.snapshots).map(Number).sort((a, b) => a - b), A = M.classification.ada;
  $("boost").innerHTML = `<h2>Boosting lab</h2>
  <p class="lead">AdaBoost re-weights the training cars after every round so the next tree concentrates on the ones the committee gets wrong. Gradient boosting instead fits each new tree to the remaining error, building an additive model.</p>
  <div class="grid">
    ${card("AdaBoost: sample weights after round <b id=\"rnd\"></b>", `<input type="range" id="rndSlider" min="0" max="${keys.length - 1}" value="0" aria-label="Boosting round">${cv("adaChart", true)}
      <p class="note">Each dot is a training car (age vs odometer). Bigger dots carry more weight. Colour = true price tier.</p>`)}
    ${card("AdaBoost: weighted error and tree weight (alpha) per round", cv("alphaChart") + `<p class="note">${A.n} rounds of depth-3 CART. Alpha = ln((1-err)/err) + ln(K-1) for K=3 classes (SAMME). Later trees see harder problems, so their error rises toward chance.</p>`)}
    ${card("Gradient boosting: error vs number of trees", cv("gbmChart") + `<p class="note">Validation error is minimal at <b>${M.regression.gbm_curve.best}</b> stages (early stopping); later trees only fit noise.</p>`)}
    ${card("Bagging vs random forest: out-of-bag error", cv("oobChart") + `<p class="note">OOB error is an honest estimate from bootstrap leftovers. Classification (left axis, error rate) and regression (RMSE of log price) shown together.</p>`)}
  </div>`;
  const tierCol = [col(1), col(2), col(3)];
  const draw = () => {
    const k = keys[+$("rndSlider").value]; $("rnd").textContent = k;
    mk("adaChart", { type: "scatter", data: { datasets: [0, 1, 2].map(t => ({ label: ["Budget", "Mid-range", "Premium"][t],
      data: V.age.map((a, i) => V.tier[i] === t ? { x: a, y: Math.expm1(V.log_km[i]) / 1000, r: 2 + V.snapshots[k][i] * 11 } : null).filter(Boolean), backgroundColor: tierCol[t] + "99", pointRadius: ctx => ctx.raw.r })) },
      options: { animation: false, scales: { x: { title: { display: true, text: "Age (years)" } }, y: { title: { display: true, text: "Odometer (thousand km)" }, max: 400 } } } });
  };
  $("rndSlider").oninput = draw; draw();
  mk("alphaChart", { data: { labels: A.errors.map((_, i) => i + 1), datasets: [
    { type: "line", label: "Weighted error", data: A.errors, borderColor: col(4), pointRadius: 0, yAxisID: "y" },
    { type: "line", label: "alpha", data: A.alphas, borderColor: col(0), pointRadius: 0, yAxisID: "y1" }] },
    options: { scales: { y: { title: { display: true, text: "Weighted error" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "alpha" } }, x: { title: { display: true, text: "Round" } } } } });
  const G = M.regression.gbm_curve;
  mk("gbmChart", { type: "line", data: { labels: G.train.map((_, i) => i + 1), datasets: [
    { label: "Train", data: G.train, borderColor: col(1), pointRadius: 0 }, { label: "Validation", data: G.val, borderColor: col(2), pointRadius: 0 }] },
    options: { scales: { y: { title: { display: true, text: "RMSE (log price)" } }, x: { title: { display: true, text: "Trees" } } } } });
  const oc = M.classification.oob_curve, orr = M.regression.oob_curve;
  mk("oobChart", { type: "line", data: { labels: oc["Bagging"].map((_, i) => i + 1), datasets: [
    { label: "Bagging (class. error)", data: oc["Bagging"], borderColor: col(0), pointRadius: 0, yAxisID: "y" },
    { label: "Random forest (class. error)", data: oc["Random forest"], borderColor: col(3), pointRadius: 0, yAxisID: "y" },
    { label: "Bagging (RMSE)", data: orr["Bagging"], borderColor: col(0), borderDash: [5, 4], pointRadius: 0, yAxisID: "y1" },
    { label: "Random forest (RMSE)", data: orr["Random forest"], borderColor: col(3), borderDash: [5, 4], pointRadius: 0, yAxisID: "y1" }] },
    options: { scales: { y: { title: { display: true, text: "Error rate" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "RMSE" } }, x: { title: { display: true, text: "Trees in the ensemble" } } } } });
}

/* ---------------- tree explorer ---------------- */
function renderTrees() {
  rendered.trees = true;
  const C = M.models.classification, t = C.cart, names = ["Budget", "Mid-range", "Premium"], P = M.classification.prune_path;
  const node = (n, label) => {
    const v = t.v[n], k = HP.argmax(v), isLeaf = t.f[n] < 0, path = state.pathNodes.has(n);
    const txt = `${label ? `<b>${label}</b> → ` : ""}${names[k]} ${pct(v[k], 0)} <span style="opacity:.7">(n=${t.n[n]})</span>`;
    if (isLeaf) return `<li><span class="tn leaf ${path ? "path" : ""}">${txt}</span></li>`;
    return `<li><details ${path || t.n[n] > t.n[0] * 0.2 ? "open" : ""}><summary><span class="tn ${path ? "path" : ""}">${txt}</span></summary><ul class="tree">${node(t.l[n], esc(cond(t.f[n], t.t[n], true)))}${node(t.r[n], esc(cond(t.f[n], t.t[n], false)))}</ul></details></li>`;
  };
  $("trees").innerHTML = `<h2>Tree explorer</h2>
  <p class="lead">A CART classifier written from scratch (weighted Gini splits, then minimal cost-complexity pruning with the strength chosen on a validation set). The highlighted branch is the path of the car you entered.</p>
  <div class="grid"><div class="card wide"><h3>Pruned classification tree: ${t.f.filter(f => f < 0).length} leaves (from ${C.cart_full_leaves} before pruning)</h3><div class="scroll"><ul class="tree">${node(0, "")}</ul></div></div>
  ${card("Cost-complexity pruning path", cv("pruneChart") + `<p class="note">Each pruning step removes the weakest link (smallest alpha). Validation accuracy peaks well before the full tree, which overfits.</p>`)}
  ${card("What the splits are made of", `<p class="note" style="margin-top:0">Features are age, odometer, owner count, fuel, seller, transmission, engine size, brand flags and two leakage-free target encodings (typical log-price of the car's model and variant, computed out-of-fold on the training rows).</p>`)}</div>`;
  mk("pruneChart", { data: { labels: P.map(p => p.leaves), datasets: [{ type: "line", label: "Validation accuracy", data: P.map(p => p.val_acc), borderColor: col(1), pointRadius: 2 }] },
    options: { scales: { x: { reverse: true, title: { display: true, text: "Leaves (more pruning →)" } }, y: { title: { display: true, text: "Validation accuracy" } } }, plugins: { legend: { display: false } } } });
}

/* ---------------- evidence ---------------- */
function renderEvidence() {
  const C = M.classification, R = M.regression, S = M.sklearn, IV = R.intervals;
  const bestC = Object.entries(C.results).filter(([n]) => !n.startsWith("Majority")).reduce((a, b) => b[1].acc > a[1].acc ? b : a)[0];
  const bestR = Object.entries(R.results).filter(([n]) => !n.startsWith("Mean")).reduce((a, b) => b[1].rmse_log < a[1].rmse_log ? b : a)[0];
  const ct = Object.entries(C.results).sort((a, b) => b[1].acc - a[1].acc);
  const rt = Object.entries(R.results).sort((a, b) => a[1].rmse_log - b[1].rmse_log);
  const cm = C.results["Stacking (softmax)"].confusion, tn = ["Budget", "Mid", "Premium"];
  const skC = S.classification_acc, skR = S.regression_rmse_log;
  const mth = IV.methods, key = "normalised conformal (HonestPrice)";
  $("evidence").innerHTML = `<h2>Evidence</h2>
  <p class="lead">All numbers are on ${M.meta.split.test} cars the models never saw (${M.meta.split.train} train, ${M.meta.split.val} validation, ${M.meta.split.cal} calibration, ${M.meta.split.test} test). ${M.meta.duplicates_removed} exact duplicate listings were removed first so that identical cars cannot appear on both sides of the split.</p>
  <div class="kpis">
    <div class="kpi"><div class="v">${pct(C.results[bestC].acc)}</div><div class="l">tier accuracy (${bestC})</div></div>
    <div class="kpi"><div class="v">${R.results[bestR].rmse_log.toFixed(3)}</div><div class="l">RMSE of log price (${bestR})</div></div>
    <div class="kpi"><div class="v">${pct(mth[key][4].coverage)}</div><div class="l">coverage of the 90% interval</div></div>
    <div class="kpi"><div class="v">${pct(C.disagreement.acc_unanimous, 0)} / ${pct(C.disagreement.acc_split, 0)}</div><div class="l">tier accuracy: council unanimous / split</div></div>
  </div>
  <div class="grid">
    ${card("Price tier classification (test set)", `<div class="scroll"><table><tr><th>Model</th><th class="n">Accuracy</th><th class="n">Macro F1</th><th class="n">Log loss</th><th class="n">Off by two tiers</th></tr>
      ${ct.map(([n, m]) => `<tr class="${n === bestC ? "best" : ""}"><td>${n}</td><td class="n">${pct(m.acc)}</td><td class="n">${m.macro_f1.toFixed(3)}</td><td class="n">${m.log_loss.toFixed(3)}</td><td class="n">${pct(m.off_by_two)}</td></tr>`).join("")}</table></div>
      <p class="note">Differences of 1-2 points between ensembles are within sampling noise on ${M.meta.split.test} cars (about &plusmn;3.5 points). The reliable findings are: every ensemble beats a lone pruned CART on log loss, and all are far above the majority baseline.</p>`)}
    ${card("Price regression (test set)", `<div class="scroll"><table><tr><th>Model</th><th class="n">RMSE log</th><th class="n">MAE</th><th class="n">Median error</th><th class="n">R² (price)</th></tr>
      ${rt.map(([n, m]) => `<tr class="${n === bestR ? "best" : ""}"><td>${n}</td><td class="n">${m.rmse_log.toFixed(3)}</td><td class="n">${inr(m.mae_inr)}</td><td class="n">${pct(m.median_ape, 0)}</td><td class="n">${m.r2_price.toFixed(2)}</td></tr>`).join("")}</table></div>
      <p class="note">RMSE of 0.30 in log price is a typical error of roughly 30%, while the median absolute error is about 17%. That is the noise level of asking prices: the same car is listed at very different prices.</p>`)}
    ${card("Are the error bars honest? (reliability)", cv("relChart") + `<p class="note">Points on the diagonal mean a 90% interval contains the truth 90% of the time. The Gaussian BMA interval, which assumes the model is right, over-covers at every level (and is wider); both conformal variants track the diagonal.</p>`)}
    ${card("Interval methods at 90%", `<table><tr><th>Method</th><th class="n">Coverage</th><th class="n">Median width (upper / lower price)</th></tr>
      ${Object.entries(mth).map(([n, v]) => `<tr class="${n === key ? "best" : ""}"><td>${n}</td><td class="n">${pct(v[4].coverage)}</td><td class="n">×${v[4].width.toFixed(2)}</td></tr>`).join("")}</table>
      <p class="note" style="margin-top:12px"><b>Coverage by predicted price third</b></p><table><tr><th>Segment</th><th class="n">Cars</th><th class="n">90% coverage</th><th class="n">Width</th></tr>
      ${IV.by_tier.map(t => `<tr><td>${t.tier}</td><td class="n">${t.n}</td><td class="n">${pct(t.coverage90)}</td><td class="n">×${t.median_ratio.toFixed(2)}</td></tr>`).join("")}</table>`)}
    ${card("What the committee agreement tells you", `<table><tr><th>Keep the N% most confident</th><th class="n">Tier accuracy</th></tr>${C.disagreement.selective.map(s => `<tr><td>${pct(s.coverage, 0)}</td><td class="n">${pct(s.acc)}</td></tr>`).join("")}</table>
      <p class="note">Members agreed unanimously on ${pct(C.disagreement.unanimous_share, 0)} of test cars. When they did, the stacked tier was right ${pct(C.disagreement.acc_unanimous, 0)} of the time versus ${pct(C.disagreement.acc_split, 0)} when they split, so abstaining on split cases is a real accuracy gain.</p>`)}
    ${card("Bayesian model averaging weights", `<table><tr><th>Model</th><th class="n">Exact posterior</th><th class="n">Tempered (production)</th><th class="n">NNLS stack</th></tr>
      ${M.models.regression.names.map(n => `<tr><td>${n}</td><td class="n">${pct(R.weights.bma_exact[n], 0)}</td><td class="n">${pct(R.weights.bma_tempered[n], 0)}</td><td class="n">${pct(R.weights.nnls[n], 0)}</td></tr>`).join("")}</table>
      <p class="note">Exact BMA piles almost all weight on one model because validation likelihoods differ by many nats. Tempering the likelihood by 1/√n keeps the average a real committee and gave the best test error.</p>`)}
    ${card("Cross-check against scikit-learn", `<table><tr><th>Task / model</th><th class="n">Ours</th><th class="n">scikit-learn</th></tr>
      ${Object.entries(skC).map(([n, v]) => `<tr><td>Tier accuracy, ${n}</td><td class="n">${pct(v[0])}</td><td class="n">${pct(v[1])}</td></tr>`).join("")}
      ${Object.entries(skR).map(([n, v]) => `<tr><td>RMSE log price, ${n}</td><td class="n">${v[0].toFixed(3)}</td><td class="n">${v[1].toFixed(3)}</td></tr>`).join("")}</table>
      <p class="note">Root split of our CART is identical to scikit-learn's: <b>${S.root_split_match ? "yes" : "no"}</b>. Small gaps come from different tie-breaking, binning (64 bins) and random seeds.</p>`)}
    ${card("Permutation importance (gradient boosting)", cv("impChart") + `<p class="note">Increase in test RMSE (log price) when one feature is shuffled.</p>`)}
    ${card("Confusion matrix: stacked council", `<table><tr><th>actual \\ predicted</th>${tn.map(n => `<th class="n">${n}</th>`).join("")}</tr>${cm.map((r, i) => `<tr><td>${tn[i]}</td>${r.map((v, j) => `<td class="n" style="${i === j ? "font-weight:600;background:var(--soft)" : ""}">${v}</td>`).join("")}</tr>`).join("")}</table>
      <p class="note">Mistakes are almost always to the neighbouring tier; skipping a tier happens for ${pct(C.results["Stacking (softmax)"].off_by_two)} of cars.</p>`)}
  </div>`;
  const lv = mth[key].map(v => v.level * 100);
  mk("relChart", { type: "line", data: { labels: lv, datasets: [
    { label: "Ideal", data: lv, borderColor: css("--mut"), borderDash: [4, 4], pointRadius: 0 },
    ...Object.entries(mth).map(([n, v], i) => ({ label: n, data: v.map(p => p.coverage * 100), borderColor: col([1, 0, 4][i]), pointRadius: 3 }))] },
    options: { scales: { x: { title: { display: true, text: "Nominal level (%)" } }, y: { title: { display: true, text: "Empirical coverage (%)" }, min: 40, max: 100 } } } });
  const im = M.importance.slice(0, 10);
  mk("impChart", { type: "bar", data: { labels: im.map(i => i[0]), datasets: [{ data: im.map(i => i[1]), backgroundColor: col(1) }] }, options: { indexAxis: "y", plugins: { legend: { display: false } } } });
}

/* ---------------- method ---------------- */
function renderMethod() {
  $("method").innerHTML = `<h2>Method</h2>
  <p class="lead">Every model is written from scratch in NumPy (<code>pipeline/trees.py</code>, <code>pipeline/ensembles.py</code>) and cross-checked against scikit-learn. The browser runs the exported trees in JavaScript, and a parity test confirms it reproduces the Python predictions.</p>
  <div class="grid">
    ${card("Unit 5 syllabus map", `<table><tr><th>Topic</th><th>Where</th></tr>
      <tr><td>CART (tree-based models)</td><td>Gini / squared-error splits, sample weights, cost-complexity pruning. Tree explorer.</td></tr>
      <tr><td>Committees</td><td>Bagging and random forests, out-of-bag error. Boosting lab.</td></tr>
      <tr><td>Boosting, AdaBoost</td><td>SAMME with weight visualiser. Boosting lab.</td></tr>
      <tr><td>Additive models</td><td>Gradient boosting as a stagewise sum of trees. Valuation, Boosting lab.</td></tr>
      <tr><td>Bayesian model averaging</td><td>Posterior weights from validation likelihood (exact and tempered). Evidence.</td></tr>
      <tr><td>Ensemble learning</td><td>Stacking (softmax and NNLS), simple averaging, comparison. Evidence.</td></tr>
      <tr><td>Practice: CART for categorisation</td><td>Price-tier classification (Budget / Mid-range / Premium).</td></tr>
      <tr><td>Practice: ensembles for classification</td><td>Bagging, random forest, AdaBoost, stacking on the same task.</td></tr></table>`)}
    ${card("Data and protocol", `<p class="note" style="margin-top:0">CarDekho used-car listings from India (${M.meta.rows_raw} rows, ${M.meta.duplicates_removed} exact duplicates removed, ${M.meta.rows_clean} used). Split into train / validation / calibration / test with a fixed seed. Validation chooses pruning strength, early stopping, temperature scaling and the model weights. The calibration set is used only for conformal quantiles. The test set is touched once.</p>
      <p class="note">Target-encoded features (typical log-price of a model or variant) are computed out-of-fold on the training rows so a car cannot encode its own price.</p>`)}
    ${card("Limitations", `<ul class="rules"><li>Prices are asking prices from one marketplace in 2020, not transaction prices, and exclude condition, accident history and city.</li>
      <li>The 90% guarantee is marginal (averaged over cars), not per segment. On the test set the cheapest third was covered ${pct(M.regression.intervals.by_tier[0].coverage90, 0)} of the time and the other thirds ${pct(M.regression.intervals.by_tier[1].coverage90, 0)}; rare brands are not guaranteed.</li>
      <li>Models cannot know about cars, trims or market shifts after 2020.</li>
      <li>Rare models fall back to brand-level averages, so intervals there are less informative.</li>
      <li>Treat the output as decision support, not a formal valuation.</li></ul>`)}
    ${card("Reproduce", `<p class="note" style="margin-top:0"><code>pip install -r requirements.txt</code><br><code>python -m pytest tests</code> &nbsp;(unit tests)<br><code>python pipeline/build.py</code><br><code>node tests/parity.test.js</code> &nbsp;(JS vs Python)<br><code>python -m http.server -d web</code></p>`)}
  </div>`;
}

/* ---------------- boot ---------------- */
const renderers = { boost: renderBoost, trees: renderTrees, evidence: renderEvidence, method: renderMethod };
function show(tab) {
  document.querySelectorAll(".tab").forEach(e => e.classList.toggle("on", e.id === tab));
  document.querySelectorAll("#nav button").forEach(b => b.classList.toggle("on", b.dataset.tab === tab));
  if (renderers[tab] && !rendered[tab]) { renderers[tab](); rendered[tab] = true; }
  history.replaceState(null, "", "#" + tab);
}
fetch("data/model.json").then(r => r.json()).then(m => {
  M = m; $("loading").remove();
  const brands = Object.keys(M.catalog).sort();
  $("brand").innerHTML = brands.map(b => `<option>${esc(b)}</option>`).join("");
  $("ex").innerHTML = `<option value="">Choose a held-out car&hellip;</option>` + M.examples.map((e, i) => `<option value="${i}">${esc(e.name)} (${2020 - e.age})</option>`).join("");
  $("brand").value = "Maruti"; fillModels(); $("model").value = "Maruti Swift"; fillVariants();
  $("brand").onchange = () => { fillModels(); update(); };
  $("model").onchange = () => { fillVariants(); update(); };
  ["variant", "year", "km", "fuel", "trans", "seller", "owner"].forEach(id => $(id).oninput = () => update());
  $("ex").onchange = () => {
    if ($("ex").value === "") return;
    const e = M.examples[+$("ex").value];
    $("brand").value = e.brand; fillModels(); $("model").value = e.model; fillVariants(); if (![...$("variant").options].some(o => o.value === e.name)) $("variant").add(new Option(e.name, e.name)); $("variant").value = e.name;
    $("year").value = 2020 - e.age; $("km").value = e.km; $("fuel").value = e.fuel; $("trans").value = e.transmission; $("seller").value = e.seller; $("owner").value = e.owner;
    update(); state.actual = e.actual; update(true);
  };
  document.querySelectorAll("#nav button").forEach(b => b.onclick = () => show(b.dataset.tab));
  update();
  const h = location.hash.slice(1); if (renderers[h]) show(h);
}).catch(e => { $("loading").textContent = "Could not load model.json: " + e; });
