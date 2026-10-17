/* HonestPrice inference engine. Pure functions (no DOM) so the same file runs in the browser and in Node tests. */
(function (root) {
  const EPS = 1e-7; // thresholds are exported rounded to 7 digits
  const softmax = z => { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };

  function engineSize(name) { const m = /(?<![0-9.])([0-9])\.([0-9])(?![0-9])/.exec(name || ""); return m ? parseFloat(m[1] + "." + m[2]) : NaN; }

  /** Rebuild the training-time feature vector for one car. */
  function features(M, c) {
    const enc = M.models.encoder, brandTe = enc.brand_te[c.brand], modelTe = enc.model_te[c.model];
    const te = modelTe !== undefined ? modelTe : (brandTe !== undefined ? brandTe : enc.prior);
    const vte = c.name && enc.name_te[c.name] !== undefined ? enc.name_te[c.name] : te;
    let eng = engineSize(c.name); if (!(eng === eng)) eng = enc.engine_med;
    return [c.age, Math.log1p(c.km), enc.owner[c.owner], +(c.fuel === "Diesel"), +(c.fuel === "Petrol"), +(c.seller === "Dealer"),
      +(c.seller === "Trustmark Dealer"), +(c.transmission === "Automatic"), te, eng, vte].concat(enc.brands.map(b => +(c.brand === b)));
  }

  function leaf(t, x) { let n = 0; while (t.f[n] >= 0) n = x[t.f[n]] <= t.t[n] + EPS ? t.l[n] : t.r[n]; return n; }
  function pathOf(t, x) { let n = 0; const p = [0]; while (t.f[n] >= 0) { n = x[t.f[n]] <= t.t[n] + EPS ? t.l[n] : t.r[n]; p.push(n); } return p; }
  const val = (t, x) => t.v[leaf(t, x)];
  function forestMean(trees, x, k) {
    if (k === 1) { let s = 0; for (const t of trees) s += val(t, x); return s / trees.length; }
    const s = new Array(k).fill(0); for (const t of trees) { const v = val(t, x); for (let i = 0; i < k; i++) s[i] += v[i]; }
    return s.map(v => v / trees.length);
  }
  const argmax = a => a.indexOf(Math.max(...a));

  function classify(M, x) {
    const C = M.models.classification, K = 3, out = {};
    out["CART (pruned)"] = val(C.cart, x);
    out["Bagging"] = forestMean(C.bagging, x, K);
    out["Random forest"] = forestMean(C.rf, x, K);
    const sc = new Array(K).fill(0); C.adaboost.trees.forEach((t, i) => { sc[argmax(val(t, x))] += C.adaboost.alphas[i]; });
    out["AdaBoost"] = softmax(sc.map(v => v / C.adaboost.tau));
    const feat = [].concat(...C.names.map(n => out[n].map(p => Math.log(Math.min(Math.max(p, 1e-4), 1)))));
    const z = feat.map((v, i) => (v - C.stack.mu[i]) / C.stack.sd[i]);
    const logits = [0, 1, 2].map(k => C.stack.W[0][k] + z.reduce((a, v, i) => a + v * C.stack.W[i + 1][k], 0));
    out["Stacking (softmax)"] = softmax(logits);
    const bmaT = [0, 1, 2].map(k => C.names.reduce((a, n, i) => a + C.bma.tempered[i] * out[n][k], 0));
    out["BMA (tempered)"] = bmaT;
    return out;
  }

  function regress(M, x) {
    const R = M.models.regression, out = {};
    const rg = R.ridge; out["Ridge (linear)"] = rg.coef[0] + x.reduce((a, v, i) => a + rg.coef[i + 1] * (v - rg.mu[i]) / rg.sd[i], 0);
    out["CART"] = val(R.cart, x);
    out["Bagging"] = forestMean(R.bagging, x, 1);
    out["Random forest"] = forestMean(R.rf, x, 1);
    let g = R.gbm.f0; for (const t of R.gbm.trees) g += R.gbm.lr * val(t, x); out["Gradient boosting"] = g;
    const names = R.names, w = R.weights.final, mean = names.reduce((a, n, i) => a + w[i] * out[n], 0);
    const between = Math.sqrt(names.reduce((a, n, i) => a + w[i] * (out[n] - mean) ** 2, 0));
    const rfv = R.rf.map(t => val(t, x)), rm = rfv.reduce((a, b) => a + b, 0) / rfv.length;
    const rfStd = Math.sqrt(rfv.reduce((a, b) => a + (b - rm) ** 2, 0) / rfv.length);
    const sigma = Math.sqrt(rfStd * rfStd + between * between) + R.floor;
    const iv = {}; for (const lv of Object.keys(R.q)) iv[lv] = [mean - R.q[lv] * sigma, mean + R.q[lv] * sigma];
    return { members: out, mean, sigma, between, rfStd, intervals: iv };
  }

  function gbmStaged(M, x) { const g = M.models.regression.gbm, out = [g.f0]; let F = g.f0; for (const t of g.trees) { F += g.lr * val(t, x); out.push(F); } return out; }

  const api = { features, classify, regress, gbmStaged, pathOf, val, engineSize, softmax, argmax };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.HP = api;
})(typeof window !== "undefined" ? window : globalThis);
