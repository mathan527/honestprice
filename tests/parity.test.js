// Verifies that the JavaScript inference engine reproduces the Python predictions.
// Run:  node tests/parity.test.js   (needs web/data/model.json and web/data/golden.json from `python pipeline/build.py`)
const fs = require("fs");
const path = require("path");
const HP = require("../web/model.js");
const M = JSON.parse(fs.readFileSync(path.join(__dirname, "../web/data/model.json"), "utf8"));
const gold = JSON.parse(fs.readFileSync(path.join(__dirname, "../web/data/golden.json"), "utf8"));
let worst = { feat: 0, cls: 0, price: 0, lo: 0 }, bad = 0;
for (const g of gold) {
  const x = HP.features(M, g.in);
  worst.feat = Math.max(worst.feat, ...x.map((v, i) => Math.abs(v - g.features[i])));
  const c = HP.classify(M, x)["Stacking (softmax)"];
  worst.cls = Math.max(worst.cls, ...c.map((v, i) => Math.abs(v - g.class_stack[i])));
  const r = HP.regress(M, x);
  worst.price = Math.max(worst.price, Math.abs(r.mean - g.price_log));
  worst.lo = Math.max(worst.lo, Math.abs(r.intervals["0.9"][0] - g.lo90), Math.abs(r.intervals["0.9"][1] - g.hi90));
}
console.log("max abs differences vs Python:", worst);
const tol = { feat: 2e-5, cls: 2e-3, price: 2e-3, lo: 5e-3 };
for (const k of Object.keys(tol)) if (!(worst[k] <= tol[k])) { console.error(`FAIL ${k}: ${worst[k]} > ${tol[k]}`); bad++; }
if (bad) process.exit(1);
console.log(`OK: ${gold.length} cars match`);
