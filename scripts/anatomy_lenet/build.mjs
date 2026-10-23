// Diagrama isométrico de LeNet-5 como banco de trabajo.
// Lee figures/lenet/anatomy_data.json (exportado por notebooks/04_lenet_mnist.ipynb)
// y escribe figures/lenet/lenet_anatomy.svg y figures/lenet/lenet_anatomy.html.
//   node scripts/anatomy_lenet/build.mjs [--dark]
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "./kit/iso-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..");
const OUT = join(ROOT, "figures", "lenet");
const D = JSON.parse(readFileSync(join(OUT, "anatomy_data.json"), "utf8"));

const WIDTH = 960;
const HEIGHT = 620;
const S = 1.3; // unidades de mundo por píxel de mapa
const RIM = 1.6; // marco de la tarjeta alrededor de la imagen
const T = 1.4; // grosor de una tarjeta

const PLATE = { x: -26, y: -158, w: 362, d: 326, r: 10 };
const PLATE_H = 12;
const FOOT_H = 5;

const X = { input: -2, c1: 40, g1: 61, s2: 78, c3: 112, g2: 131, s4: 146, w5: [160, 216], f5: 228, w6: [238, 262], f6: 272, w7: [282, 296], out: 310 };
const RAIL_H = 3;

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [X.input, -24, 70], [X.input, 24, 70]],
  WIDTH,
  HEIGHT,
  { pad: 34, azimuth: 38, elevation: 32 },
);

const at = (p) => k.iso(p, P);
const Sd = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const Dt = (points, style) => k.dotsSvg(points, style);
const depth = (x, y, z = 0) => k.depthOf([x, y, z], P);
const r1 = (v) => Math.round(v * 1000) / 1000;

// Afinidad de un plano del mundo a la pantalla: origen o, ejes u (columnas) y v (filas).
function planeMatrix(o, u, v) {
  const p0 = at(o);
  const pu = at([o[0] + u[0], o[1] + u[1], o[2] + u[2]]);
  const pv = at([o[0] + v[0], o[1] + v[1], o[2] + v[2]]);
  return `matrix(${r1(pu[0] - p0[0])} ${r1(pu[1] - p0[1])} ${r1(pv[0] - p0[0])} ${r1(pv[1] - p0[1])} ${r1(p0[0])} ${r1(p0[1])})`;
}

// Celdas de una matriz como trayectorias agrupadas por nivel de opacidad (coordenadas locales enteras).
const LEVELS = 7;
function cellPaths(grid, scale, cls, { signed = false } = {}) {
  const buckets = new Map();
  grid.forEach((row, r) =>
    row.forEach((value, c) => {
      const v = Math.min(1, Math.abs(value) / scale);
      if (v < 0.05) return;
      const level = Math.max(1, Math.ceil(v * LEVELS));
      const key = `${signed && value < 0 ? "lc-neg" : cls}|${level}`;
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(`M${c} ${r}h1v1h-1z`);
    }),
  );
  return [...buckets.entries()]
    .map(([key, cells]) => {
      const [name, level] = key.split("|");
      return `<path class="${name}" fill-opacity="${((Number(level) / LEVELS) * 0.92).toFixed(2)}" d="${cells.join("")}"/>`;
    })
    .join("");
}

const maxOf = (a) => (Array.isArray(a) ? Math.max(...a.map(maxOf)) : Math.abs(a));
const byDepth = (items) => [...items].sort((a, b) => a.depth - b.depth);

// ---------------------------------------------------------------- base
function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of k.corners(PLATE, 26).sort((a, b) => depth(...a) - depth(...b))) {
    out.push(Sd(k.cylinder(x, y, 8, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(Sd(k.cylinder(x, y, 6.2, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  const seam = k.sideSeam(PLATE, -4, P, 8);
  out.push(Sd(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid", inner: L(seam, { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 11);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.4, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(Dt(screws.map(([x, y]) => at([x, y, 0])), { size: 0.55 }));
  return out.join("");
}

function ruler() {
  const strip = { x: -14, y: PLATE.y + PLATE.d - 14, w: PLATE.w - 24, d: 7, r: 1 };
  const out = [Sd(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" })];
  const ticks = k.topTicks(strip.x + 4, strip.x + strip.w - 4, 4, 5, strip.y + 0.8, 0.6, [2, 3.8], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(Dt([at([strip.x + 2, strip.y + 3.5, 0.6]), at([strip.x + strip.w - 2, strip.y + 3.5, 0.6])], { size: 0.45 }));
  return out.join("");
}

function labelPlate() {
  const plan = { x: 262, y: 120, w: 40, d: 20, r: 1.5 };
  const z = 0.8;
  const out = [Sd(k.slabOf(plan, 0, z, P, 4), { tone: "mid" })];
  out.push(L(k.planOutline(k.insetPlan(plan, 1.6), z, P, 4), { tone: "faint" }));
  out.push(k.faceTextSvg(k.topMatrix([plan.x + 4, plan.y + 7], z, P, "x"), "LeNet-5", { size: 5, tone: "hi" }));
  for (const [i, len] of [[0, 30], [1, 22], [2, 26]]) {
    const y = plan.y + 10.5 + i * 2.6;
    out.push(L(k.lineOnTop([plan.x + 4, y], [plan.x + 4 + len, y], z, P), { tone: "lo", free: true }));
  }
  out.push(Dt([at([plan.x + 2.6, plan.y + 2.6, z]), at([plan.x + plan.w - 2.6, plan.y + plan.d - 2.6, z])], { size: 0.45 }));
  return out.join("");
}

// ---------------------------------------------------------------- piezas
function rail(xc, half, w = 8) {
  const plan = { x: xc - w / 2, y: -half, w, d: 2 * half, r: 1.5 };
  return Sd(k.slabOf(plan, 0, RAIL_H, P, 3, 0.4), { tone: "mid" }) + L(k.lineOnTop([xc, -half], [xc, half], RAIL_H, P), { tone: "lo" });
}

function stop(xc, y, w = 8) {
  const plan = { x: xc - w / 2, y: y - 2, w, d: 4, r: 1 };
  return Sd(k.slabOf(plan, 0, 7, P, 3, 0.4), { tone: "mid" }) + Dt([at([xc - 2, y, 7]), at([xc + 2, y, 7])], { size: 0.45 });
}

// Tarjeta vertical (plano y-z) con la cara +x grabada con un mapa de activación.
function card(xc, yc, z0, map, scale, { input = false } = {}) {
  const n = map.length;
  const side = n * S + 2 * RIM;
  const plan = { x: xc - T / 2, y: yc - side / 2, w: T, d: side, r: 0.35 };
  const face = xc + T / 2;
  const zTop = z0 + side;
  const m = planeMatrix([face, yc + side / 2 - RIM, zTop - RIM], [0, -S, 0], [0, 0, -S]);
  const frame = `<path class="iso-line" data-tone="faint" d="M0 0H${n}V${n}H0Z"/>`;
  const cells = cellPaths(map, scale, input ? "lc-digit" : "lc-ink");
  const inner = `<g transform="${m}">${frame}${cells}</g>`;
  return Sd(k.slabOf(plan, z0, side, P, 2), { tone: "hi", crease: "lo", inner });
}

// Sello de pesos: bloque ámbar con el kernel grabado en la cara superior.
function stamp(xc, yc, z0, kernel, scale, cell, layers = 1, layerH = 3) {
  const n = kernel.length;
  const side = n * cell + 2;
  const plan = { x: xc - side / 2, y: yc - side / 2, w: side, d: side, r: 0.8 };
  const h = layers * layerH;
  const zTop = z0 + h;
  const m = planeMatrix([xc - side / 2 + 1, yc + side / 2 - 1, zTop], [0, -cell, 0], [cell, 0, 0]);
  const seams = [];
  for (let i = 1; i < layers; i++) seams.push(L(k.sideSeam(plan, z0 + i * layerH, P, 3), { tone: "faint" }));
  const top = `<g transform="${m}"><path class="iso-line" data-tone="faint" d="M0 0H${n}V${n}H0Z"/>${cellPaths(kernel, scale, "lc-pos", { signed: true })}</g>`;
  return Sd(k.slabOf(plan, z0, h, P, 3, 0.4), { tone: "hi", crease: "lo", className: "wt", inner: seams.join("") + top });
}

function grate(xc, half) {
  const plan = { x: xc - 4, y: -half, w: 8, d: 2 * half, r: 1 };
  const out = [Sd(k.slabOf(plan, 0, 1.2, P, 3), { tone: "lo" })];
  out.push(L(k.planOutline(k.insetPlan(plan, 0.9), 1.2, P, 3), { tone: "faint" }));
  const rows = Math.floor((2 * half - 4) / 2.6);
  out.push(Dt(k.dotGrid(xc - 2.6, -half + 2 + ((2 * half - 4) - (rows - 1) * 2.6) / 2, 3, rows, 2.6, 1.2, P), { size: 0.42, tone: "lo" }));
  return out.join("");
}

// Placa de pesos de una capa totalmente conectada: filas = neuronas de salida (en -y), columnas = entradas (en +x).
function matrixPlate(x0, x1, half, grid) {
  const frame = { x: x0, y: -half, w: x1 - x0, d: 2 * half, r: 1.5 };
  const inner = k.insetPlan(frame, 2);
  const rows = grid.length;
  const cols = grid[0].length;
  const zTop = 3.2;
  const m = planeMatrix([inner.x, inner.y + inner.d, zTop], [inner.w / cols, 0, 0], [0, -inner.d / rows, 0]);
  const scale = maxOf(grid) * 0.8;
  const out = [Sd(k.slabOf(frame, 0, 1.6, P, 3, 0.4), { tone: "mid" })];
  const top = `<g transform="${m}"><path class="iso-line" data-tone="faint" d="M0 0H${cols}V${rows}H0Z"/>${cellPaths(grid, scale, "lc-pos", { signed: true })}</g>`;
  out.push(Sd(k.slabOf(inner, 1.6, zTop - 1.6, P, 3, 0.4), { tone: "hi", crease: "lo", className: "wt", inner: top }));
  out.push(Dt([[frame.x + 1, frame.y + 1], [frame.x + frame.w - 1, frame.y + frame.d - 1], [frame.x + 1, frame.y + frame.d - 1]].map(([x, y]) => at([x, y, 1.6])), { size: 0.42 }));
  return out.join("");
}

// Hilera de pernos: una neurona por perno, altura = activación.
function pins(xc, values, pitch, r, gain) {
  const n = values.length;
  const y0 = ((n - 1) * pitch) / 2;
  const items = values.map((v, i) => {
    const y = y0 - i * pitch;
    const h = 0.8 + v * gain;
    return { depth: depth(xc, y), svg: Sd(k.cylinder(xc, y, r, RAIL_H, h, P, 10), { tone: v > 0 ? "hi" : "lo", crease: "none" }) };
  });
  return byDepth(items).map((i) => i.svg).join("");
}

// ---------------------------------------------------------------- estaciones
const rowCenters = (count, side, gap) => {
  const pitch = side + gap;
  const y0 = ((count - 1) * pitch) / 2;
  return Array.from({ length: count }, (_, i) => y0 - i * pitch);
};

const SC = { input: 1, c1: maxOf(D.c1), s2: maxOf(D.s2), c3: maxOf(D.c3), s4: maxOf(D.s4) };
const C1_SIDE = 28 * S + 2 * RIM;
const C3_SIDE = 10 * S + 2 * RIM;
const Y6 = rowCenters(6, C1_SIDE, 5);
const Y16 = rowCenters(16, C3_SIDE, 1.6);
const HALF6 = Y6[0] + C1_SIDE / 2 + 4.5;
const HALF16 = Y16[0] + C3_SIDE / 2 + 4.5;

function inputStation() {
  const ped = { x: X.input - 13, y: -31, w: 26, d: 62, r: 2 };
  const pedH = 18;
  const base = { x: X.input - 8, y: -29, w: 16, d: 58, r: 1.5 };
  const z1 = pedH + 4;
  const side = 32 * S + 2 * RIM;
  const out = [];
  out.push(Sd(k.slabOf(ped, 0, pedH, P, 4, 0.8), { tone: "mid", inner: L(k.sideSeam(ped, 4, P, 4), { tone: "faint" }) }));
  out.push(Dt(k.corners(ped, 2.4).map(([x, y]) => at([x, y, pedH])), { size: 0.45 }));
  out.push(Sd(k.slabOf(base, pedH, 4, P, 3, 0.4), { tone: "mid" }));
  out.push(L(k.lineOnTop([X.input, base.y], [X.input, base.y + base.d], z1, P), { tone: "lo" }));
  const clip = (y) => Sd(k.slabOf({ x: X.input - 3, y: y - 2, w: 6, d: 4, r: 0.8 }, z1, 14, P, 3, 0.4), { tone: "mid" }) + Dt([at([X.input, y, z1 + 14])], { size: 0.45 });
  out.push(clip(-(side / 2 + 2)));
  out.push(card(X.input, 0, z1, D.input, SC.input, { input: true }));
  out.push(clip(side / 2 + 2));
  return out.join("");
}

// Una fila de tarjetas en su riel. Con kernels, cada tarjeta lleva encima el sello que la imprimió.
function cardRow(xc, centers, half, maps, scale, seal = null) {
  const out = [rail(xc, half), stop(xc, -half - 2)];
  const items = centers.map((y, i) => {
    let svg = card(xc, y, RAIL_H, maps[i], scale);
    if (seal) {
      const top = RAIL_H + maps[i].length * S + 2 * RIM;
      svg += Sd(k.slabOf({ x: xc - 1.6, y: y - 3, w: 3.2, d: 6, r: 0.6 }, top, 1.4, P, 2), { tone: "mid" });
      svg += stamp(xc, y, top + 1.4, seal.kernels[i], seal.scale, seal.cell, seal.layers, seal.layerH);
    }
    return { depth: depth(xc, y), svg };
  });
  out.push(byDepth(items).map((i) => i.svg).join(""));
  out.push(stop(xc, half + 2));
  return out.join("");
}

// Sellos de C3: seis láminas, la de abajo es el canal 0 y la grabada arriba es el canal 5.
function outputStation() {
  const panel = { x: X.out - 10, y: -104, w: 20, d: 208, r: 2 };
  const pz = 4;
  const out = [Sd(k.slabOf(panel, 0, pz, P, 4, 0.6), { tone: "mid", inner: L(k.sideSeam(panel, 1.6, P, 4), { tone: "faint" }) })];
  const items = D.probs.map((p, c) => {
    const y = 90 - c * 20;
    const lit = c === D.pred;
    const h = 1.6 + p * 22;
    const parts = [
      Sd(k.cylinder(X.out - 2, y, 6.2, pz, 1.4, P, 28), { tone: "lo" }),
      L(k.ring(X.out - 2, y, 7.6, pz, P, 28), { tone: "faint" }),
      Sd(k.cylinder(X.out - 2, y, 4.2, pz + 1.4, h, P, 24, 0.5), { tone: lit ? "hi" : "mid", lit }),
      k.faceTextSvg(planeMatrix([X.out + 8.6, y, pz], [0, -1, 0], [1, 0, 0]), String(c), { anchor: "middle", size: 5, tone: lit ? "hi" : "lo" }),
    ];
    return { depth: depth(X.out, y), svg: parts.join("") };
  });
  out.push(byDepth(items).map((i) => i.svg).join(""));
  return out.join("");
}

const body = [
  `<rect class="lc-card" x="0" y="0" width="${WIDTH}" height="${HEIGHT}"/>`,
  plate(),
  inputStation(),
  cardRow(X.c1, Y6, HALF6, D.c1, SC.c1, { kernels: D.k1, scale: maxOf(D.k1), cell: 2.0, layers: 1, layerH: 3.6 }),
  grate(X.g1, HALF6),
  cardRow(X.s2, Y6, HALF6, D.s2, SC.s2),
  cardRow(X.c3, Y16, HALF16, D.c3, SC.c3, { kernels: D.k3.map((kern) => kern[5]), scale: maxOf(D.k3), cell: 1.6, layers: 6, layerH: 1.1 }),
  grate(X.g2, HALF16),
  cardRow(X.s4, Y16, HALF16, D.s4, SC.s4),
  matrixPlate(X.w5[0], X.w5[1], 140, D.w5),
  rail(X.f5, 141) + pins(X.f5, D.f5, 2.3, 0.8, 1.5),
  matrixPlate(X.w6[0], X.w6[1], 136, D.w6),
  rail(X.f6, 137) + pins(X.f6, D.f6, 3.2, 1.1, 1.5),
  matrixPlate(X.w7[0], X.w7[1], 102, D.w7),
  outputStation(),
  labelPlate(),
  ruler(),
];

const EXTRA_CSS = `
.iso{--w-pos:#ffb347;--w-neg:#6b9bff;--digit:#f2f2f2}
.iso[data-theme="light"]{--w-pos:#b8650a;--w-neg:#2a5fc4;--digit:#111114}
.iso .wt{--anatomy-top:#2f2615;--anatomy-shade-0:#1a140a;--anatomy-shade-1:#1f180d;--anatomy-shade-2:#251d10;--anatomy-shade-3:#2a2113;--anatomy-hi:#e0a64e;--anatomy-lo:#6e5530;--anatomy-faint:#4a3a20}
.iso[data-theme="light"] .wt{--anatomy-top:#fbf0dc;--anatomy-shade-0:#e3c48f;--anatomy-shade-1:#ebd2a6;--anatomy-shade-2:#f1ddb9;--anatomy-shade-3:#f6e7cb;--anatomy-hi:#8a5a12;--anatomy-lo:#cfaa6c;--anatomy-faint:#e2c896}
.lc-card{fill:var(--anatomy-card)}
.lc-ink{fill:var(--anatomy-hi)}
.lc-digit{fill:var(--digit)}
.lc-pos{fill:var(--w-pos)}
.lc-neg{fill:var(--w-neg)}
`;

const label =
  "LeNet-5 dibujada como un banco de trabajo. Al fondo, una tarjeta con el dígito de entrada de 32 por 32. Delante, una bandeja con 6 sellos ámbar de 5 por 5 (los kernels de C1), la fila de 6 tarjetas de 28 por 28 de C1, una rejilla de pooling, 6 tarjetas de 14 por 14 de S2, una bandeja con 16 sellos de seis láminas (los kernels de C3), 16 tarjetas de 10 por 10 de C3, otra rejilla y 16 tarjetas de 5 por 5 de S4. Siguen las placas ámbar de pesos de F5, F6 y salida, con hileras de 120 y 84 pernos cuya altura es la activación, y diez indicadores de salida con la clase predicha resaltada.";

const theme = process.argv.includes("--dark") ? "dark" : "light";
const svg = k
  .figureSvg({ width: WIDTH, height: HEIGHT, label, body, standalone: theme })
  .replace("<style>", `<style>${EXTRA_CSS}`);
writeFileSync(join(OUT, "lenet_anatomy.svg"), svg);

const total = Object.values(D.params).reduce((a, b) => a + b, 0);
const plateBody = k.plateHtml({
  fig: "Fig 1",
  title: "LeNet-5 · MNIST",
  hint: "Activaciones en gris, pesos en ámbar",
  readout: `predice ${D.pred} · p = ${D.probs[D.pred].toFixed(3)} · prueba ${(D.test_acc * 100).toFixed(2)} %`,
  keys: [
    { mark: "raised", label: "Tarjetas: mapas de activación del dígito de ejemplo" },
    { mark: "lit", label: "Indicador de la clase predicha" },
    { mark: "red", label: "Pesos: sellos (kernels) y placas (capas densas), ámbar positivo y azul negativo" },
  ],
  caption: `De atrás hacia adelante: entrada 1×32×32, kernels C1 (6×5×5), C1 6×28×28, pooling, S2 6×14×14, kernels C3 (16×6×5×5), C3 16×10×10, pooling, S4 16×5×5, pesos F5 (400×120), F5 120, pesos F6 (120×84), F6 84, pesos de salida (84×10) y 10 clases. ${total.toLocaleString("es-MX")} parámetros en total.`,
  body: k.figureSvg({ width: WIDTH, height: HEIGHT, label, body }),
});
const page = k.pageHtml({ title: "LeNet-5 anatomía", theme, body: plateBody, width: 1040 }).replace("</style>", `${EXTRA_CSS}</style>`);
writeFileSync(join(OUT, "lenet_anatomy.html"), page);
console.log("wrote figures/lenet/lenet_anatomy.svg", `(${(svg.length / 1024).toFixed(0)} KB)`, "and lenet_anatomy.html");
