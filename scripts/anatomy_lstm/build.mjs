// Figura de una celda LSTM como un canal con tres compuertas de esclusa y un indicador
// Biblia/Corán. Lee figures/lstm/anatomy_data.json y data/processed/lstm_resultados.json,
// ambos de scripts/train_lstm_biblia_coran.py, y escribe:
//   figures/lstm/lstm_celda.svg   figura estática en la última palabra del primer fragmento
//   spa/lstm_datos.js             datos y figura para la sección 5, sin necesidad de fetch
//     node scripts/anatomy_lstm/build.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../anatomy_lenet/kit/iso-kit.mjs";

const RAIZ = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const A = JSON.parse(readFileSync(join(RAIZ, "figures", "lstm", "anatomy_data.json"), "utf8"));
const R = JSON.parse(readFileSync(join(RAIZ, "data", "processed", "lstm_resultados.json"), "utf8"));

const ANCHO = 980;
const ALTO = 520;
const PLACA = { x: -22, y: -32, w: 372, d: 166, r: 8 };
const CANAL = { x0: 0, x1: 300, medio: 14, muro: 2, alto: 14 };
const X_F = 58;
const X_I = 122;
const X_O = 214;
const LADO = { medio: 11, fin: 78 }; // canales laterales de entrada y de salida
const Y_ESCLUSA = 46;
const ALTO_HOJA = 24; // la hoja sube esto cuando la compuerta vale 1
const RIEL = { x: 160, y: 101, w: 178, d: 13 };
const CARRO = 16;
const ESCALA_AGUA = 12 / 0.5; // nivel del agua por unidad de |c| media

const P = k.fitProjection(
  [...k.boxCorners(PLACA, -14, 0), [X_F, 0, 86], [X_I, Y_ESCLUSA, 80], [X_O, Y_ESCLUSA, 80]],
  ANCHO,
  ALTO,
  { pad: 18, azimuth: 66, elevation: 30 },
);
const at = (p) => k.iso(p, P);
const Sd = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const r3 = (v) => Math.round(v * 1000) / 1000;
const resta = (a, b) => [r3(a[0] - b[0]), r3(a[1] - b[1])];
const SUBE = resta(at([0, 0, 1]), at([0, 0, 0]));
const AVANZA = resta(at([1, 0, 0]), at([0, 0, 0]));
const mover = (v, n) => `translate(${r3(v[0] * n)} ${r3(v[1] * n)})`;
const nivel = (m) => Math.min(13, 1 + m * ESCALA_AGUA);

const caja = (x, y, w, d, z, h, opciones = {}) =>
  Sd(k.slabOf({ x, y, w, d, r: Math.min(0.8, w / 2, d / 2) }, z, h, P, 2, 0.2), { tone: "mid", ...opciones });
const cuadro = (x0, y0, x1, y1, z, clase) => `<path class="${clase}" d="${k.pathOf([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]].map(at), true)}"/>`;

const agua = (clave, x0, y0, x1, y1, m) => `<g data-agua="${clave}" transform="${mover(SUBE, nivel(m))}">${cuadro(x0, y0, x1, y1, 0, "lc-agua")}</g>`;

// Hoja de la compuerta: placa ámbar que sube en z según el valor medio de la compuerta.
function hoja(clave, plano, valor) {
  return `<g data-compuerta="${clave}" transform="${mover(SUBE, valor * ALTO_HOJA)}">${Sd(k.slabOf(plano, 0, ALTO_HOJA, P, 2, 0.2), { tone: "hi", crease: "lo", className: "wt" })}</g>`;
}

// Esclusa sobre el canal principal (la hoja corta el canal a lo ancho, en y).
function esclusaPrincipal(valor) {
  const ym = CANAL.medio + CANAL.muro / 2;
  const techo = ALTO_HOJA * 2 + 2;
  return {
    atras: caja(X_F - 3, -ym - 2, 6, 4, 0, techo),
    hoja: hoja("f", { x: X_F - 1.2, y: -CANAL.medio, w: 2.4, d: 2 * CANAL.medio, r: 0.3 }, valor),
    delante: caja(X_F - 3, ym - 2, 6, 4, 0, techo) + caja(X_F - 3, -ym - 2, 6, 2 * ym + 4, techo, 3, { tone: "hi" }),
  };
}

// Esclusa sobre un canal lateral (la hoja corta el canal a lo ancho, en x).
function esclusaLateral(clave, xc, valor) {
  const xm = LADO.medio + CANAL.muro / 2;
  const techo = ALTO_HOJA * 2 + 2;
  return {
    atras: caja(xc - xm - 2, Y_ESCLUSA - 3, 4, 6, 0, techo),
    hoja: hoja(clave, { x: xc - LADO.medio, y: Y_ESCLUSA - 1.2, w: 2 * LADO.medio, d: 2.4, r: 0.3 }, valor),
    delante: caja(xc + xm - 2, Y_ESCLUSA - 3, 4, 6, 0, techo) + caja(xc - xm - 2, Y_ESCLUSA - 3, 2 * xm + 4, 6, techo, 3, { tone: "hi" }),
  };
}

function canalLateral(clave, xc, valor, mNivel) {
  const m = LADO.medio;
  const y0 = CANAL.medio + CANAL.muro;
  const e = esclusaLateral(clave, xc, valor);
  return [
    caja(xc - m - CANAL.muro, y0, CANAL.muro, LADO.fin - y0, 0, CANAL.alto),
    cuadro(xc - m, y0 - CANAL.muro, xc + m, LADO.fin, 0.2, "lc-fondo"),
    agua(clave === "i" ? "nueva" : "salida", xc - m, y0 - CANAL.muro, xc + m, LADO.fin, mNivel),
    e.atras,
    e.hoja,
    e.delante,
    caja(xc + m, y0, CANAL.muro, LADO.fin - y0, 0, CANAL.alto),
    caja(xc - m - CANAL.muro, LADO.fin, 2 * m + 2 * CANAL.muro, CANAL.muro, 0, CANAL.alto),
  ].join("");
}

// Rótulo horizontal en pantalla, anclado a un punto del mundo.
function rotulo(punto, texto, { clase = "lc-rotulo", clave, ancla = "middle", dy = 0 } = {}) {
  const [x, y] = at(punto);
  const atributo = clave ? ` data-texto="${clave}"` : "";
  return `<text class="${clase}"${atributo} x="${r3(x)}" y="${r3(y + dy)}" text-anchor="${ancla}">${texto}</text>`;
}

function rotuloCompuerta(punto, nombre, simbolo, clave, valor) {
  return rotulo(punto, nombre, { dy: -30 }) + rotulo(punto, `${simbolo} = ${valor.toFixed(2).replace(".", ",")}`, { clase: "lc-valor", clave, dy: -13 });
}

function tarjetaPalabra(palabra) {
  const xc = X_I;
  const y = LADO.fin + 16;
  const out = [];
  out.push(caja(xc - 25, y - 4, 50, 8, 0, 3));
  out.push(Sd(k.slabOf({ x: xc - 23, y: y - 0.8, w: 46, d: 1.6, r: 0.3 }, 3, 15, P, 2), { tone: "hi", crease: "lo" }));
  out.push(rotulo([xc, y + 0.8, 10.5], palabra, { clase: "lc-palabra", clave: "palabra", dy: 5 }));
  return out.join("");
}

function indicador(p) {
  const out = [];
  const zTop = 3;
  out.push(Sd(k.slabOf({ ...RIEL, r: 1.5 }, 0, zTop, P, 3, 0.4), { tone: "mid" }));
  out.push(L(k.lineOnTop([RIEL.x + 6, RIEL.y + RIEL.d / 2], [RIEL.x + RIEL.w - 6, RIEL.y + RIEL.d / 2], zTop, P), { tone: "lo" }));
  const xm = RIEL.x + RIEL.w / 2;
  out.push(L(k.lineOnTop([xm, RIEL.y], [xm, RIEL.y + RIEL.d], zTop, P), { tone: "hi" }));
  const recorrido = RIEL.w - CARRO - 4;
  const x0 = RIEL.x + 2;
  const yc = RIEL.y + RIEL.d / 2;
  out.push(rotulo([RIEL.x, RIEL.y + RIEL.d + 3, 0], "Biblia", { dy: 16 }));
  out.push(rotulo([RIEL.x + RIEL.w, RIEL.y + RIEL.d + 3, 0], "Corán", { dy: 16 }));
  out.push(`<g data-carro transform="${mover(AVANZA, p * recorrido)}">${Sd(k.slabOf({ x: x0, y: RIEL.y - 1.5, w: CARRO, d: RIEL.d + 3, r: 1.2 }, zTop, 7, P, 3, 0.5), { tone: "hi", lit: true })}${rotulo([x0 + CARRO / 2, yc, zTop + 7], `P(Corán) = ${p.toFixed(2).replace(".", ",")}`, { clase: "lc-prob", clave: "p", dy: -12 })}</g>`);
  return out.join("");
}

function dibujo(doc, t) {
  const c = doc.celda;
  const media = (v) => v.reduce((a, b) => a + b, 0) / v.length;
  const f = media(c.f[t]);
  const i = media(c.i[t]);
  const o = media(c.o[t]);
  const y1 = CANAL.medio;
  const e = esclusaPrincipal(f);
  const muroCerca = (x0, x1) => caja(x0, y1, x1 - x0, CANAL.muro, 0, CANAL.alto);
  return [
    `<rect class="lc-tarjeta" width="${ANCHO}" height="${ALTO}"/>`,
    `<path class="iso-halo" d="${k.haloOf(PLACA, -14, 12, P)}"/>`,
    Sd(k.slabOf(PLACA, -12, 12, P, 8, 1.4), { tone: "mid" }),
    // Canal principal: la memoria c
    caja(CANAL.x0, -y1 - CANAL.muro, CANAL.x1 - CANAL.x0, CANAL.muro, 0, CANAL.alto),
    cuadro(CANAL.x0, -y1, CANAL.x1, y1, 0.2, "lc-fondo"),
    agua("previa", CANAL.x0, -y1, X_F, y1, c.memoria_previa[t]),
    agua("retenida", X_F, -y1, X_I, y1, c.memoria_retenida[t]),
    agua("memoria", X_I, -y1, CANAL.x1, y1, c.memoria[t]),
    e.atras,
    e.hoja,
    muroCerca(CANAL.x0, X_I - LADO.medio - CANAL.muro),
    e.delante,
    canalLateral("i", X_I, i, c.memoria_nueva[t]),
    muroCerca(X_I + LADO.medio + CANAL.muro, X_O - LADO.medio - CANAL.muro),
    canalLateral("o", X_O, o, c.salida[t]),
    muroCerca(X_O + LADO.medio + CANAL.muro, CANAL.x1),
    caja(CANAL.x1, -y1 - CANAL.muro, CANAL.muro, 2 * y1 + 2 * CANAL.muro, 0, CANAL.alto),
    rotulo([CANAL.x0 + 2, -y1, CANAL.alto], "Memoria", { clase: "lc-valor", ancla: "start", dy: -10 }),
    rotulo([CANAL.x1, -y1, CANAL.alto], "Memoria nueva", { clase: "lc-valor", dy: -10 }),
    rotuloCompuerta([X_F, 0, ALTO_HOJA * 2 + 5], "Olvido", "f", "f", f),
    rotuloCompuerta([X_I, Y_ESCLUSA, ALTO_HOJA * 2 + 5], "Entrada", "i", "i", i),
    rotuloCompuerta([X_O, Y_ESCLUSA, ALTO_HOJA * 2 + 5], "Salida", "o", "o", o),
    tarjetaPalabra(doc.palabras[t]),
    indicador(doc.p_coran[t]),
  ].join("");
}

const CSS = `
.iso .wt{--anatomy-top:#e8b35e;--anatomy-shade-0:#b77f2e;--anatomy-shade-1:#c48c38;--anatomy-shade-2:#d29a45;--anatomy-shade-3:#dea652;--anatomy-hi:#7a4e0e;--anatomy-lo:#a8742a;--anatomy-faint:#c48c38}
.lc-tarjeta{fill:var(--anatomy-card)}
.lc-agua{fill:#7fb0d6;fill-opacity:.8;stroke:#4f86b0;stroke-width:.5}
.lc-fondo{fill:#ded9cc}
.lc-rotulo,.lc-valor,.lc-palabra,.lc-prob{font-family:"Inter",ui-sans-serif,system-ui,sans-serif;paint-order:stroke;stroke:#fcfcfa;stroke-width:4px;stroke-linejoin:round;fill:#1b1c18}
.lc-rotulo{font-size:16px;font-weight:700}
.lc-valor{font-size:14px;font-weight:500;fill:#45474c}
.lc-palabra{font-size:17px;font-weight:700;stroke-width:0}
.lc-prob{font-size:16px;font-weight:700}
`;

const etiqueta = "Una celda LSTM dibujada como un canal de agua. El agua es la memoria c. La compuerta de olvido corta el canal principal, la de entrada controla un canal lateral por donde llega lo nuevo de la palabra actual y la de salida controla otro canal lateral que entrega h. Un carro sobre un riel indica la probabilidad entre Biblia y Corán.";
const ultimo = A.documentos[0].palabras.length - 1;
const svg = k.figureSvg({ width: ANCHO, height: ALTO, label: etiqueta, body: dibujo(A.documentos[0], ultimo), standalone: "light" }).replace("<style>", `<style>${CSS}`);
writeFileSync(join(RAIZ, "figures", "lstm", "lstm_celda.svg"), svg);

const datos = {
  resultados: R,
  documentos: A.documentos,
  figura: { svg, sube: SUBE, avanza: AVANZA, altoHoja: ALTO_HOJA, recorrido: RIEL.w - CARRO - 4, escalaAgua: ESCALA_AGUA },
};
writeFileSync(join(RAIZ, "spa", "lstm_datos.js"), `// Generado por scripts/anatomy_lstm/build.mjs. No editar a mano.\nwindow.LSTM_DATOS = ${JSON.stringify(datos)};\n`);
console.log(`escrito figures/lstm/lstm_celda.svg (${(svg.length / 1024).toFixed(0)} KB) y spa/lstm_datos.js`);
