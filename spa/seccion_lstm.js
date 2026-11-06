// Sección 5 de la demo: clasificador LSTM. Los datos llegan en spa/lstm_datos.js como
// script normal, así que la sección también funciona al abrir el HTML con doble clic.
(function () {
  const COLOR = { biblia: "#1e3a8a", coran: "#059669", serieA: "#b45309", serieB: "#6d4fc4", tinta: "#45474c" };
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip-lstm";
  document.body.appendChild(tooltip);

  const fmt = (v, d = 3) => Number(v).toLocaleString("es-ES", { minimumFractionDigits: d, maximumFractionDigits: d, useGrouping: "always" });
  const entero = (v) => fmt(v, 0);
  const firmado = (v, d = 3) => (v > 0 ? "+" : "") + fmt(v, d);
  const poner = (id, html) => { document.getElementById(id).innerHTML = html; };
  const media = (v) => v.reduce((a, b) => a + b, 0) / v.length;

  function mostrarTooltip(evento, html) {
    tooltip.innerHTML = html;
    tooltip.classList.add("visible");
    const margen = 14;
    const ancho = tooltip.offsetWidth;
    const x = evento.clientX + margen + ancho > innerWidth ? evento.clientX - ancho - margen : evento.clientX + margen;
    tooltip.style.left = Math.max(4, x) + "px";
    tooltip.style.top = Math.max(4, evento.clientY - tooltip.offsetHeight - margen) + "px";
  }
  const ocultarTooltip = () => tooltip.classList.remove("visible");

  function leyenda(id, entradas) {
    poner(id, entradas.map(([color, texto, bloque]) => `<span><i class="${bloque ? "bloque" : ""}" style="background:${color}"></i>${texto}</span>`).join(""));
  }

  function lienzo(selector, margen) {
    const svg = d3.select(selector);
    svg.selectAll("*").remove();
    const caja = svg.node().getBoundingClientRect();
    svg.attr("viewBox", `0 0 ${caja.width} ${caja.height}`);
    return { svg, ancho: caja.width - margen.left - margen.right, alto: caja.height - margen.top - margen.bottom, g: svg.append("g").attr("transform", `translate(${margen.left},${margen.top})`) };
  }

  function lineas(selector, op) {
    const margen = { top: 16, right: 24, bottom: 38, left: 48 };
    const { svg, ancho, alto, g } = lienzo(selector, margen);
    const x = op.escalaX().domain(op.dominioX).range([0, ancho]);
    const y = d3.scaleLinear().domain(op.dominioY).range([alto, 0]);
    g.append("g").attr("class", "rejilla-eje").call(d3.axisLeft(y).ticks(5).tickSize(-ancho).tickFormat(""));
    g.append("g").attr("class", "eje-lstm").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).tickValues(op.marcasX).tickFormat((v) => entero(v)));
    g.append("g").attr("class", "eje-lstm").call(d3.axisLeft(y).ticks(5).tickFormat((v) => fmt(v, op.decimalesY)));
    g.append("text").attr("x", ancho / 2).attr("y", alto + 32).attr("text-anchor", "middle").text(op.tituloX);
    if (op.marca !== undefined) {
      g.append("line").attr("x1", x(op.marca)).attr("x2", x(op.marca)).attr("y1", 0).attr("y2", alto).attr("stroke", COLOR.tinta);
      g.append("text").attr("x", x(op.marca) + 5).attr("y", 10).text(op.textoMarca);
    }
    const linea = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(d3.curveMonotoneX);
    op.series.forEach((s) => {
      g.append("path").datum(s.valores).attr("d", linea).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", 2).attr("stroke-linejoin", "round").attr("stroke-linecap", "round");
      if (op.puntos) g.selectAll(null).data(s.valores).join("circle").attr("cx", (d) => x(d[0])).attr("cy", (d) => y(d[1])).attr("r", 4).attr("fill", s.color).attr("stroke", "#ffffff").attr("stroke-width", 2);
    });
    const cruz = g.append("line").attr("y1", 0).attr("y2", alto).attr("stroke", COLOR.tinta).attr("opacity", 0);
    const valoresX = op.series[0].valores.map((d) => d[0]);
    svg.append("rect").attr("x", margen.left).attr("y", margen.top).attr("width", ancho).attr("height", alto).attr("fill", "transparent")
      .on("pointermove", function (e) {
        const [px] = d3.pointer(e, this);
        const buscado = x.invert(px - margen.left);
        const n = d3.leastIndex(valoresX, (v) => Math.abs(v - buscado));
        cruz.attr("x1", x(valoresX[n])).attr("x2", x(valoresX[n])).attr("opacity", 1);
        mostrarTooltip(e, `<strong>${op.tituloX}: ${entero(valoresX[n])}</strong><br>` + op.series.map((s) => `${s.nombre}: ${fmt(s.valores[n][1], op.decimalesTooltip)}`).join("<br>"));
      })
      .on("pointerleave", () => { cruz.attr("opacity", 0); ocultarTooltip(); });
  }

  function dibujarCurvas(R) {
    const H = R.historia;
    lineas("#lstm-perdida", {
      escalaX: d3.scaleLinear, dominioX: [1, H.length], marcasX: [1, 10, 20, 30, 40].filter((v) => v <= H.length),
      dominioY: [0, d3.max(H, (h) => Math.max(h.train_loss, h.val_loss)) * 1.05], decimalesY: 1, decimalesTooltip: 4,
      tituloX: "Época", marca: R.mejor_epoca, textoMarca: `pesos elegidos, época ${R.mejor_epoca}`,
      series: [
        { nombre: "Entrenamiento", color: COLOR.serieA, valores: H.map((h) => [h.epoca, h.train_loss]) },
        { nombre: "Validación", color: COLOR.serieB, valores: H.map((h) => [h.epoca, h.val_loss]) },
      ],
    });
    lineas("#lstm-largo", {
      escalaX: d3.scaleLog, dominioX: [10, 200], marcasX: R.por_largo.map((f) => f.largo), dominioY: [0.6, 1], puntos: true,
      decimalesY: 1, decimalesTooltip: 3, tituloX: "Palabras por documento",
      series: [
        { nombre: "LSTM", color: COLOR.serieA, valores: R.por_largo.map((f) => [f.largo, f.lstm.balanced_accuracy]) },
        { nombre: "TF-IDF + regresión logística", color: COLOR.serieB, valores: R.por_largo.map((f) => [f.largo, f.base.balanced_accuracy]) },
      ],
    });
  }

  function dibujarTop(selector, filas) {
    const margen = { top: 8, right: 22, bottom: 42, left: 78 };
    const { g, ancho, alto } = lienzo(selector, margen);
    const extremo = d3.max(filas, (f) => Math.abs(f.contribucion)) * 1.1;
    const x = d3.scaleLinear().domain([-extremo, extremo]).range([0, ancho]);
    const y = d3.scaleBand().domain(filas.map((f) => f.palabra)).range([0, alto]).padding(0.28);
    g.append("g").attr("class", "rejilla-eje").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).ticks(5).tickSize(-alto).tickFormat(""));
    g.append("g").attr("class", "eje-lstm").attr("transform", `translate(0,${alto})`).call(d3.axisBottom(x).ticks(5).tickFormat((v) => firmado(v, 1)));
    g.append("text").attr("x", ancho / 2).attr("y", alto + 34).attr("text-anchor", "middle").text("Contribución media al logit");
    const grosor = Math.min(y.bandwidth(), 24);
    const barra = (f) => {
      const x0 = x(0), x1 = x(f.contribucion), s = x1 >= x0 ? 1 : -1, r = Math.min(4, Math.abs(x1 - x0)), t = y(f.palabra) + (y.bandwidth() - grosor) / 2;
      return `M${x0} ${t}H${x1 - s * r}Q${x1} ${t} ${x1} ${t + r}V${t + grosor - r}Q${x1} ${t + grosor} ${x1 - s * r} ${t + grosor}H${x0}Z`;
    };
    g.selectAll(null).data(filas).join("path").attr("d", barra).attr("fill", (f) => (f.contribucion > 0 ? COLOR.coran : COLOR.biblia));
    g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", alto).attr("stroke", COLOR.tinta);
    g.selectAll(null).data(filas).join("text").attr("x", -8).attr("y", (f) => y(f.palabra) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").text((f) => f.palabra);
    g.selectAll(null).data(filas).join("rect").attr("x", -margen.left).attr("width", ancho + margen.left).attr("y", (f) => y(f.palabra)).attr("height", y.step()).attr("fill", "transparent")
      .on("pointermove", (e, f) => mostrarTooltip(e, `<strong>${f.palabra}</strong><br>Contribución: ${firmado(f.contribucion)} (IC 95 %: ${firmado(f.ic95[0])} a ${firmado(f.ic95[1])})<br>Apariciones: ${entero(f.n_biblia)} en Biblia, ${entero(f.n_coran)} en Corán<br>Estabilidad en el top 20: ${fmt(f.estabilidad * 100, 0)} %`))
      .on("pointerleave", ocultarTooltip);
  }

  const tablaTop = (filas) => `<div class="tabla-lstm-marco"><table class="tabla-lstm"><thead><tr><th>Palabra</th><th>Contribución</th><th>IC 95 %</th><th>Biblia</th><th>Corán</th><th>Estabilidad</th></tr></thead><tbody>${filas.map((f) => `<tr><td>${f.palabra}</td><td>${firmado(f.contribucion)}</td><td>${firmado(f.ic95[0])} a ${firmado(f.ic95[1])}</td><td>${entero(f.n_biblia)}</td><td>${entero(f.n_coran)}</td><td>${fmt(f.estabilidad * 100, 0)} %</td></tr>`).join("")}</tbody></table></div>`;

  function pintarMetricas(R) {
    const P = R.prueba, E = R.entorno, D = R.datos;
    const parametros = Object.values(R.parametros).reduce((a, b) => a + b, 0);
    poner("lstm-stats", [
      [fmt(P.accuracy), "exactitud"], [fmt(P.balanced_accuracy), "exactitud balanceada"], [fmt(P.precision_coran), "precisión (Corán)"], [fmt(P.recall_coran), "recall (Corán)"],
      [fmt(P.f1_coran), "F1 (Corán)"], [fmt(P.roc_auc), "ROC-AUC"], [fmt(P.pr_auc), "PR-AUC"], [fmt(P.log_loss, 4), "log-loss"], [fmt(P.brier, 4), "Brier"],
    ].map(([v, t]) => `<div class="stat"><strong>${v}</strong>${t}</div>`).join(""));
    const c = P.confusion;
    poner("lstm-confusion", `<div class="tabla-lstm-marco"><table class="tabla-lstm confusion-lstm"><thead><tr><th>Real</th><th>Predice Biblia</th><th>Predice Corán</th></tr></thead><tbody><tr><td>Biblia</td><td class="acierto">${entero(c.tn)}</td><td>${entero(c.fp)}</td></tr><tr><td>Corán</td><td>${entero(c.fn)}</td><td class="acierto">${entero(c.tp)}</td></tr></tbody></table></div>`);
    poner("lstm-dato-datos", `${entero(P.n)} documentos de prueba de hasta ${entero(D.ventana)} palabras. Entrenamiento de ${fmt(E.segundos_entrenamiento, 1)} s en la GPU (${R.historia.length} épocas), ${entero(parametros)} parámetros.`);
    poner("lstm-dato-cero", `El resultado perfecto se debe a que con ${entero(D.ventana)} palabras la tarea es fácil: la línea base de TF-IDF con regresión logística también llega a ${fmt(R.por_largo[R.por_largo.length - 1].base.balanced_accuracy)}. Con ${entero(P.n_coran)} documentos del Corán sin errores, la tasa real de error puede llegar a ${fmt(3 / P.n_coran * 100, 1)} % (regla del tres, 95 %).`);
    const L20 = R.por_largo.find((f) => f.largo === 20);
    poner("lstm-dato-largo", `La LSTM aporta en fragmentos cortos: con 20 palabras logra ${fmt(L20.lstm.balanced_accuracy)} frente a ${fmt(L20.base.balanced_accuracy)}.`);
  }

  function pintarExplicacion(R) {
    const S = R.shap;
    const a200 = R.fidelidad.find((f) => f.largo === 200), a20 = R.fidelidad.find((f) => f.largo === 20);
    poner("lstm-tabla-top", tablaTop(S.top));
    poner("lstm-tabla-contenido", tablaTop(S.top_sin_stopwords));
    poner("lstm-dato-shap", `Valores de Shapley sobre ${entero(S.documentos)} documentos de prueba con clases balanceadas. Al quitar el top 20 con todas las palabras, la exactitud balanceada con 200 palabras baja de ${fmt(a200.sin_mascara.balanced_accuracy)} a ${fmt(a200.top20.balanced_accuracy)}. Al quitar el top 20 sin stopwords se mantiene en ${fmt(a200.top20_sin_stopwords.balanced_accuracy)}, y solo se nota en fragmentos cortos: con 20 palabras baja de ${fmt(a20.sin_mascara.balanced_accuracy)} a ${fmt(a20.top20_sin_stopwords.balanced_accuracy)}. Las palabras funcionales pesan más que el vocabulario temático. Parte de esa señal puede ser el estilo de cada traducción, así que el resultado queda sin comprobar con otras versiones de estos textos.`);
    const escala = d3.quantile(R.ejemplos.flatMap((e) => e.shap.map(Math.abs)).sort(d3.ascending), 0.98);
    poner("lstm-ejemplos", R.ejemplos.slice(0, 2).map((e) => {
      const marcas = e.palabras.map((p, t) => {
        const v = e.shap[t], alfa = (Math.min(1, Math.abs(v) / escala) * 0.6).toFixed(2);
        return `<mark style="background:${v > 0 ? `rgba(5,150,105,${alfa})` : `rgba(30,58,138,${alfa})`}" title="${firmado(v)}">${p}</mark>`;
      }).join(" ");
      return `<div class="doc-lstm"><h4><span>${e.real === "coran" ? "Corán" : "Biblia"}</span><span>P(Corán) = ${fmt(e.p_coran, 4)}</span></h4><p>${marcas} …</p></div>`;
    }).join(""));
  }

  function pintarTeoria(R) {
    const D = R.datos;
    poner("lstm-dato-ecuaciones", `Estas ecuaciones son las que calcula el modelo: recorridas a mano con los pesos entrenados, difieren de <code>nn.LSTM</code> en ${R.error_ecuaciones.toExponential(1).replace(".", ",")} como máximo.`);
    poner("lstm-dato-previa", `La pérdida no lleva pesos por clase, así que \\(\\hat p\\) incorpora la proporción de entrenamiento (${D.reparto.train.coran} documentos del Corán y ${D.reparto.train.biblia} de la Biblia). Para comparar los textos en igualdad de condiciones: \\(\\ell_{50/50} = \\ell - \\ln(${D.reparto.train.coran}/${D.reparto.train.biblia}) = \\ell + ${fmt(-Math.log(D.reparto.train.coran / D.reparto.train.biblia), 2).replace(",", "{,}")}\\).`);
  }

  // ---------- Figura de la celda: compuertas y clasificación, palabra por palabra ----------
  function iniciarCelda(D) {
    const fig = D.figura;
    const marco = document.getElementById("lstm-figura");
    marco.innerHTML = fig.svg;
    const svg = marco.querySelector("svg");
    const sel = (a) => svg.querySelector(a);
    const trasladar = (v, n) => `translate(${(v[0] * n).toFixed(2)} ${(v[1] * n).toFixed(2)})`;
    const nivel = (m) => Math.min(13, 1 + m * fig.escalaAgua);
    const quieto = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const actual = {}; // valores animados
    let doc = 0, paso = 0, pausa = quieto, objetivo = {};

    function valores(d, t) {
      const c = d.celda;
      return {
        f: media(c.f[t]), i: media(c.i[t]), o: media(c.o[t]), p: d.p_coran[t],
        previa: c.memoria_previa[t], retenida: c.memoria_retenida[t], nueva: c.memoria_nueva[t], memoria: c.memoria[t], salida: c.salida[t],
      };
    }

    function pintar(v) {
      for (const g of ["f", "i", "o"]) sel(`[data-compuerta="${g}"]`).setAttribute("transform", trasladar(fig.sube, v[g] * fig.altoHoja));
      for (const a of ["previa", "retenida", "nueva", "memoria", "salida"]) sel(`[data-agua="${a}"]`).setAttribute("transform", trasladar(fig.sube, nivel(v[a])));
      sel("[data-carro]").setAttribute("transform", trasladar(fig.avanza, v.p * fig.recorrido));
    }

    let ultimoCuadro = 0;
    function animar(ahora) {
      const dt = Math.min(50, ahora - ultimoCuadro);
      ultimoCuadro = ahora;
      const alfa = 1 - Math.exp(-dt / 160);
      for (const k of Object.keys(objetivo)) actual[k] += (objetivo[k] - actual[k]) * alfa;
      pintar(actual);
      requestAnimationFrame(animar);
    }

    function mostrar(d, t) {
      doc = d; paso = t;
      const documento = D.documentos[d];
      objetivo = valores(documento, t);
      if (quieto || actual.f === undefined) { Object.assign(actual, objetivo); pintar(actual); }
      sel('[data-texto="palabra"]').textContent = documento.palabras[t];
      for (const g of ["f", "i", "o"]) sel(`[data-texto="${g}"]`).textContent = `${g} = ${fmt(objetivo[g], 2)}`;
      sel('[data-texto="p"]').textContent = `P(Corán) = ${fmt(objetivo.p, 2)}`;
      document.querySelectorAll("[data-fragmento]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.fragmento) === d)));
      poner("lstm-palabras", documento.palabras.map((p, n) => {
        const pc = documento.p_coran[n];
        return `<button type="button" class="palabra-lstm${n === t ? " activa" : ""}" data-n="${n}" title="P(Corán) = ${fmt(pc, 2)}" style="--tono:${pc >= 0.5 ? COLOR.coran : COLOR.biblia};--fuerza:${Math.round(Math.abs(pc - 0.5) * 2 * 100)}%">${p}</button>`;
      }).join(""));
      const nombres = { f: ["Olvido", "qué parte de la memoria se conserva"], i: ["Entrada", "qué parte de lo nuevo se escribe"], o: ["Salida", "qué parte de la memoria se entrega"] };
      poner("lstm-unidades", ["f", "i", "o"].map((g) => `<div class="fila-unidades"><div class="nombre-unidades"><strong>${nombres[g][0]}</strong> <span>${nombres[g][1]}</span></div><div class="valor-unidades">${fmt(objetivo[g], 2)}</div><div class="tira-unidades" role="img" aria-label="Valores de las 64 unidades de la compuerta de ${nombres[g][0].toLowerCase()}">${documento.celda[g][t].map((u) => `<i style="opacity:${Math.max(0.04, u).toFixed(2)}" title="${fmt(u, 2)}"></i>`).join("")}</div></div>`).join(""));
      const clase = objetivo.p >= 0.5 ? "Corán" : "Biblia";
      poner("lstm-clasificacion", `Tras leer ${t + 1} de ${documento.palabras.length} palabras: <strong>${clase}</strong>, P(Corán) = ${fmt(objetivo.p, 2)} y P(Biblia) = ${fmt(1 - objetivo.p, 2)}. El fragmento es ${documento.real === "coran" ? "del Corán" : "de la Biblia"}.`);
    }

    document.getElementById("lstm-palabras").addEventListener("click", (e) => {
      const b = e.target.closest("[data-n]");
      if (b) { detener(); mostrar(doc, Number(b.dataset.n)); }
    });
    document.querySelectorAll("[data-fragmento]").forEach((b) => b.addEventListener("click", () => mostrar(Number(b.dataset.fragmento), 0)));
    const botonPausa = document.getElementById("lstm-pausa");
    function detener() { pausa = true; botonPausa.textContent = "Reproducir"; }
    botonPausa.addEventListener("click", () => {
      pausa = !pausa;
      botonPausa.textContent = pausa ? "Reproducir" : "Pausar";
    });

    mostrar(1, 0);
    if (quieto) { detener(); return; }
    requestAnimationFrame((t) => { ultimoCuadro = t; animar(t); });
    let visible = true;
    new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(marco);
    setInterval(() => {
      if (pausa || !visible || document.hidden) return;
      const largo = D.documentos[doc].palabras.length;
      if (paso + 1 < largo) mostrar(doc, paso + 1);
      else mostrar((doc + 1) % D.documentos.length, 0);
    }, 1500);
  }

  async function iniciar() {
    const D = window.LSTM_DATOS, R = D.resultados;
    leyenda("lstm-leyenda-perdida", [[COLOR.serieA, "Entrenamiento"], [COLOR.serieB, "Validación"]]);
    leyenda("lstm-leyenda-largo", [[COLOR.serieA, "LSTM"], [COLOR.serieB, "TF-IDF + regresión logística"]]);
    for (const id of ["lstm-leyenda-top", "lstm-leyenda-contenido", "lstm-leyenda-ejemplos"]) leyenda(id, [[COLOR.biblia, "Empuja hacia Biblia", true], [COLOR.coran, "Empuja hacia Corán", true]]);
    pintarMetricas(R);
    pintarExplicacion(R);
    pintarTeoria(R);
    iniciarCelda(D);
    const dibujar = () => { dibujarCurvas(R); dibujarTop("#lstm-top20", R.shap.top); dibujarTop("#lstm-top20-contenido", R.shap.top_sin_stopwords); };
    dibujar();
    let pendiente;
    addEventListener("resize", () => { clearTimeout(pendiente); pendiente = setTimeout(dibujar, 150); });
    if (window.MathJax && MathJax.startup) {
      await MathJax.startup.promise;
      await MathJax.typesetPromise([document.getElementById("seccion-lstm")]);
    }
  }

  // MathJax carga con defer: se espera al documento completo para encontrarlo listo.
  const arrancar = () => iniciar().catch((error) => {
    console.error("Error en la sección LSTM:", error);
    poner("lstm-stats", `<div class="stat">No se pudo preparar la sección: ${error.message}</div>`);
  });
  if (document.readyState === "complete") arrancar();
  else addEventListener("load", arrancar);
})();
