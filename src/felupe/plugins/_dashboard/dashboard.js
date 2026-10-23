// SPDX-License-Identifier: GPL-3.0-or-later
//
// The frontend of the DashboardWidgetPlugin of FElupe (an anywidget ES module). It
// renders the deformed model, the load path, the solver figures and the performance
// statistics from the synchronized traits of the widget. The model is drawn by the
// WebGL viewer of the plotting backend "webgl" (createViewer of viewer.js, which is
// placed before this module). All controls are evaluated in the frontend, i.e. they
// work while the kernel is busy with the job.

// the number of the bands of the colors of the field and the colors of the model
const BANDS = 10;
const MODEL_COLOR = "#add8e6";
const EDGE_COLOR = "#1b2631";
const PALETTE = {
  light: { line: "#2f5d8a", amber: "#b07400", fail: "#b3261e", well: "#e8ecef", soft: "#f1d48a" },
  dark: { line: "#7fb0dd", amber: "#e0a526", fail: "#ef6b62", well: "#2a333b", soft: "#5e4a14" },
};

// ------------------------------------------------------------------- utilities

// typed arrays of the binary buffers of the widget, without a copy if the buffer is
// aligned to 4 bytes (otherwise, e.g. for a buffer at an odd offset of a message, the
// bytes are copied)
function typed(value, Type) {
  if (value instanceof ArrayBuffer) return new Type(value);
  if (!ArrayBuffer.isView(value)) return new Type(0);
  const { buffer, byteOffset, byteLength } = value;
  if (byteOffset % 4 === 0) return new Type(buffer, byteOffset, byteLength / 4);
  return new Type(buffer.slice(byteOffset, byteOffset + byteLength));
}
const f32 = (value) => typed(value, Float32Array);
const u32 = (value) => typed(value, Uint32Array);

// an array of the scene of the viewer
const item = (data, dtype) => ({ dtype, size: data.length, data });

function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) {
    const [m, e] = v.toExponential(digits - 1).split("e");
    return m + "e" + (e[0] === "-" ? "-" : "+") + e.slice(1).padStart(2, "0");
  }
  return Number(v.toPrecision(digits + 1)).toString();
}
const int = (v) => (v === null || v === undefined ? "n/a" : Number(v).toLocaleString("en-US"));
function secs(t) {
  if (t === null || t === undefined) return "n/a";
  if (t < 1e-3) return (t * 1e6).toFixed(0) + " µs";
  if (t < 1) return (t * 1e3).toFixed(1) + " ms";
  if (t < 120) return t.toFixed(2) + " s";
  return clock(t);
}
function clock(t) {
  const m = Math.floor(t / 60);
  const s = t - 60 * m;
  return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
}
function bytes(b) {
  if (!b) return "n/a";
  const u = ["B", "kB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) {
    b /= 1024;
    i++;
  }
  return b.toFixed(i ? 1 : 0) + " " + u[i];
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, h];
}

// the values of a line segment (start and end) of each line cell
function atLines(array, lines, size) {
  const result = new Float32Array(size * lines.length);
  for (let s = 0; s < lines.length; s++) {
    for (let k = 0; k < size; k++) result[size * s + k] = array[size * lines[s] + k];
  }
  return result;
}

// ---------------------------------------------------------------------- markup

const MARKUP = `
<div class="fd-head">
  <span class="fd-brand">FElupe <span class="fd-version"></span></span>
  <span class="fd-title"></span>
  <span class="fd-status" data-s="waiting">Waiting for the job</span>
  <span class="fd-pct">0 %</span>
  <span class="fd-clock">0:00.0</span>
</div>
<div class="fd-where">The dashboard appears when the job starts.</div>
<canvas class="fd-path" aria-label="Load path, one cell for each substep"></canvas>
<div class="fd-main">
  <div class="fd-left">
    <div class="fd-view">
      <div class="fd-caption"></div>
    </div>
    <div class="fd-tools">
      <label>Scale <input class="fd-scale" type="number" value="1" step="0.5" min="0"></label>
      <label><input class="fd-edges" type="checkbox" checked> Mesh lines</label>
      <span class="fd-views" role="group" aria-label="View">
        <button type="button" data-view="fit">Fit</button>
        <button type="button" data-view="xy">xy</button>
        <button type="button" data-view="xz">xz</button>
        <button type="button" data-view="yz">yz</button>
        <button type="button" data-view="iso">iso</button>
      </span>
    </div>
  </div>
  <div class="fd-side">
    <div class="fd-kv fd-solver"></div>
    <canvas class="fd-spark" aria-label="Residual norms of the attempts of the current substep"></canvas>
    <div class="fd-kv fd-counts"></div>
    <div class="fd-kv fd-model"></div>
  </div>
</div>
<div class="fd-foot">
  <div class="fd-split" aria-hidden="true"></div>
  <div class="fd-perf"></div>
  <div class="fd-event"></div>
</div>`;

// ---------------------------------------------------------------------- widget

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "fd";
  root.innerHTML = MARKUP;
  el.appendChild(root);
  const $ = (selector) => root.querySelector(selector);

  const ui = { scale: 1, edges: true };
  let geo = null; // the geometry of the model
  let points = null; // the buffer of the points of the last geometry (a new job)
  let frame = null; // the decoded field values of the latest converged substep
  let resetView = true; // the view is reset for the geometry of a new job
  let received = performance.now();
  let pal = PALETTE.light;
  let pending = { view: true, path: true, side: true };
  let raf = 0;

  // the model is drawn by the WebGL viewer (the wheel zooms only after a click into
  // the view, otherwise it scrolls the notebook)
  const viewer = createViewer($(".fd-view"), { objects: [] }, { wheelOnFocus: true });

  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  // canvas fonts do not resolve CSS variables
  const font = (size) => `${size}px ${css("--fd-font") || "system-ui, sans-serif"}`;

  function theme() {
    // a light text color indicates a dark theme of the editor
    const m = getComputedStyle(root).color.match(/\d+(\.\d+)?/g);
    const rgb = m ? m.slice(0, 3).map(Number) : [0, 0, 0];
    const dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] > 140;
    pal = dark ? PALETTE.dark : PALETTE.light;
    root.classList.toggle("fd-dark", dark);
  }

  function schedule(part) {
    pending[part] = true;
    if (!raf) raf = requestAnimationFrame(flush);
  }
  function flush() {
    raf = 0;
    theme();
    if (pending.side) {
      pending.side = false;
      updateSide();
    }
    if (pending.path) {
      pending.path = false;
      drawPath();
    }
    if (pending.view) {
      pending.view = false;
      drawModel();
    }
  }

  // ------------------------------------------------------------ model state

  function buildGeometry() {
    const info = model.get("info") || {};
    const P = f32(model.get("points"));
    // the points of the surface (info.npoints is the number of points of the mesh)
    const n = Math.floor(P.length / 3);
    geo = n
      ? {
          info, n, P,
          triangles: u32(model.get("triangles")),
          faces: u32(model.get("faces")),
          edges: u32(model.get("edges")),
          lines: u32(model.get("lines")),
        }
      : null;

    $(".fd-scale").disabled = !info.deform;

    // the view is reset for the geometry of a new job (not for an update of info)
    if (model.get("points") !== points) {
      points = model.get("points");
      resetView = true;
    }
    $(".fd-title").textContent = info.title || "";
    $(".fd-version").textContent = info.version || "";
  }

  function readFrame() {
    const fr = model.get("frame") || {};
    frame = {
      kind: fr.kind,
      step: fr.step,
      substep: fr.substep,
      values: f32(fr.values),
      limits: fr.limits || null,
    };
  }

  // --------------------------------------------------------------- viewport

  function scene() {
    "The scene of the viewer: the model, deformed and colored by the field.";
    if (!geo) return { objects: [], axes: false };

    const { info, n, P, triangles, faces, edges, lines } = geo;

    // the norm of the values of the first field of the unknowns, which also deform
    // the model (if the field has as many components as the points)
    const fdim = info.field_dim || 0;
    const F = fdim && frame.values.length >= fdim * n ? frame.values : null;
    let values = null;
    let displacement = null;
    if (F) {
      values = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let q = 0; q < fdim; q++) s += F[fdim * i + q] * F[fdim * i + q];
        values[i] = fdim === 1 ? F[i] : Math.sqrt(s);
      }
      if (info.deform) {
        displacement = new Float32Array(3 * n);
        for (let i = 0; i < n; i++) {
          for (let k = 0; k < info.dim; k++) displacement[3 * i + k] = F[fdim * i + k];
        }
      }
    }

    // the limits of the legend are evaluated by the values at all points (also inside)
    let range = frame.limits;
    if (values && !range) {
      let lo = Infinity;
      let hi = -Infinity;
      for (const v of values) {
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
      range = lo <= hi ? [lo, hi] : null;
    }

    const deform = displacement ? { displacement: item(displacement, "f4"), factor: ui.scale } : {};
    const base = { positions: item(P, "f4"), ...deform };
    const colors = values
      ? { values: item(values, "f4"), range, bands: BANDS, label: fdim > 1 ? "Field norm" : "Field", verticalBar: true }
      : {};

    const objects = [];
    if (triangles.length) {
      // the faces of linear cells are shaded by a common normal (like polygons)
      const shading = faces.length ? { faceIds: item(faces, "u4") } : { smooth: true, sharp: true };
      objects.push({ type: "surface", ...base, triangles: item(triangles, "u4"), color: MODEL_COLOR, ...shading, ...colors });
    }
    if (ui.edges && edges.length) {
      objects.push({ type: "surface", ...base, faces: false, edges: item(edges, "u4"), edgeColor: EDGE_COLOR, opacity: 0.45 });
    }
    if (lines.length) {
      // the line cells (e.g. of a truss), colored by the values at their points
      objects.push({
        type: "lines",
        positions: item(atLines(P, lines, 3), "f4"),
        ...(displacement ? { displacement: item(atLines(displacement, lines, 3), "f4"), factor: ui.scale } : {}),
        color: MODEL_COLOR,
        width: 3,
        ...colors,
        ...(values ? { values: item(atLines(values, lines, 1), "f4") } : {}),
      });
    }
    if (!triangles.length && !edges.length && !lines.length) {
      // the points of a mesh without supported cells
      objects.push({ type: "points", ...base, color: MODEL_COLOR, size: 5, ...colors });
    }
    return { objects, view: "default", axes: true, legend: null };
  }

  function drawModel() {
    viewer.update(scene(), { keepView: !resetView });
    resetView = false;

    let text = "Initial state";
    if (frame.kind === "substep") {
      const nsub = ((geo && geo.info.nsubsteps) || [])[frame.step];
      text = `Step ${frame.step + 1}, substep ${frame.substep + 1} of ${nsub}`;
    }
    if (geo && geo.info.deform && ui.scale !== 1) text += `, deformation scaled by ${ui.scale}`;
    $(".fd-caption").textContent = geo ? text : "";
  }

  // -------------------------------------------------------------- load path

  function drawPath() {
    const canvas = $(".fd-path");
    const [ctx, W, H] = fitCanvas(canvas);
    ctx.clearRect(0, 0, W, H);
    const info = model.get("info") || {};
    const path = model.get("path") || {};
    const nsub = info.nsubsteps || [];
    const status = path.status || "";
    const total = nsub.reduce((a, b) => a + b, 0);
    if (!total) return;
    const lf = path.load_factors || {};
    const gap = nsub.length > 1 ? 4 : 0;
    const avail = W - gap * (nsub.length - 1);
    const top = 9;
    const bh = H - top - 1;
    const color = { c: pal.line, b: pal.amber, f: pal.fail, r: pal.soft };
    let x = 0;
    let index = 0;
    nsub.forEach((count) => {
      const w = (avail * count) / total;
      const cw = w / count;
      ctx.fillStyle = pal.well;
      ctx.fillRect(x, top, w, bh);
      for (let i = 0; i < count; i++, index++) {
        const code = status[index] || ".";
        const x0 = x + i * cw;
        if (code !== ".") {
          ctx.fillStyle = color[code];
          ctx.fillRect(x0, top, Math.max(1, cw - (cw > 4 ? 1 : 0)), bh);
          if (code === "r") {
            ctx.strokeStyle = pal.line;
            ctx.lineWidth = 1.5;
            ctx.strokeRect(x0 + 0.75, top + 0.75, Math.max(1, cw - 1.5), bh - 1.5);
          }
        }
        const marks = lf[String(index)];
        if (marks && marks.length > 1) {
          ctx.strokeStyle = pal.amber;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x0, top - 3.5);
          ctx.lineTo(x0 + cw, top - 3.5);
          for (const t of marks) {
            ctx.moveTo(x0 + t * cw, top - 7);
            ctx.lineTo(x0 + t * cw, top);
          }
          ctx.stroke();
        }
      }
      x += w + gap;
    });
  }

  // ---------------------------------------------------------- side and foot

  // rows of keys and values (no dl-elements, which are styled by the frontends)
  function kv(box, rows) {
    box.innerHTML = "";
    for (const [key, value, big] of rows) {
      const row = document.createElement("div");
      const k = document.createElement("span");
      const v = document.createElement("span");
      row.className = big ? "fd-row fd-big" : "fd-row";
      k.textContent = key;
      v.textContent = value;
      row.append(k, v);
      box.appendChild(row);
    }
  }

  function updateSide() {
    const info = model.get("info") || {};
    const p = model.get("progress") || {};
    const nsub = info.nsubsteps || [];
    const total = nsub.reduce((a, b) => a + b, 0);
    const state = p.status || "waiting";
    const label = { waiting: "Waiting for the job", running: "Running", finished: "Finished", failed: "Failed" };
    const status = $(".fd-status");
    status.textContent = label[state] || state;
    status.dataset.s = state;
    $(".fd-pct").textContent = total ? `${Math.floor((100 * (p.converged || 0)) / total)} %` : "0 %";

    if (p.step !== null && p.step !== undefined && p.substep !== null && p.substep !== undefined) {
      const where = `step ${p.step + 1} of ${nsub.length}, substep ${p.substep + 1} of ${nsub[p.step]}`;
      const text = {
        running: `Running ${where}.`,
        finished: `Finished, ${p.converged} of ${total} substeps converged.`,
        failed: `Failed at ${where}. ${p.converged} of ${total} substeps converged.`,
      }[state];
      $(".fd-where").textContent = text || "";
    } else if (info.npoints) {
      $(".fd-where").textContent = "Starting the job.";
    }

    kv($(".fd-solver"), [
      ["Residual norm", fmt(p.fnorm), true],
      ["Tolerance", fmt(p.tol)],
      ["Attempt, iteration", p.attempt ? `${p.attempt}, ${p.iteration}` : "n/a"],
    ]);
    kv($(".fd-counts"), [
      ["Iterations", int(p.iterations)],
      ["Substeps with cutbacks", int(p.cutbacks)],
      ["Failed attempts", int(p.failed_attempts)],
    ]);
    let quadrature = info.quadrature || "n/a";
    if (info.quadrature && info.quadrature_points_per_cell) {
      quadrature += `, ${info.quadrature_points_per_cell} points`;
    }
    kv($(".fd-model"), [
      ["Points", int(info.npoints)],
      ["Cells", `${int(info.ncells)} ${info.cell_type || ""}`],
      ["Element", info.element || "n/a"],
      ["Quadrature", quadrature],
      ["Dofs", int(info.ndof)],
      ["Active dofs", int(p.active_dofs)],
    ]);
    drawSpark();

    // performance: the split of the elapsed wall time
    const T = p.timing || { assembly: 0, solve: 0, residual: 0 };
    const elapsed = p.elapsed || 0;
    const parts = [
      ["Assembly", T.assembly, pal.line],
      ["solve", T.solve, pal.amber],
      ["residual", T.residual, "var(--fd-ink-2)"],
      ["other", Math.max(0, elapsed - T.assembly - T.solve - T.residual), pal.well],
    ];
    const sum = parts.reduce((a, q) => a + q[1], 0) || 1;
    const split = $(".fd-split");
    split.innerHTML = "";
    for (const [, value, color] of parts) {
      const div = document.createElement("div");
      div.style.width = `${((100 * value) / sum).toFixed(2)}%`;
      div.style.background = color;
      split.appendChild(div);
    }
    const perIteration = p.iterations ? secs((T.assembly + T.solve + T.residual) / p.iterations) : "n/a";
    $(".fd-perf").textContent =
      parts.map(([name, value]) => `${name} ${Math.round((100 * value) / sum)} %`).join(", ") +
      `. ${perIteration} per iteration, peak memory ${bytes(p.memory)}.`;
    const event = $(".fd-event");
    event.textContent = p.event ? p.event.text : "";
    event.dataset.kind = p.event ? p.event.kind : "";
  }

  function drawSpark() {
    const canvas = $(".fd-spark");
    const [ctx, W, H] = fitCanvas(canvas);
    ctx.clearRect(0, 0, W, H);
    const newton = model.get("newton") || {};
    const attempts = newton.attempts || [];
    const values = [];
    for (const a of attempts) for (const v of a.fnorms) if (v > 0) values.push(v);
    if (newton.tol > 0) values.push(newton.tol);
    const ink = css("--fd-ink-2") || "#666";
    ctx.font = font(10);
    ctx.fillStyle = ink;
    if (!values.length) {
      ctx.fillText("Residual norms of the substep", 0, H / 2);
      return;
    }
    let ymin = Math.floor(Math.log10(Math.min(...values)));
    let ymax = Math.ceil(Math.log10(Math.max(...values)));
    if (ymax <= ymin) ymax = ymin + 1;
    const pad = 30;
    const count = attempts.reduce((a, x) => a + x.fnorms.length, 0);
    const X = (i) => pad + ((W - pad - 4) * i) / Math.max(1, count - 1);
    const Y = (v) => 3 + ((H - 6) * (ymax - Math.log10(v))) / (ymax - ymin);
    ctx.textBaseline = "top";
    ctx.fillText("1e" + ymax, 0, 0);
    ctx.textBaseline = "bottom";
    ctx.fillText("1e" + ymin, 0, H);
    if (newton.tol > 0) {
      ctx.strokeStyle = ink;
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, Y(newton.tol));
      ctx.lineTo(W, Y(newton.tol));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    let i0 = 0;
    attempts.forEach((a, n) => {
      const color = a.failed ? pal.fail : n > 0 ? pal.amber : pal.line;
      ctx.strokeStyle = ctx.fillStyle = color;
      ctx.lineWidth = 1.4;
      ctx.setLineDash(a.failed ? [3, 2] : []);
      ctx.beginPath();
      let started = false;
      a.fnorms.forEach((v, j) => {
        if (!(v > 0)) return;
        const x = X(i0 + j);
        const y = Y(v);
        if (!started) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        started = true;
      });
      ctx.stroke();
      ctx.setLineDash([]);
      const last = a.fnorms[a.fnorms.length - 1];
      if (last > 0) ctx.fillRect(X(i0 + a.fnorms.length - 1) - 1.5, Y(last) - 1.5, 3, 3);
      i0 += a.fnorms.length;
    });
  }

  // ------------------------------------------------------------------ clock

  function tick() {
    const p = model.get("progress") || {};
    let t = p.elapsed || 0;
    if (p.status === "running") t += (performance.now() - received) / 1000;
    $(".fd-clock").textContent = clock(t);
  }
  const timer = setInterval(tick, 100);

  // --------------------------------------------------------------- controls

  $(".fd-scale").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    ui.scale = Number.isFinite(v) ? v : 0;
    schedule("view");
  });
  $(".fd-edges").addEventListener("change", (e) => {
    ui.edges = e.target.checked;
    schedule("view");
  });
  // the views of the camera ("fit" keeps the rotation, "iso" is the default view of
  // the plots of FElupe with the z-axis upwards)
  const VIEW_NAMES = { fit: null, xy: "xy", xz: "xz", yz: "yz", iso: "default3d" };
  root.querySelectorAll("[data-view]").forEach((button) =>
    button.addEventListener("click", () => viewer.setView(VIEW_NAMES[button.dataset.view])),
  );

  // ----------------------------------------------------------- model events

  const onGeometry = () => {
    buildGeometry();
    schedule("view");
    schedule("path");
    schedule("side");
  };
  const onFrame = () => {
    readFrame();
    schedule("view");
  };
  const onPath = () => schedule("path");
  const onProgress = () => {
    received = performance.now();
    schedule("side");
  };
  const handlers = [
    ["change:info", onGeometry],
    ["change:points", onGeometry],
    ["change:triangles", onGeometry],
    ["change:quads", onGeometry],
    ["change:lines", onGeometry],
    ["change:frame", onFrame],
    ["change:path", onPath],
    ["change:progress", onProgress],
    ["change:newton", onProgress],
  ];
  for (const [event, handler] of handlers) model.on(event, handler);

  // the viewer is resized by itself
  const resize = new ResizeObserver(() => {
    schedule("path");
    schedule("side");
  });
  resize.observe(root);

  buildGeometry();
  readFrame();
  flush();

  return () => {
    for (const [event, handler] of handlers) model.off(event, handler);
    resize.disconnect();
    clearInterval(timer);
    if (raf) cancelAnimationFrame(raf);
    viewer.dispose();
  };
}

export default { render };
