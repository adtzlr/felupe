// SPDX-License-Identifier: GPL-3.0-or-later
//
// The frontend of the DashboardWidgetPlugin of FElupe (an anywidget ES module). It
// renders the deformed model, the load path, the solver figures and the performance
// statistics from the synchronized traits of the widget. All controls are evaluated
// in the frontend, i.e. they work while the kernel is busy with the job.

const BANDS = 10;
const PALETTE = {
  light: { line: "#2f5d8a", amber: "#b07400", fail: "#b3261e", well: "#e8ecef", soft: "#f1d48a", view: "#fbfcfc" },
  dark: { line: "#7fb0dd", amber: "#e0a526", fail: "#ef6b62", well: "#2a333b", soft: "#5e4a14", view: "#1b2127" },
};

// ------------------------------------------------------------------- utilities

// typed arrays of the binary buffers of the widget, without a copy if the buffer is
// aligned to 4 bytes (otherwise, e.g. for a buffer at an odd offset of a message, the
// bytes are copied)
function view(value, Type) {
  if (value instanceof ArrayBuffer) return new Type(value);
  if (!ArrayBuffer.isView(value)) return new Type(0);
  const { buffer, byteOffset, byteLength } = value;
  if (byteOffset % 4 === 0) return new Type(buffer, byteOffset, byteLength / 4);
  return new Type(buffer.slice(byteOffset, byteOffset + byteLength));
}
const f32 = (value) => view(value, Float32Array);
const u32 = (value) => view(value, Uint32Array);

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

function turbo(x) {
  x = Math.min(1, Math.max(0, x));
  const r = 0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
  const g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
  const b = 0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
  const c = (v) => 255 * Math.min(1, Math.max(0, v));
  return [c(r), c(g), c(b)];
}
const BAND_COLORS = Array.from({ length: BANDS }, (_, i) => turbo((i + 0.5) / BANDS));

const mul = (A, B) => {
  const C = new Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
  return C;
};
const rotX = (a) => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
const rotY = (a) => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];

// rows of the rotation: screen right, screen up and towards the viewer
const VIEWS = {
  xy: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  xz: [1, 0, 0, 0, 0, 1, 0, -1, 0],
  yz: [0, 1, 0, 0, 0, 1, 1, 0, 0],
  iso: mul(rotX((35.264 * Math.PI) / 180), rotY((-45 * Math.PI) / 180)),
};

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

// clip a polygon (x, y, s) to s >= a, the result is written to CX, CY
const CX = new Float64Array(8);
const CY = new Float64Array(8);
function clipGE(xs, ys, ss, n, a) {
  let m = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ini = ss[i] >= a;
    const inj = ss[j] >= a;
    if (ini) {
      CX[m] = xs[i];
      CY[m] = ys[i];
      m++;
    }
    if (ini !== inj) {
      const t = (a - ss[i]) / (ss[j] - ss[i]);
      CX[m] = xs[i] + t * (xs[j] - xs[i]);
      CY[m] = ys[i] + t * (ys[j] - ys[i]);
      m++;
    }
  }
  return m;
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
      <canvas class="fd-canvas" tabindex="0" aria-label="Deformed model. Drag to rotate, shift-drag to pan, click and use the wheel to zoom, double-click to fit."></canvas>
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
  const cam = { R: VIEWS.iso.slice(), zoom: 1, panx: 0, pany: 0 };
  let geo = null; // the geometry of the model
  let points = null; // the buffer of the points of the last geometry (a new job)
  let frame = null; // the decoded field values of the latest converged substep
  let received = performance.now();
  let pal = PALETTE.light;
  let pending = { view: true, path: true, side: true };
  let raf = 0;

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
      drawView();
    }
  }

  // ------------------------------------------------------------ model state

  function buildGeometry() {
    const info = model.get("info") || {};
    const P = f32(model.get("points"));
    // the points of the surface (info.npoints is the number of points of the mesh)
    const n = Math.floor(P.length / 3);
    if (!n) {
      geo = null;
      return;
    }
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++)
      for (let k = 0; k < 3; k++) {
        const v = P[3 * i + k];
        if (v < lo[k]) lo[k] = v;
        if (v > hi[k]) hi[k] = v;
      }
    const tri = u32(model.get("triangles"));
    const quad = u32(model.get("quads"));
    const nf = tri.length / 3 + quad.length / 4;
    const fstart = new Uint32Array(nf + 1);
    const fidx = new Uint32Array(tri.length + quad.length);
    let f = 0;
    let p = 0;
    for (let i = 0; i < tri.length; i += 3) {
      fstart[f++] = p;
      for (let k = 0; k < 3; k++) fidx[p++] = tri[i + k];
    }
    for (let i = 0; i < quad.length; i += 4) {
      fstart[f++] = p;
      for (let k = 0; k < 4; k++) fidx[p++] = quad[i + k];
    }
    fstart[nf] = p;
    geo = {
      info, n, P, nf, fstart, fidx,
      lines: u32(model.get("lines")),
      center: lo.map((v, k) => 0.5 * (v + hi[k])),
      radius: Math.max(1e-12, 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])),
      buf: {
        VX: new Float32Array(n), VY: new Float32Array(n), VZ: new Float32Array(n),
        SX: new Float32Array(n), SY: new Float32Array(n), S: new Float32Array(n),
        depth: new Float32Array(nf), order: new Uint32Array(nf),
      },
    };

    $(".fd-scale").disabled = !info.deform;

    // the camera is reset for the geometry of a new job (not for an update of info)
    if (model.get("points") !== points) {
      points = model.get("points");
      cam.R = (info.dim === 3 ? VIEWS.iso : VIEWS.xy).slice();
      cam.zoom = 1;
      cam.panx = cam.pany = 0;
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

  function drawView() {
    const canvas = $(".fd-canvas");
    const [ctx, W, H] = fitCanvas(canvas);
    ctx.fillStyle = pal.view;
    ctx.fillRect(0, 0, W, H);
    if (!geo || !frame) return;

    const { info, n, P, buf: B } = geo;
    const R = cam.R;
    const legendW = 64;
    const cx = (W - legendW) / 2 + cam.panx;
    const cy = H / 2 + 8 + cam.pany;
    const c0 = geo.center;
    // the values of the first field of the unknowns, which also deform the model
    const fdim = info.field_dim || 0;
    const F = fdim ? frame.values : null;
    const U = info.deform ? F : null;
    const dim = info.dim;
    let ext = 0;
    let smin = Infinity;
    let smax = -Infinity;
    for (let i = 0; i < n; i++) {
      let x = P[3 * i] - c0[0];
      let y = P[3 * i + 1] - c0[1];
      let z = P[3 * i + 2] - c0[2];
      if (U && U.length >= dim * n) {
        x += ui.scale * U[dim * i];
        if (dim > 1) y += ui.scale * U[dim * i + 1];
        if (dim > 2) z += ui.scale * U[dim * i + 2];
      }
      const vx = R[0] * x + R[1] * y + R[2] * z;
      const vy = R[3] * x + R[4] * y + R[5] * z;
      B.VX[i] = vx;
      B.VY[i] = vy;
      B.VZ[i] = R[6] * x + R[7] * y + R[8] * z;
      ext = Math.max(ext, vx * vx + vy * vy);
      let s = 0;
      if (F && F.length >= fdim * n) {
        if (fdim === 1) s = F[i];
        else {
          for (let q = 0; q < fdim; q++) s += F[fdim * i + q] * F[fdim * i + q];
          s = Math.sqrt(s);
        }
      }
      B.S[i] = s;
      if (s < smin) smin = s;
      if (s > smax) smax = s;
    }
    const fit = Math.min(3 * geo.radius, Math.max(geo.radius, Math.sqrt(ext) || 0));
    const k = (cam.zoom * 0.42 * Math.min(W - legendW, H - 20)) / fit;
    for (let i = 0; i < n; i++) {
      B.SX[i] = cx + k * B.VX[i];
      B.SY[i] = cy - k * B.VY[i];
    }

    // the limits of the legend are evaluated by the values at all points (also inside)
    let lo = frame.limits ? frame.limits[0] : smin;
    let hi = frame.limits ? frame.limits[1] : smax;

    // a uniform field is drawn in one colour
    const flat = !(hi - lo > 1e-9 * Math.max(Math.abs(hi), Math.abs(lo), 1e-200));
    if (flat) hi = lo + 1;
    const dl = (hi - lo) / BANDS;
    const band = (s) => Math.min(BANDS - 1, Math.max(0, Math.floor((s - lo) / dl)));
    const styles = new Map();
    const style = (b, shade) => {
      const q = Math.round(shade * 32);
      const key = b * 64 + q;
      let st = styles.get(key);
      if (!st) {
        const c = BAND_COLORS[b];
        const f = q / 32;
        st = `rgb(${(c[0] * f) | 0},${(c[1] * f) | 0},${(c[2] * f) | 0})`;
        styles.set(key, st);
      }
      return st;
    };

    // depth sorting (painter's algorithm)
    for (let f = 0; f < geo.nf; f++) {
      const a = geo.fstart[f];
      const b = geo.fstart[f + 1];
      let z = 0;
      for (let q = a; q < b; q++) z += B.VZ[geo.fidx[q]];
      B.depth[f] = z / (b - a);
      B.order[f] = f;
    }
    const order = B.order.sort((p, q) => B.depth[p] - B.depth[q]);

    const xs = new Float64Array(4), ys = new Float64Array(4), ss = new Float64Array(4);
    const tx = new Float64Array(3), ty = new Float64Array(3), ts = new Float64Array(3);
    const edge = css("--fd-ink") || "#000";
    ctx.lineJoin = "round";
    for (let o = 0; o < geo.nf; o++) {
      const f = order[o];
      const a = geo.fstart[f];
      const nv = geo.fstart[f + 1] - a;
      for (let q = 0; q < nv; q++) {
        const i = geo.fidx[a + q];
        xs[q] = B.SX[i];
        ys[q] = B.SY[i];
        ss[q] = B.S[i];
      }
      // flat shading by the normal in view space
      const i0 = geo.fidx[a], i1 = geo.fidx[a + 1], i2 = geo.fidx[a + 2];
      const ax = B.VX[i1] - B.VX[i0], ay = B.VY[i1] - B.VY[i0], az = B.VZ[i1] - B.VZ[i0];
      const bx = B.VX[i2] - B.VX[i0], by = B.VY[i2] - B.VY[i0], bz = B.VZ[i2] - B.VZ[i0];
      const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const shade = 0.6 + (0.4 * Math.abs(nz)) / (Math.hypot(nx, ny, nz) || 1);

      let kmin = BANDS;
      let kmax = -1;
      for (let q = 0; q < nv; q++) {
        const b = band(ss[q]);
        kmin = Math.min(kmin, b);
        kmax = Math.max(kmax, b);
      }

      // base layer: the whole face in the colour of its lowest band
      ctx.beginPath();
      ctx.moveTo(xs[0], ys[0]);
      for (let q = 1; q < nv; q++) ctx.lineTo(xs[q], ys[q]);
      ctx.closePath();
      const base = style(kmin, shade);
      ctx.fillStyle = base;
      ctx.fill();
      if (!ui.edges) {
        ctx.strokeStyle = base;
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }

      // upper layers: the nested polygons s >= level of each band
      if (kmax > kmin) {
        for (let t = 0; t < nv - 2; t++) {
          tx[0] = xs[0]; ty[0] = ys[0]; ts[0] = ss[0];
          tx[1] = xs[t + 1]; ty[1] = ys[t + 1]; ts[1] = ss[t + 1];
          tx[2] = xs[t + 2]; ty[2] = ys[t + 2]; ts[2] = ss[t + 2];
          const tmax = band(Math.max(ts[0], ts[1], ts[2]));
          for (let b = kmin + 1; b <= tmax; b++) {
            const m = clipGE(tx, ty, ts, 3, lo + b * dl);
            if (m < 3) continue;
            ctx.beginPath();
            ctx.moveTo(CX[0], CY[0]);
            for (let q = 1; q < m; q++) ctx.lineTo(CX[q], CY[q]);
            ctx.closePath();
            const st = style(b, shade);
            ctx.fillStyle = st;
            ctx.fill();
            ctx.strokeStyle = st;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }
      if (ui.edges) {
        ctx.beginPath();
        ctx.moveTo(xs[0], ys[0]);
        for (let q = 1; q < nv; q++) ctx.lineTo(xs[q], ys[q]);
        ctx.closePath();
        ctx.globalAlpha = 0.4;
        ctx.strokeStyle = "#1b2631";
        ctx.lineWidth = 0.6;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // line cells (trusses) and points
    const L = geo.lines;
    ctx.lineCap = "round";
    for (let q = 0; q < L.length; q += 2) {
      const i = L[q];
      const j = L[q + 1];
      ctx.beginPath();
      ctx.moveTo(B.SX[i], B.SY[i]);
      ctx.lineTo(B.SX[j], B.SY[j]);
      ctx.strokeStyle = style(band(0.5 * (B.S[i] + B.S[j])), 1);
      ctx.lineWidth = 3;
      ctx.stroke();
    }
    if (info.surface === "points") {
      for (let i = 0; i < n; i++) {
        ctx.fillStyle = style(band(B.S[i]), 1);
        ctx.fillRect(B.SX[i] - 2, B.SY[i] - 2, 4, 4);
      }
    }

    drawLegend(ctx, W, H, legendW, lo, hi, fdim, flat, edge);
    drawTriad(ctx, H, dim, edge);
    drawCaption(ctx, info, edge);
  }

  function drawLegend(ctx, W, H, lw, lo, hi, fdim, flat, ink) {
    const x = W - lw + 4;
    const top = 26;
    const bw = 10;
    const h = Math.max(60, Math.min(H - top - 16, 170));
    const bh = h / BANDS;
    ctx.font = font(11);
    ctx.fillStyle = ink;
    ctx.textBaseline = "alphabetic";
    ctx.globalAlpha = 0.75;
    ctx.fillText(fdim > 1 ? "Field norm" : fdim ? "Field" : "", x, 16, lw - 8);
    ctx.globalAlpha = 1;
    if (flat) {
      const c = BAND_COLORS[0];
      ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
      ctx.fillRect(x, top, bw, 12);
      ctx.fillStyle = ink;
      ctx.textBaseline = "middle";
      ctx.fillText(fmt(lo, 3), x + bw + 4, top + 6, lw - bw - 8);
      return;
    }
    for (let b = 0; b < BANDS; b++) {
      const c = BAND_COLORS[b];
      ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
      ctx.fillRect(x, top + (BANDS - 1 - b) * bh, bw, bh + 0.5);
    }
    ctx.fillStyle = ink;
    ctx.textBaseline = "middle";
    const every = h / BANDS < 14 ? 2 : 1;
    for (let b = 0; b <= BANDS; b += every) {
      ctx.fillText(fmt(lo + ((hi - lo) * b) / BANDS, 3), x + bw + 4, top + (BANDS - b) * bh, lw - bw - 8);
    }
    ctx.textBaseline = "alphabetic";
  }

  function drawTriad(ctx, H, dim, ink) {
    const R = cam.R;
    const ox = 22;
    const oy = H - 22;
    const L = 14;
    ctx.font = font(11);
    ctx.strokeStyle = ctx.fillStyle = ink;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1;
    (dim === 3 ? ["x", "y", "z"] : ["x", "y"]).forEach((name, j) => {
      const dx = R[j];
      const dy = R[3 + j];
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + L * dx, oy - L * dy);
      ctx.stroke();
      ctx.fillText(name, ox + (L + 6) * dx - 3, oy - (L + 6) * dy + 4);
    });
    ctx.globalAlpha = 1;
  }

  function drawCaption(ctx, info, ink) {
    let text = "Initial state";
    if (frame.kind === "substep") {
      const nsub = (info.nsubsteps || [])[frame.step];
      text = `Step ${frame.step + 1}, substep ${frame.substep + 1} of ${nsub}`;
    }
    if (info.deform && ui.scale !== 1) text += `, deformation scaled by ${ui.scale}`;
    ctx.font = font(12);
    ctx.fillStyle = ink;
    ctx.textBaseline = "alphabetic";
    ctx.fillText(text, 10, 18);
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
  root.querySelectorAll("[data-view]").forEach((button) =>
    button.addEventListener("click", () => {
      if (button.dataset.view !== "fit") cam.R = VIEWS[button.dataset.view].slice();
      cam.zoom = 1;
      cam.panx = cam.pany = 0;
      schedule("view");
    }),
  );

  const view = $(".fd-canvas");
  let drag = null;
  view.addEventListener("contextmenu", (e) => e.preventDefault());
  view.addEventListener("pointerdown", (e) => {
    view.focus({ preventScroll: true });
    drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button !== 0 };
    view.setPointerCapture(e.pointerId);
  });
  view.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.pan) {
      cam.panx += dx;
      cam.pany += dy;
    } else cam.R = mul(mul(rotY(dx * 0.008), rotX(dy * 0.008)), cam.R);
    schedule("view");
  });
  const end = () => (drag = null);
  view.addEventListener("pointerup", end);
  view.addEventListener("pointercancel", end);
  // the wheel zooms only after a click into the viewport, otherwise it scrolls
  view.addEventListener(
    "wheel",
    (e) => {
      if (document.activeElement !== view) return;
      e.preventDefault();
      cam.zoom *= Math.exp(-e.deltaY * 0.0015);
      schedule("view");
    },
    { passive: false },
  );
  view.addEventListener("dblclick", () => {
    cam.zoom = 1;
    cam.panx = cam.pany = 0;
    schedule("view");
  });

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

  const resize = new ResizeObserver(() => {
    schedule("view");
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
  };
}

export default { render };
