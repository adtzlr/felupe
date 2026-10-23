// SPDX-License-Identifier: GPL-3.0-or-later

// The WebGL2 viewer of the plotting backend "webgl" and of the dashboard of FElupe. A
// dependency-free ES module, which draws a scene of surfaces (with scalars, edges and
// an undeformed ghost), points and lines with a colormap (continuous or in bands), a
// scalar bar, a legend and the axes. Drag to rotate, shift-drag (or right-drag) to pan,
// scroll to zoom, double-click to reset the view.

// the colors of the overlays follow the theme of VS Code (--vscode-*) or of JupyterLab
// (--jp-*), otherwise the light or dark color scheme of the page
const STYLES = `
.fv { position: relative; width: 100%; height: 100%; overflow: hidden;
  font-size: 12px; line-height: 1.35;
  font-family: var(--vscode-font-family, var(--jp-ui-font-family, system-ui, sans-serif));
  color: var(--vscode-editor-foreground, var(--jp-ui-font-color1, #1b2631));
  --fv-panel: var(--vscode-editorWidget-background, var(--jp-layout-color1, rgba(255, 255, 255, 0.82)));
  --fv-rule: var(--vscode-editorWidget-border, var(--jp-border-color2, #d3d9de));
  user-select: none; -webkit-user-select: none; touch-action: none; }
@media (prefers-color-scheme: dark) {
  .fv { color: var(--vscode-editor-foreground, var(--jp-ui-font-color1, #e6e8ea));
    --fv-panel: var(--vscode-editorWidget-background, var(--jp-layout-color1, rgba(30, 31, 34, 0.82)));
    --fv-rule: var(--vscode-editorWidget-border, var(--jp-border-color2, #3a3d42)); } }
.fv canvas.fv-gl { position: absolute; inset: 0; width: 100%; height: 100%;
  display: block; cursor: grab; outline: none; }
.fv canvas.fv-gl:active { cursor: grabbing; }
.fv-bars { position: absolute; left: 50%; bottom: 10px; transform: translateX(-50%);
  display: flex; flex-direction: column; gap: 6px; width: min(60%, 420px);
  pointer-events: none; }
.fv-bars.fv-vertical { left: auto; right: 12px; top: 50%; bottom: auto;
  transform: translateY(-50%); width: auto; flex-direction: row; gap: 14px; }
.fv-bar { display: grid; gap: 2px; }
.fv-bar-title { text-align: center; font-weight: 600; }
.fv-bar canvas { width: 100%; height: 12px; border: 1px solid var(--fv-rule);
  border-radius: 2px; }
.fv-ticks { display: flex; justify-content: space-between;
  font-variant-numeric: tabular-nums; }
.fv-vertical .fv-bar { grid-template-columns: auto auto; align-items: stretch;
  grid-template-rows: auto 1fr; }
.fv-vertical .fv-bar-title { grid-column: 1 / 3; }
.fv-vertical .fv-bar canvas { width: 14px; height: min(40vh, 220px); }
.fv-vertical .fv-ticks { flex-direction: column-reverse; }
.fv-legend { position: absolute; top: 10px; right: 10px; padding: 6px 9px;
  background: var(--fv-panel); border: 1px solid var(--fv-rule); border-radius: 4px;
  display: grid; gap: 3px; pointer-events: none; }
.fv-legend div { display: flex; align-items: center; gap: 7px; }
.fv-legend span { width: 11px; height: 11px; border-radius: 2px; flex: none; }
.fv-axes { position: absolute; left: 6px; bottom: 6px; width: 72px; height: 72px;
  pointer-events: none; }
.fv-message { position: absolute; inset: 0; display: grid; place-items: center;
  padding: 16px; text-align: center; }
`;

// -------------------------------------------------------------------- utilities

const TYPES = { f4: Float32Array, u4: Uint32Array, u1: Uint8Array };

function decode(item) {
  "Decode a flat array of the scene (little-endian), by base64 or a binary buffer.";
  if (!item) return null;
  const Type = TYPES[item.dtype];
  const data = item.data;

  if (typeof data === "string") {
    const binary = atob(data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Type(bytes.buffer, 0, item.size);
  }

  // a DataView (binary buffer of a widget), an ArrayBuffer or a typed array
  const buffer = data.buffer || data;
  const offset = data.byteOffset || 0;
  if (offset % Type.BYTES_PER_ELEMENT === 0) return new Type(buffer, offset, item.size);
  return new Type(buffer.slice(offset, offset + item.size * Type.BYTES_PER_ELEMENT));
}

let colorContext = null;

function parseColor(css, fallback = [0.5, 0.5, 0.5]) {
  "Return the RGB components (0..1) of a CSS color.";
  if (css == null) return fallback;
  colorContext = colorContext || document.createElement("canvas").getContext("2d");
  colorContext.fillStyle = "#010203";
  colorContext.fillStyle = css;
  const value = colorContext.fillStyle;
  if (value[0] === "#") {
    const n = parseInt(value.slice(1), 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }
  const parts = value.match(/[\d.]+/g) || [];
  return parts.length >= 3 ? parts.slice(0, 3).map((p) => p / 255) : fallback;
}

function turbo(t) {
  "The colormap turbo by a polynomial approximation (if no lookup table is given).";
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [r, g, b].map((c) => Math.round(255 * Math.min(Math.max(c, 0), 1)));
}

function lookupTable(item) {
  "Return the lookup table of 256 colors (RGB, uint8) of the colormap.";
  const lut = decode(item);
  if (lut && lut.length === 768) return lut;
  const table = new Uint8Array(768);
  for (let i = 0; i < 256; i++) table.set(turbo(i / 255), 3 * i);
  return table;
}

function colorAt(table, t, bands = 0) {
  "Return the color (RGB, 0..255) of the lookup table at a parameter (in bands).";
  if (bands > 0) t = (Math.min(Math.floor(t * bands), bands - 1) + 0.5) / bands;
  const x = Math.min(Math.max(t, 0), 1) * 255;
  const i = Math.min(Math.floor(x), 254);
  const f = x - i;
  return [0, 1, 2].map((k) => Math.round((1 - f) * table[3 * i + k] + f * table[3 * i + 3 + k]));
}

function ticks(bands, vertical) {
  "Return the parameters of the ticks of a scalar bar (at the limits of the bands).";
  if (!bands) return [0, 0.25, 0.5, 0.75, 1];
  const most = vertical ? 10 : 5;
  let step = 1;
  while (bands / step > most || bands % step) step++;
  const result = [];
  for (let k = 0; k <= bands; k += step) result.push(k / bands);
  return result;
}

function formatNumber(x) {
  if (x === 0 || !Number.isFinite(x)) return String(x);
  const a = Math.abs(x);
  return a >= 1e4 || a < 1e-3 ? x.toExponential(2) : Number(x.toPrecision(3)).toString();
}

// ------------------------------------------------------------------ linear algebra

const vec = {
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ],
  norm: (a) => Math.hypot(a[0], a[1], a[2]),
  unit: (a) => {
    const n = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / n, a[1] / n, a[2] / n];
  },
};

function rotate(v, axis, angle) {
  "Rotate a vector about a unit axis (Rodrigues' formula).";
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const k = vec.cross(axis, v);
  const d = vec.dot(axis, v) * (1 - c);
  return [
    v[0] * c + k[0] * s + axis[0] * d,
    v[1] * c + k[1] * s + axis[1] * d,
    v[2] * c + k[2] * s + axis[2] * d,
  ];
}

// the camera basis (right, up, back) of the views, "default" is the default view of
// the plots of FElupe by PyVista (the isometric view, rotated by an azimuth of -100
// and an elevation of -15 degrees)
const VIEWS = {
  xy: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  yx: [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
  xz: [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
  zx: [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
  yz: [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
  zy: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
};

function obliqueView(azimuth, elevation) {
  const az = (azimuth * Math.PI) / 180;
  const el = (elevation * Math.PI) / 180;
  const back = [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
  const right = vec.unit(vec.cross([0, 0, 1], back));
  return [right, vec.cross(back, right), back];
}

VIEWS.iso = obliqueView(45, 35.264);
VIEWS.default3d = obliqueView(-55, 20.264);

// --------------------------------------------------------------------- shaders

// the color of a value by the lookup table, optionally in bands of uniform colors
const COLORMAP = `
uniform bool useValues; uniform vec2 range; uniform sampler2D lut; uniform float bands;
vec3 colormap(vec3 base, float value) {
  if (!useValues) return base;
  if (value != value) return vec3(0.6);
  float t = range.y > range.x ? clamp((value - range.x) / (range.y - range.x), 0.0, 1.0) : 0.5;
  if (bands > 0.5) t = (min(floor(t * bands), bands - 1.0) + 0.5) / bands;
  return texture(lut, vec2(t * 255.0 / 256.0 + 0.5 / 256.0, 0.5)).rgb;
}`;

const SURFACE_VS = `#version 300 es
in vec3 position; in vec3 normal; in float value;
uniform mat4 viewProjection; uniform mat3 viewRotation;
out vec3 vNormal; out float vValue;
void main() {
  vNormal = viewRotation * normal; vValue = value;
  gl_Position = viewProjection * vec4(position, 1.0);
}`;

const SURFACE_FS = `#version 300 es
precision highp float;
in vec3 vNormal; in float vValue;
uniform vec4 color; uniform bool lighting;
${COLORMAP}
out vec4 fragColor;
void main() {
  vec3 base = colormap(color.rgb, vValue);
  // a key light (from the upper front) and a headlight, lit on both sides
  vec3 n = normalize(vNormal);
  float key = abs(dot(n, normalize(vec3(0.2, 0.7, 0.7))));
  float head = abs(n.z);
  float specular = pow(head, 40.0);
  float light = 0.25 + 0.55 * key + 0.25 * head;
  vec3 shaded = lighting ? base * light + 0.06 * specular : base;
  fragColor = vec4(shaded * color.a, color.a);
}`;

const LINE_VS = `#version 300 es
in vec2 corner; in vec3 start; in vec3 end; in float startValue; in float endValue;
uniform mat4 viewProjection; uniform vec2 viewport; uniform float width;
out float vValue;
void main() {
  vValue = corner.x < 0.5 ? startValue : endValue;
  vec4 a = viewProjection * vec4(start, 1.0);
  vec4 b = viewProjection * vec4(end, 1.0);
  vec2 sa = a.xy / a.w * viewport; vec2 sb = b.xy / b.w * viewport;
  vec2 dir = sb - sa;
  dir = length(dir) > 1e-9 ? normalize(dir) : vec2(1.0, 0.0);
  vec2 offset = vec2(-dir.y, dir.x) * width * corner.y / viewport;
  vec4 p = corner.x < 0.5 ? a : b;
  gl_Position = vec4(p.xy + offset * p.w, p.z, p.w);
}`;

const POINT_VS = `#version 300 es
in vec3 position; in float value;
uniform mat4 viewProjection; uniform float size;
out float vValue;
void main() {
  vValue = value;
  gl_Position = viewProjection * vec4(position, 1.0);
  gl_PointSize = size;
}`;

const FLAT_FS = `#version 300 es
precision highp float;
in float vValue;
uniform vec4 color; uniform bool roundPoints;
${COLORMAP}
out vec4 fragColor;
void main() {
  if (roundPoints && length(gl_PointCoord - 0.5) > 0.5) discard;
  fragColor = vec4(colormap(color.rgb, vValue) * color.a, color.a);
}`;

function compile(gl, vertex, fragment) {
  const program = gl.createProgram();
  for (const [type, source] of [
    [gl.VERTEX_SHADER, vertex],
    [gl.FRAGMENT_SHADER, fragment],
  ]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  const uniforms = {};
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const name = gl.getActiveUniform(program, i).name;
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  const attributes = {};
  const nattributes = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES);
  for (let i = 0; i < nattributes; i++) {
    const name = gl.getActiveAttrib(program, i).name;
    attributes[name] = gl.getAttribLocation(program, name);
  }
  return { program, uniforms, attributes };
}

// -------------------------------------------------------------------- geometry

function deformed(obj) {
  "Return the (deformed) positions of the points of an object.";
  const positions = decode(obj.positions);
  const displacement = decode(obj.displacement);
  if (!displacement) return positions;
  const factor = obj.factor ?? 1;
  const result = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i++) {
    result[i] = positions[i] + factor * displacement[i];
  }
  return result;
}

function surfaceGeometry(obj, positions) {
  "Return the vertices (per corner of each triangle) of a surface with normals and values.";
  const triangles = decode(obj.triangles);
  const values = decode(obj.values);
  const faceIds = obj.smooth ? null : decode(obj.faceIds);
  const ntriangles = triangles.length / 3;
  const npoints = positions.length / 3;

  // the (area-weighted) normals of the triangles, of the faces and of the points
  const normals = new Float32Array(3 * ntriangles);
  for (let t = 0; t < ntriangles; t++) {
    const a = 3 * triangles[3 * t];
    const b = 3 * triangles[3 * t + 1];
    const c = 3 * triangles[3 * t + 2];
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a];
    const vy = positions[c + 1] - positions[a + 1];
    const vz = positions[c + 2] - positions[a + 2];
    normals[3 * t] = uy * vz - uz * vy;
    normals[3 * t + 1] = uz * vx - ux * vz;
    normals[3 * t + 2] = ux * vy - uy * vx;
  }

  let sums = null;
  if (faceIds) {
    let nfaces = 0;
    for (let t = 0; t < ntriangles; t++) nfaces = Math.max(nfaces, faceIds[t] + 1);
    sums = new Float32Array(3 * nfaces);
    for (let t = 0; t < ntriangles; t++) {
      for (let k = 0; k < 3; k++) sums[3 * faceIds[t] + k] += normals[3 * t + k];
    }
  }

  let pointNormals = null;
  if (obj.smooth) {
    pointNormals = new Float32Array(3 * npoints);
    for (let t = 0; t < ntriangles; t++) {
      for (let j = 0; j < 3; j++) {
        const p = 3 * triangles[3 * t + j];
        for (let k = 0; k < 3; k++) pointNormals[p + k] += normals[3 * t + k];
      }
    }
    for (let p = 0; p < npoints; p++) {
      const n = Math.hypot(pointNormals[3 * p], pointNormals[3 * p + 1], pointNormals[3 * p + 2]) || 1;
      for (let k = 0; k < 3; k++) pointNormals[3 * p + k] /= n;
    }
  }

  const nvertices = 3 * ntriangles;
  const vertexPositions = new Float32Array(3 * nvertices);
  const vertexNormals = new Float32Array(3 * nvertices);
  const vertexValues = new Float32Array(nvertices);
  const sharp = Math.cos((30 * Math.PI) / 180);
  const cell = obj.valuesAt === "cell";

  for (let t = 0; t < ntriangles; t++) {
    // the normal of the triangle (or of its face)
    const source = sums || normals;
    const f = sums ? 3 * faceIds[t] : 3 * t;
    const length = Math.hypot(source[f], source[f + 1], source[f + 2]) || 1;
    const fx = source[f] / length;
    const fy = source[f + 1] / length;
    const fz = source[f + 2] / length;

    for (let j = 0; j < 3; j++) {
      const p = triangles[3 * t + j];
      const v = 3 * (3 * t + j);
      vertexPositions[v] = positions[3 * p];
      vertexPositions[v + 1] = positions[3 * p + 1];
      vertexPositions[v + 2] = positions[3 * p + 2];
      let nx = fx;
      let ny = fy;
      let nz = fz;
      if (pointNormals) {
        const sx = pointNormals[3 * p];
        const sy = pointNormals[3 * p + 1];
        const sz = pointNormals[3 * p + 2];
        if (!obj.sharp || sx * fx + sy * fy + sz * fz > sharp) {
          nx = sx;
          ny = sy;
          nz = sz;
        }
      }
      vertexNormals[v] = nx;
      vertexNormals[v + 1] = ny;
      vertexNormals[v + 2] = nz;
      if (values) vertexValues[3 * t + j] = cell ? values[t] : values[p];
    }
  }

  return { vertexPositions, vertexNormals, vertexValues, count: nvertices };
}

function segments(obj, positions) {
  "Return the start and end points of the line segments of an object.";
  if (obj.type === "lines") return positions;
  const edges = decode(obj.edges);
  const result = new Float32Array(3 * edges.length);
  for (let i = 0; i < edges.length; i++) {
    result.set(positions.subarray(3 * edges[i], 3 * edges[i] + 3), 3 * i);
  }
  return result;
}

// ---------------------------------------------------------------------- viewer

export function createViewer(element, scene, options = {}) {
  "Create a viewer of a scene in an element and return an object to update it.";

  // with wheelOnFocus, the wheel zooms only after a click into the view (otherwise it
  // scrolls the page, e.g. a notebook)
  const { wheelOnFocus = false } = options;

  if (!document.getElementById("felupe-viewer-styles")) {
    const style = document.createElement("style");
    style.id = "felupe-viewer-styles";
    style.textContent = STYLES;
    document.head.appendChild(style);
  }

  const root = document.createElement("div");
  root.className = "fv";
  element.appendChild(root);

  const canvas = document.createElement("canvas");
  canvas.className = "fv-gl";
  canvas.tabIndex = 0;
  canvas.setAttribute("aria-label", "3D view of the model. Drag to rotate, shift-drag to pan, scroll to zoom, double-click to reset.");
  root.appendChild(canvas);

  const gl = canvas.getContext("webgl2", { antialias: true, alpha: true, premultipliedAlpha: true });
  if (!gl) {
    const message = document.createElement("div");
    message.className = "fv-message";
    message.textContent = "The 3D view requires WebGL2, which is not available.";
    root.appendChild(message);
    return { root, update() {}, dispose() { root.remove(); } };
  }

  const programs = {
    surface: compile(gl, SURFACE_VS, SURFACE_FS),
    line: compile(gl, LINE_VS, FLAT_FS),
    point: compile(gl, POINT_VS, FLAT_FS),
  };

  // a quad of two triangles for the line segments, instanced per segment
  const cornerBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([0, -1, 1, -1, 1, 1, 0, -1, 1, 1, 0, 1]),
    gl.STATIC_DRAW,
  );

  let drawables = [];
  let bounds = null;
  let center = null;
  const camera = { basis: VIEWS.default3d, target: [0, 0, 0], radius: 1, zoom: 1, parallel: false };
  const overlays = { bars: null, legend: null, axes: null };
  let pending = false;

  function buffer(data) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return b;
  }

  function attribute(program, name, b, size, divisor = 0) {
    const location = program.attributes[name];
    if (location === undefined || location < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(location, divisor);
  }

  function instanced(program, b, pairs, size, stride) {
    "Bind the attributes of the start and the end of the instanced line segments.";
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    for (const [name, offset] of pairs) {
      const location = program.attributes[name];
      if (location === undefined || location < 0) continue;
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(location, 1);
    }
  }

  function colormap(obj, values) {
    "The colormap of the values of an object (if there are values).";
    return {
      useValues: Boolean(values), range: obj.range || [0, 1], bands: obj.bands || 0,
      texture: values ? lutTexture(lookupTable(obj.lut)) : null,
    };
  }

  function lineDrawable(starts, color, width, opacity, values = null, map = colormap({}, null)) {
    "A drawable of line segments by instanced quads (optionally with values).";
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const b = buffer(starts);
    attribute(programs.line, "corner", cornerBuffer, 2);
    instanced(programs.line, b, [["start", 0], ["end", 12]], 3, 24);
    const buffers = [b];
    if (values) {
      const v = buffer(values);
      instanced(programs.line, v, [["startValue", 0], ["endValue", 4]], 1, 8);
      buffers.push(v);
    }
    gl.bindVertexArray(null);
    return {
      kind: "line", vao, count: starts.length / 6, color: parseColor(color, [0, 0, 0]),
      width, opacity, buffers, ...map,
    };
  }

  function lutTexture(table) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, table);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  function extend(positions) {
    for (let i = 0; i < positions.length; i += 3) {
      const p = [positions[i], positions[i + 1], positions[i + 2]];
      if (!p.every(Number.isFinite)) continue;
      if (!bounds) bounds = [p.slice(), p.slice()];
      for (let k = 0; k < 3; k++) {
        bounds[0][k] = Math.min(bounds[0][k], p[k]);
        bounds[1][k] = Math.max(bounds[1][k], p[k]);
      }
    }
  }

  function build() {
    for (const d of drawables) {
      gl.deleteVertexArray(d.vao);
      for (const b of d.buffers) gl.deleteBuffer(b);
      if (d.texture) gl.deleteTexture(d.texture);
    }
    drawables = [];
    bounds = null;

    for (const obj of scene.objects || []) {
      const positions = deformed(obj);
      extend(positions);

      if (obj.type === "surface") {
        const opacity = obj.opacity ?? 1;
        if (obj.faces !== false && obj.triangles && obj.triangles.size > 0) {
          const g = surfaceGeometry(obj, positions);
          const vao = gl.createVertexArray();
          gl.bindVertexArray(vao);
          const buffers = [buffer(g.vertexPositions), buffer(g.vertexNormals), buffer(g.vertexValues)];
          attribute(programs.surface, "position", buffers[0], 3);
          attribute(programs.surface, "normal", buffers[1], 3);
          attribute(programs.surface, "value", buffers[2], 1);
          gl.bindVertexArray(null);
          drawables.push({
            kind: "surface", vao, count: g.count, buffers, opacity,
            color: parseColor(obj.color, [0.68, 0.85, 0.9]),
            ...colormap(obj, obj.values),
          });
        }
        if (obj.edges && obj.edges.size > 0) {
          const width = obj.lineWidth ?? 1;
          drawables.push(lineDrawable(segments(obj, positions), obj.edgeColor, width, opacity));
        }
      } else if (obj.type === "lines" && positions.length > 0) {
        // the values at the start and the end points of the line segments
        const values = decode(obj.values);
        drawables.push(
          lineDrawable(positions, obj.color, obj.width ?? 1, obj.opacity ?? 1, values, colormap(obj, values)),
        );
      } else if (obj.type === "points" && positions.length > 0) {
        const values = decode(obj.values);
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const buffers = [buffer(positions)];
        attribute(programs.point, "position", buffers[0], 3);
        if (values) {
          buffers.push(buffer(values));
          attribute(programs.point, "value", buffers[1], 1);
        }
        gl.bindVertexArray(null);
        drawables.push({
          kind: "point", vao, count: positions.length / 3, buffers,
          color: parseColor(obj.color), size: obj.size ?? 5, opacity: obj.opacity ?? 1,
          ...colormap(obj, values),
        });
      }
    }
    if (!bounds) bounds = [[-1, -1, -1], [1, 1, 1]];
  }

  // ------------------------------------------------------------------ camera

  function reset(view = scene.view) {
    "Reset the camera to a view of the scene (the default view of the scene by default).";
    const planar = bounds[1][2] - bounds[0][2] <= 1e-9 * Math.max(1e-300, ...bounds[1].map((b, k) => b - bounds[0][k]));
    if (view === "default" || view == null || !(view in VIEWS)) view = planar ? "xy" : "default3d";
    camera.basis = VIEWS[view].map((v) => v.slice());
    camera.parallel = planar;
    center = vec.scale(vec.add(bounds[0], bounds[1]), 0.5);
    camera.target = center.slice();
    camera.radius = Math.max(0.5 * vec.norm(vec.sub(bounds[1], bounds[0])), 1e-12);
    camera.zoom = 1;
  }

  function matrices(width, height) {
    "Return the view-projection matrix (column-major) and the rotation of the view.";
    const [right, up, back] = camera.basis;
    const fov = (30 * Math.PI) / 180;
    const aspect = width / Math.max(height, 1);
    const fit = camera.radius / camera.zoom;
    const distance = camera.radius / Math.sin(fov / 2);
    const eye = vec.add(camera.target, vec.scale(back, distance));

    // world to view coordinates
    const t = [-vec.dot(right, eye), -vec.dot(up, eye), -vec.dot(back, eye)];
    const view = [
      right[0], up[0], back[0], 0,
      right[1], up[1], back[1], 0,
      right[2], up[2], back[2], 0,
      t[0], t[1], t[2], 1,
    ];

    const near = Math.max(distance - 3 * camera.radius, distance * 1e-3);
    const far = distance + 3 * camera.radius;
    let projection;
    if (camera.parallel) {
      const h = 1.1 * fit;
      const w = h * aspect;
      projection = [
        1 / w, 0, 0, 0,
        0, 1 / h, 0, 0,
        0, 0, -2 / (far - near), 0,
        0, 0, -(far + near) / (far - near), 1,
      ];
    } else {
      // the zoom narrows the field of view (the clipping planes are unchanged)
      const f = 1 / Math.tan(fov / 2) / 1.1 * camera.zoom;
      projection = [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) / (near - far), -1,
        0, 0, (2 * far * near) / (near - far), 0,
      ];
    }

    const m = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += projection[k * 4 + r] * view[c * 4 + k];
        m[c * 4 + r] = s;
      }
    }
    const rotation = new Float32Array([
      right[0], up[0], back[0],
      right[1], up[1], back[1],
      right[2], up[2], back[2],
    ]);
    return { viewProjection: m, rotation };
  }

  // ------------------------------------------------------------------- render

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  function setColormap(p, d) {
    "Set the uniforms of the colormap of a drawable.";
    gl.uniform1i(p.uniforms.useValues, d.useValues ? 1 : 0);
    gl.uniform2f(p.uniforms.range, d.range[0], d.range[1]);
    gl.uniform1f(p.uniforms.bands, d.bands);
    if (d.texture) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, d.texture);
      gl.uniform1i(p.uniforms.lut, 0);
    }
  }

  function draw() {
    pending = false;
    resize();
    const ratio = window.devicePixelRatio || 1;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    const { viewProjection, rotation } = matrices(canvas.width, canvas.height);

    // opaque objects first, then transparent objects without writing the depth
    const passes = [drawables.filter((d) => d.opacity >= 1), drawables.filter((d) => d.opacity < 1)];
    passes.forEach((list, pass) => {
      gl.depthMask(pass === 0);
      for (const d of list) {
        if (d.kind === "surface") {
          const p = programs.surface;
          gl.useProgram(p.program);
          gl.uniformMatrix4fv(p.uniforms.viewProjection, false, viewProjection);
          gl.uniformMatrix3fv(p.uniforms.viewRotation, false, rotation);
          gl.uniform4f(p.uniforms.color, ...d.color, d.opacity);
          gl.uniform1i(p.uniforms.lighting, 1);
          setColormap(p, d);
          // opaque faces are pushed back (behind their edges and behind coincident
          // transparent faces, e.g. of the undeformed model at a fixed boundary)
          gl.enable(gl.POLYGON_OFFSET_FILL);
          gl.polygonOffset(pass === 0 ? 1 : 0, pass === 0 ? 1 : 0);
          gl.bindVertexArray(d.vao);
          gl.drawArrays(gl.TRIANGLES, 0, d.count);
          gl.disable(gl.POLYGON_OFFSET_FILL);
        } else if (d.kind === "line") {
          const p = programs.line;
          gl.useProgram(p.program);
          gl.uniformMatrix4fv(p.uniforms.viewProjection, false, viewProjection);
          gl.uniform2f(p.uniforms.viewport, canvas.width, canvas.height);
          gl.uniform1f(p.uniforms.width, Math.max(1, d.width * ratio));
          gl.uniform4f(p.uniforms.color, ...d.color, d.opacity);
          gl.uniform1i(p.uniforms.roundPoints, 0);
          setColormap(p, d);
          gl.bindVertexArray(d.vao);
          gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, d.count);
        } else if (d.kind === "point") {
          const p = programs.point;
          gl.useProgram(p.program);
          gl.uniformMatrix4fv(p.uniforms.viewProjection, false, viewProjection);
          gl.uniform1f(p.uniforms.size, Math.max(1, d.size * ratio));
          gl.uniform4f(p.uniforms.color, ...d.color, d.opacity);
          gl.uniform1i(p.uniforms.roundPoints, 1);
          setColormap(p, d);
          gl.bindVertexArray(d.vao);
          gl.drawArrays(gl.POINTS, 0, d.count);
        }
      }
    });
    gl.bindVertexArray(null);
    gl.depthMask(true);
    drawAxes();
  }

  function redraw() {
    if (!pending) {
      pending = true;
      requestAnimationFrame(draw);
    }
  }

  // ----------------------------------------------------------------- overlays

  function buildOverlays() {
    for (const key of ["bars", "legend", "axes"]) {
      if (overlays[key]) overlays[key].remove();
      overlays[key] = null;
    }

    // one scalar bar per label of the scalars
    const bars = [];
    const labels = new Set();
    for (const obj of scene.objects || []) {
      if (obj.values && obj.scalarBar !== false && obj.range && !labels.has(obj.label)) {
        labels.add(obj.label);
        bars.push(obj);
      }
    }
    if (bars.length) {
      const container = document.createElement("div");
      container.className = "fv-bars" + (bars[0].verticalBar ? " fv-vertical" : "");
      for (const obj of bars) {
        const bar = document.createElement("div");
        bar.className = "fv-bar";
        const title = document.createElement("div");
        title.className = "fv-bar-title";
        title.textContent = obj.label;
        const gradient = document.createElement("canvas");
        const vertical = Boolean(obj.verticalBar);
        gradient.width = vertical ? 1 : 256;
        gradient.height = vertical ? 256 : 1;
        const context = gradient.getContext("2d");
        const image = context.createImageData(gradient.width, gradient.height);
        const table = lookupTable(obj.lut);
        const [a, b] = obj.range;

        // a uniform range is drawn in the color of its value (like the surface)
        const uniform = !(b > a);
        for (let i = 0; i < 256; i++) {
          const j = vertical ? 255 - i : i;
          const color = colorAt(table, uniform ? 0.5 : (i + 0.5) / 256, obj.bands || 0);
          image.data.set([...color, 255], 4 * j);
        }
        context.putImageData(image, 0, 0);
        const labels = document.createElement("div");
        labels.className = "fv-ticks";
        for (const s of uniform ? [0] : ticks(obj.bands || 0, vertical)) {
          const tick = document.createElement("span");
          tick.textContent = formatNumber(a + s * (b - a));
          labels.appendChild(tick);
        }
        bar.append(title, gradient, labels);
        container.appendChild(bar);
      }
      root.appendChild(container);
      overlays.bars = container;
    }

    if (scene.legend && scene.legend.length) {
      const legend = document.createElement("div");
      legend.className = "fv-legend";
      for (const entry of scene.legend) {
        const row = document.createElement("div");
        const swatch = document.createElement("span");
        swatch.style.background = entry.color;
        const text = document.createElement("div");
        text.textContent = entry.label;
        row.append(swatch, text);
        legend.appendChild(row);
      }
      root.appendChild(legend);
      overlays.legend = legend;
    }

    if (scene.axes) {
      const axes = document.createElement("canvas");
      axes.className = "fv-axes";
      root.appendChild(axes);
      overlays.axes = axes;
    }
  }

  function drawAxes() {
    const axes = overlays.axes;
    if (!axes) return;
    const ratio = window.devicePixelRatio || 1;
    const size = Math.round(72 * ratio);
    if (axes.width !== size) axes.width = axes.height = size;
    const context = axes.getContext("2d");
    context.clearRect(0, 0, size, size);
    const [right, up, back] = camera.basis;
    const items = [
      ["x", [1, 0, 0], "#d62728"],
      ["y", [0, 1, 0], "#2ca02c"],
      ["z", [0, 0, 1], "#1f77b4"],
    ].map(([name, axis, color]) => ({
      name, color, x: vec.dot(axis, right), y: vec.dot(axis, up), z: vec.dot(axis, back),
    }));
    items.sort((a, b) => a.z - b.z);
    const c = size / 2;
    const length = 0.32 * size;
    context.lineWidth = 2 * ratio;
    context.font = `${11 * ratio}px system-ui, sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    for (const item of items) {
      context.strokeStyle = context.fillStyle = item.color;
      // an axis along the view direction is labelled at the lower left of the origin
      if (Math.hypot(item.x, item.y) < 0.05) {
        context.fillText(item.name, c - 0.25 * length, c + 0.25 * length);
        continue;
      }
      context.beginPath();
      context.moveTo(c, c);
      context.lineTo(c + length * item.x, c - length * item.y);
      context.stroke();
      context.fillText(item.name, c + 1.3 * length * item.x, c - 1.3 * length * item.y);
    }
  }

  // ------------------------------------------------------------- interaction

  const pointers = new Map();
  let pinch = null;

  function rotateBy(dx, dy) {
    const angle = 0.008;
    let [right, up, back] = camera.basis;
    // rotate the view about the screen axes (the scene follows the pointer)
    const axisUp = up;
    right = rotate(right, axisUp, -dx * angle);
    back = rotate(back, axisUp, -dx * angle);
    up = rotate(up, right, -dy * angle);
    back = rotate(back, right, -dy * angle);
    // orthonormalize
    back = vec.unit(back);
    right = vec.unit(vec.cross(up, back));
    up = vec.cross(back, right);
    camera.basis = [right, up, back];
  }

  function panBy(dx, dy) {
    const [right, up] = camera.basis;
    const scale = (2.2 * camera.radius) / camera.zoom / Math.max(canvas.clientHeight, 1);
    camera.target = vec.add(
      camera.target,
      vec.add(vec.scale(right, -dx * scale), vec.scale(up, dy * scale)),
    );
  }

  canvas.addEventListener("pointerdown", (event) => {
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY, button: event.button, shift: event.shiftKey });
  });

  canvas.addEventListener("pointermove", (event) => {
    const previous = pointers.get(event.pointerId);
    if (!previous) return;
    const dx = event.clientX - previous.x;
    const dy = event.clientY - previous.y;
    previous.x = event.clientX;
    previous.y = event.clientY;

    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch) camera.zoom *= distance / pinch;
      pinch = distance;
      panBy(dx / 2, dy / 2);
    } else if (previous.button === 2 || previous.button === 1 || event.shiftKey) {
      panBy(dx, dy);
    } else {
      rotateBy(dx, dy);
    }
    redraw();
  });

  const release = (event) => {
    pointers.delete(event.pointerId);
    pinch = null;
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  canvas.addEventListener(
    "wheel",
    (event) => {
      if (wheelOnFocus && document.activeElement !== canvas) return;
      event.preventDefault();
      camera.zoom *= Math.exp(-event.deltaY * 0.0015);
      camera.zoom = Math.min(Math.max(camera.zoom, 0.05), 200);
      redraw();
    },
    { passive: false },
  );

  canvas.addEventListener("dblclick", () => {
    reset();
    redraw();
  });

  const observer = new ResizeObserver(() => redraw());
  observer.observe(canvas);

  function update(newScene, { keepView = false } = {}) {
    "Show a new scene. The view is reset, or kept (the rotation, zoom and pan).";
    const kept = keepView && center
      ? { basis: camera.basis, zoom: camera.zoom, pan: vec.sub(camera.target, center) }
      : null;
    scene = newScene;
    build();
    buildOverlays();
    reset();

    // the kept view follows the center and the size of the new scene
    if (kept) {
      camera.basis = kept.basis;
      camera.zoom = kept.zoom;
      camera.target = vec.add(center, kept.pan);
    }
    redraw();
  }

  function setView(view) {
    "Set a view of the camera (or keep the rotation if null) and fit the scene.";
    const basis = camera.basis;
    reset(view ?? undefined);
    if (view == null) camera.basis = basis;
    redraw();
  }

  update(scene);

  return {
    root,
    camera,
    update,
    setView,
    redraw,
    dispose() {
      observer.disconnect();
      // the context is released at once (a page has a limited number of contexts)
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      root.remove();
    },
  };
}
