// WebGL2 Gaussian splat viewer.
//
// Cameras follow the dataset convention (OpenCV: +x right, +y down, +z
// forward), so a dataset camera can be shown exactly as it was rendered.
// Splats are drawn back to front with premultiplied alpha after a depth
// sort in a worker; each splat is an instanced quad spanning 3 sigma of its
// projected 2D Gaussian (with the same 0.3 px low-pass the trainer uses).

const PER_ROW = 680;          // splats per texture row (3 texels each)

const SPLAT_VS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform usampler2D uData;
uniform mat4 uView;
uniform vec2 uFocal;
uniform vec2 uCenter;
uniform vec2 uViewport;
in vec2 aCorner;
in uint aIndex;
out vec4 vColor;
out vec2 vPos;
const uint PER_ROW = ${PER_ROW}u;
void main() {
  uint row = aIndex / PER_ROW;
  uint col = (aIndex - row * PER_ROW) * 3u;
  uvec4 t0 = texelFetch(uData, ivec2(col, row), 0);
  vec4 cam = uView * vec4(uintBitsToFloat(t0.xyz), 1.0);
  if (cam.z < 0.02) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec2 pix = vec2(uFocal.x * cam.x / cam.z, uFocal.y * cam.y / cam.z) + uCenter;
  vec2 ndc = vec2(pix.x / uViewport.x * 2.0 - 1.0, 1.0 - pix.y / uViewport.y * 2.0);
  if (abs(ndc.x) > 1.3 || abs(ndc.y) > 1.3) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 c1 = uintBitsToFloat(texelFetch(uData, ivec2(col + 1u, row), 0));
  vec2 c2 = uintBitsToFloat(texelFetch(uData, ivec2(col + 2u, row), 0).xy);
  mat3 S = mat3(c1.x, c1.y, c1.z, c1.y, c1.w, c2.x, c1.z, c2.x, c2.y);
  mat3 R = mat3(uView);
  mat3 Sc = R * S * transpose(R);
  float z2 = cam.z * cam.z;
  vec3 r0 = vec3(uFocal.x / cam.z, 0.0, -uFocal.x * cam.x / z2);
  vec3 r1 = vec3(0.0, uFocal.y / cam.z, -uFocal.y * cam.y / z2);
  float a = dot(r0, Sc * r0) + 0.3;
  float b = dot(r0, Sc * r1);
  float c = dot(r1, Sc * r1) + 0.3;
  float mid = 0.5 * (a + c);
  float rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.1);
  vec2 v1 = abs(b) > 1e-9 ? normalize(vec2(b, l1 - a)) : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 v2 = vec2(-v1.y, v1.x);
  vec2 off = aCorner.x * 3.0 * min(sqrt(l1), 2048.0) * v1 + aCorner.y * 3.0 * min(sqrt(l2), 2048.0) * v2;
  gl_Position = vec4(ndc + vec2(off.x, -off.y) / uViewport * 2.0, 0.0, 1.0);
  vPos = aCorner * 3.0;
  uint rgba = t0.w;
  vColor = vec4(float(rgba & 255u), float((rgba >> 8) & 255u), float((rgba >> 16) & 255u), float(rgba >> 24)) / 255.0;
}`;

const SPLAT_FS = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vPos;
out vec4 fragColor;
void main() {
  float d = dot(vPos, vPos);
  if (d > 9.0) discard;
  float a = vColor.a * exp(-0.5 * d);
  if (a < 1.0 / 255.0) discard;
  fragColor = vec4(vColor.rgb * a, a);
}`;

const POINT_VS = `#version 300 es
precision highp float;
uniform mat4 uView;
uniform vec2 uFocal, uCenter, uViewport;
uniform float uSize;
in vec3 aPos;
in vec4 aColor;
out vec4 vColor;
void main() {
  vec4 cam = uView * vec4(aPos, 1.0);
  if (cam.z < 0.02) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec2 pix = vec2(uFocal.x * cam.x / cam.z, uFocal.y * cam.y / cam.z) + uCenter;
  gl_Position = vec4(pix.x / uViewport.x * 2.0 - 1.0, 1.0 - pix.y / uViewport.y * 2.0, 0.5, 1.0);
  gl_PointSize = uSize;
  vColor = aColor;
}`;

const POINT_FS = `#version 300 es
precision highp float;
in vec4 vColor;
uniform float uRound;
out vec4 fragColor;
void main() {
  if (uRound > 0.5 && length(gl_PointCoord - 0.5) > 0.5) discard;
  fragColor = vColor;
}`;

function compile(gl, vs, fs) {
  const program = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const uniforms = {};
  const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(program, i).name;
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  return { program, uniforms };
}

// -- small vector helpers ------------------------------------------------------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function basis(up) {
  const helper = Math.abs(up[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const a1 = norm(cross(helper, up));
  const a2 = cross(up, a1);
  return [a1, a2];
}

export const UP_AXES = { "+z": [0, 0, 1], "+y": [0, 1, 0], "-y": [0, -1, 0], "-z": [0, 0, -1] };

export class SplatViewer {
  constructor(container, options = {}) {
    this.container = container;
    this.canvas = document.createElement("canvas");
    this.canvas.tabIndex = 0;
    container.prepend(this.canvas);
    const gl = this.canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.splatProgram = compile(gl, SPLAT_VS, SPLAT_FS);
    this.pointProgram = compile(gl, POINT_VS, POINT_FS);
    this.background = options.background || [0.027, 0.031, 0.043];
    this.onInfo = options.onInfo || (() => {});
    this.up = UP_AXES[options.up || "+z"];
    this.fov = 55 * Math.PI / 180;
    this.target = [0, 0, 0];
    this.distance = 5;
    this.yaw = 0.8;
    this.pitch = 0.35;
    this.fixedView = null;       // {view, fx, fy, cx, cy, w, h} when showing a dataset camera
    this.splats = null;
    this.points = null;
    this.lines = null;
    this.showSplats = true;
    this.showPoints = true;
    this.showCameras = true;
    this.pointSize = 2;
    this.keys = new Set();
    this.dirty = true;
    this.sortToken = 0;
    this.sortPending = false;
    this.lastSortView = null;
    this.frames = 0;
    this.fpsTime = performance.now();
    this.fps = 0;

    const quad = new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]);
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    this.indexBuffer = gl.createBuffer();
    this.texture = gl.createTexture();
    this.splatVao = gl.createVertexArray();
    gl.bindVertexArray(this.splatVao);
    const sp = this.splatProgram.program;
    const aCorner = gl.getAttribLocation(sp, "aCorner");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
    const aIndex = gl.getAttribLocation(sp, "aIndex");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.enableVertexAttribArray(aIndex);
    gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(aIndex, 1);
    gl.bindVertexArray(null);

    this.worker = new Worker(new URL("./sort-worker.js", import.meta.url));
    this.worker.onmessage = (e) => this._onSorted(e.data);

    this._bindControls();
    this._resizeObserver = new ResizeObserver(() => { this.dirty = true; });
    this._resizeObserver.observe(container);
    this._raf = requestAnimationFrame(this._frame.bind(this));
  }

  destroy() {
    cancelAnimationFrame(this._raf);
    this._resizeObserver.disconnect();
    this.worker.terminate();
    this._unbind.forEach(fn => fn());
    const lose = this.gl.getExtension("WEBGL_lose_context");
    if (lose) lose.loseContext();
    this.canvas.remove();
  }

  // -- data --------------------------------------------------------------------------

  setSplats(data, { keepView = false } = {}) {
    const gl = this.gl;
    this.splats = data;
    if (!data || !data.count) { this.dirty = true; return; }
    const rows = Math.ceil(data.count / PER_ROW);
    const tex = new Uint32Array(PER_ROW * 3 * 4 * rows);
    const texF = new Float32Array(tex.buffer);
    const { positions: p, covariances: c, colors: col } = data;
    for (let i = 0; i < data.count; i++) {
      const o = i * 12;
      texF[o] = p[3 * i]; texF[o + 1] = p[3 * i + 1]; texF[o + 2] = p[3 * i + 2];
      tex[o + 3] = col[4 * i] | (col[4 * i + 1] << 8) | (col[4 * i + 2] << 16) | (col[4 * i + 3] << 24);
      texF[o + 4] = c[6 * i]; texF[o + 5] = c[6 * i + 1]; texF[o + 6] = c[6 * i + 2]; texF[o + 7] = c[6 * i + 3];
      texF[o + 8] = c[6 * i + 4]; texF[o + 9] = c[6 * i + 5];
    }
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, PER_ROW * 3, rows, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.drawCount = 0;
    this.worker.postMessage({ positions: data.positions.slice() });
    this.lastSortView = null;
    if (!keepView) this.fit();
    this.dirty = true;
  }

  setPoints(xyzrgb) {
    const gl = this.gl;
    if (!xyzrgb || !xyzrgb.length) { this.points = null; this.dirty = true; return; }
    const n = xyzrgb.length / 6;
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const interleaved = new Float32Array(n * 7);
    for (let i = 0; i < n; i++) {
      interleaved.set(xyzrgb.subarray(6 * i, 6 * i + 6), 7 * i);
      interleaved[7 * i + 6] = 1;
    }
    gl.bufferData(gl.ARRAY_BUFFER, interleaved, gl.STATIC_DRAW);
    this.points = { buffer, count: n, xyz: xyzrgb };
    if (!this.splats) this.fit();
    this.dirty = true;
  }

  setCameras(cameras, highlight = -1) {
    // Frustum wireframes from OpenCV camera-to-world matrices.
    const gl = this.gl;
    if (!cameras || !cameras.length) { this.lines = null; this.dirty = true; return; }
    this.cameras = cameras;
    const radius = this._sceneRadius() || 1;
    const size = radius * 0.06;
    const verts = [];
    cameras.forEach((cam, idx) => {
      const m = cam.camera_to_world;
      const center = [m[0][3], m[1][3], m[2][3]];
      const hw = cam.width / (2 * cam.fx) * size, hh = cam.height / (2 * cam.fy) * size;
      const corner = (x, y) => [0, 1, 2].map(r => m[r][3] + m[r][0] * x + m[r][1] * y + m[r][2] * size);
      const cs = [corner(-hw, -hh), corner(hw, -hh), corner(hw, hh), corner(-hw, hh)];
      const color = idx === highlight ? [1, 0.8, 0.3, 1] : [0.55, 0.45, 1, 0.9];
      const seg = (a, b) => verts.push(...a, ...color, ...b, ...color);
      cs.forEach((c, i) => { seg(center, c); seg(c, cs[(i + 1) % 4]); });
      // A small tick on the top edge marks "up" in the image.
      const top = [0, 1, 2].map(r => (cs[0][r] + cs[1][r]) / 2);
      const tip = [0, 1, 2].map(r => top[r] - m[r][1] * hh * 0.6);
      seg(cs[0], tip); seg(tip, cs[1]);
    });
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
    this.lines = { buffer, count: verts.length / 7 };
    this.dirty = true;
  }

  setUp(name) {
    this.up = UP_AXES[name] || UP_AXES["+z"];
    this.fixedView = null;
    this.dirty = true;
  }

  // -- camera ------------------------------------------------------------------------

  _sceneRadius() {
    const src = this.splats ? this.splats.positions : this.points ? this.points.xyz : null;
    if (!src) return null;
    const stride = this.splats ? 3 : 6;
    const n = src.length / stride;
    const step = Math.max(1, Math.floor(n / 20000));
    const xs = [[], [], []];
    for (let i = 0; i < n; i += step) for (let k = 0; k < 3; k++) xs[k].push(src[i * stride + k]);
    const med = xs.map(a => a.sort((x, y) => x - y)[Math.floor(a.length / 2)]);
    const dists = [];
    for (let j = 0; j < xs[0].length; j++) dists.push(Math.hypot(xs[0][j] - med[0], xs[1][j] - med[1], xs[2][j] - med[2]));
    dists.sort((x, y) => x - y);
    this._center = med;
    return dists[Math.floor(dists.length * 0.9)] || 1;
  }

  fit() {
    const radius = this._sceneRadius();
    if (!radius) return;
    this.target = this._center.slice();
    this.distance = radius / Math.tan(this.fov / 2) * 1.1;
    this.fixedView = null;
    this.dirty = true;
  }

  eye() {
    const [a1, a2] = basis(this.up);
    const cp = Math.cos(this.pitch);
    const dir = add(add(scale(a1, cp * Math.cos(this.yaw)), scale(a2, cp * Math.sin(this.yaw))), scale(this.up, Math.sin(this.pitch)));
    return add(this.target, scale(dir, this.distance));
  }

  viewMatrix() {
    if (this.fixedView) return this.fixedView.view;
    const eye = this.eye();
    const f = norm(sub(this.target, eye));
    let r = cross(f, this.up);
    if (Math.hypot(...r) < 1e-6) r = basis(this.up)[0];
    r = norm(r);
    const d = cross(f, r);
    const t = [-dot(r, eye), -dot(d, eye), -dot(f, eye)];
    // column-major
    return new Float32Array([r[0], d[0], f[0], 0, r[1], d[1], f[1], 0, r[2], d[2], f[2], 0, t[0], t[1], t[2], 1]);
  }

  viewFromCamera(cam) {
    // cam: {camera_to_world (OpenCV, rows), fx, fy, cx, cy, width, height}
    const m = cam.camera_to_world;
    const R = [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]; // world->cam rows
    const c = [m[0][3], m[1][3], m[2][3]];
    const t = R.map(row => -dot(row, c));
    const view = new Float32Array([R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0, t[0], t[1], t[2], 1]);
    this.fixedView = { view, cam };
    // Keep the orbit consistent so dragging continues from this viewpoint.
    const forward = [m[0][2], m[1][2], m[2][2]];
    const dist = this.distance || 5;
    this.target = add(c, scale(forward, dist));
    const back = scale(forward, -1);
    const [a1, a2] = basis(this.up);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, dot(back, this.up))));
    this.yaw = Math.atan2(dot(back, a2), dot(back, a1));
    this.dirty = true;
  }

  _intrinsics(w, h) {
    if (this.fixedView) {
      const cam = this.fixedView.cam;
      const s = Math.min(w / cam.width, h / cam.height);
      return { fx: cam.fx * s, fy: cam.fy * s, cx: w / 2 + (cam.cx - cam.width / 2) * s, cy: h / 2 + (cam.cy - cam.height / 2) * s };
    }
    const f = 0.5 * h / Math.tan(this.fov / 2);
    return { fx: f, fy: f, cx: w / 2, cy: h / 2 };
  }

  _releaseFixed() {
    if (this.fixedView) this.fixedView = null;
  }

  // -- input -----------------------------------------------------------------------------

  _bindControls() {
    const c = this.canvas;
    let drag = null;
    const on = (el, type, fn, opts) => { el.addEventListener(type, fn, opts); return () => el.removeEventListener(type, fn, opts); };
    this._unbind = [
      on(c, "pointerdown", (e) => {
        c.focus();
        c.setPointerCapture(e.pointerId);
        drag = { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey };
      }),
      on(c, "pointermove", (e) => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.x = e.clientX; drag.y = e.clientY;
        this._releaseFixed();
        if (drag.button === 2 || drag.button === 1 || drag.shift) {
          const f = norm(sub(this.target, this.eye()));
          const r = norm(cross(f, this.up));
          const u = cross(r, f);
          const k = this.distance * 0.0015;
          this.target = add(this.target, add(scale(r, -dx * k), scale(u, dy * k)));
        } else {
          this.yaw -= dx * 0.005;
          this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.005));
        }
        this.dirty = true;
      }),
      on(c, "pointerup", () => { drag = null; }),
      on(c, "contextmenu", (e) => e.preventDefault()),
      on(c, "wheel", (e) => {
        e.preventDefault();
        this._releaseFixed();
        this.distance *= Math.exp(e.deltaY * 0.001);
        this.distance = Math.max(1e-3, this.distance);
        this.dirty = true;
      }, { passive: false }),
      on(c, "dblclick", () => this.fit()),
      on(c, "keydown", (e) => { this.keys.add(e.key.toLowerCase()); }),
      on(c, "keyup", (e) => { this.keys.delete(e.key.toLowerCase()); }),
      on(c, "blur", () => this.keys.clear()),
    ];
  }

  _applyKeys(dt) {
    if (!this.keys.size) return;
    const speed = this.distance * 0.8 * dt * (this.keys.has("shift") ? 3 : 1);
    const f = norm(sub(this.target, this.eye()));
    const r = norm(cross(f, this.up));
    let move = [0, 0, 0];
    if (this.keys.has("w") || this.keys.has("arrowup")) move = add(move, f);
    if (this.keys.has("s") || this.keys.has("arrowdown")) move = sub(move, f);
    if (this.keys.has("d") || this.keys.has("arrowright")) move = add(move, r);
    if (this.keys.has("a") || this.keys.has("arrowleft")) move = sub(move, r);
    if (this.keys.has("e")) move = add(move, this.up);
    if (this.keys.has("q")) move = sub(move, this.up);
    if (move[0] || move[1] || move[2]) {
      this._releaseFixed();
      this.target = add(this.target, scale(move, speed));
      this.dirty = true;
    }
  }

  // -- drawing -----------------------------------------------------------------------------

  _onSorted({ order, token }) {
    this.sortPending = false;
    if (!this.splats || order.length !== this.splats.count) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
    this.drawCount = order.length;
    this.dirty = true;
  }

  _requestSort(view) {
    if (!this.splats || this.sortPending) return;
    const last = this.lastSortView;
    if (last) {
      let diff = 0;
      for (const i of [2, 6, 10]) diff += Math.abs(view[i] - last[i]);
      diff += Math.abs(view[14] - last[14]) / (this.distance || 1);
      if (diff < 0.01) return;
    }
    this.lastSortView = view.slice();
    this.sortPending = true;
    this.worker.postMessage({ view, token: ++this.sortToken });
  }

  _frame(now) {
    const dt = Math.min(0.05, (now - (this._last || now)) / 1000);
    this._last = now;
    this._applyKeys(dt);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.container.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.container.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h; this.dirty = true;
    }
    if (this.dirty) {
      this.dirty = false;
      this._draw(w, h);
      this.frames++;
    }
    if (now - this.fpsTime > 1000) {
      this.fps = this.frames * 1000 / (now - this.fpsTime);
      this.frames = 0; this.fpsTime = now;
      this.onInfo({ fps: this.fps, splats: this.splats ? this.splats.count : 0, points: this.points ? this.points.count : 0 });
    }
    this._raf = requestAnimationFrame(this._frame.bind(this));
  }

  _draw(w, h) {
    const gl = this.gl;
    const view = this.viewMatrix();
    const K = this._intrinsics(w, h);
    gl.viewport(0, 0, w, h);
    gl.clearColor(this.background[0], this.background[1], this.background[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);

    const setCommon = (u) => {
      gl.uniformMatrix4fv(u.uView, false, view);
      gl.uniform2f(u.uFocal, K.fx, K.fy);
      gl.uniform2f(u.uCenter, K.cx, K.cy);
      gl.uniform2f(u.uViewport, w, h);
    };

    if (this.showPoints && this.points && !(this.showSplats && this.splats)) {
      this._drawPoints(this.points, gl.POINTS, setCommon, this.pointSize * Math.min(window.devicePixelRatio || 1, 2), true);
    }
    if (this.showSplats && this.splats && this.drawCount) {
      const { program, uniforms: u } = this.splatProgram;
      gl.useProgram(program);
      setCommon(u);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(u.uData, 0);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this.splatVao);
      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.drawCount);
      gl.bindVertexArray(null);
    }
    if (this.showCameras && this.lines) {
      this._drawPoints(this.lines, gl.LINES, setCommon, 1, false);
    }
    this._requestSort(view);
  }

  _drawPoints(item, mode, setCommon, size, round) {
    const gl = this.gl;
    const { program, uniforms: u } = this.pointProgram;
    gl.useProgram(program);
    setCommon(u);
    gl.uniform1f(u.uSize, size);
    gl.uniform1f(u.uRound, round ? 1 : 0);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindBuffer(gl.ARRAY_BUFFER, item.buffer);
    const aPos = gl.getAttribLocation(program, "aPos");
    const aColor = gl.getAttribLocation(program, "aColor");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(aColor);
    gl.vertexAttribPointer(aColor, 4, gl.FLOAT, false, 28, 12);
    gl.drawArrays(mode, 0, item.count);
    gl.disableVertexAttribArray(aPos);
    gl.disableVertexAttribArray(aColor);
  }

  snapshot() {
    return this.canvas.toDataURL("image/png");
  }
}
