// Splat file loaders -> SplatData { count, positions, covariances, colors }
//   positions    Float32Array(3N)  world xyz
//   covariances  Float32Array(6N)  xx xy xz yy yz zz of the 3D covariance
//   colors       Uint8Array(4N)    rgba (alpha = opacity)

const SH_C0 = 0.28209479177387814;

function covariance(out, o, sx, sy, sz, qw, qx, qy, qz) {
  const n = Math.hypot(qw, qx, qy, qz) || 1;
  qw /= n; qx /= n; qy /= n; qz /= n;
  const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
  const r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
  const r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);
  // M = R * S ; cov = M M^T
  const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
  const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
  const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;
  out[o] = m00 * m00 + m01 * m01 + m02 * m02;
  out[o + 1] = m00 * m10 + m01 * m11 + m02 * m12;
  out[o + 2] = m00 * m20 + m01 * m21 + m02 * m22;
  out[o + 3] = m10 * m10 + m11 * m11 + m12 * m12;
  out[o + 4] = m10 * m20 + m11 * m21 + m12 * m22;
  out[o + 5] = m20 * m20 + m21 * m21 + m22 * m22;
}

function allocate(count) {
  return {
    count,
    positions: new Float32Array(count * 3),
    covariances: new Float32Array(count * 6),
    colors: new Uint8Array(count * 4),
  };
}

export function parseSplat(buffer) {
  const count = Math.floor(buffer.byteLength / 32);
  const f = new Float32Array(buffer, 0, count * 8);
  const u = new Uint8Array(buffer);
  const data = allocate(count);
  for (let i = 0; i < count; i++) {
    data.positions[3 * i] = f[8 * i];
    data.positions[3 * i + 1] = f[8 * i + 1];
    data.positions[3 * i + 2] = f[8 * i + 2];
    const b = 32 * i;
    data.colors[4 * i] = u[b + 24];
    data.colors[4 * i + 1] = u[b + 25];
    data.colors[4 * i + 2] = u[b + 26];
    data.colors[4 * i + 3] = u[b + 27];
    covariance(data.covariances, 6 * i, f[8 * i + 3], f[8 * i + 4], f[8 * i + 5],
      (u[b + 28] - 128) / 128, (u[b + 29] - 128) / 128, (u[b + 30] - 128) / 128, (u[b + 31] - 128) / 128);
  }
  return data;
}

const PLY_TYPES = {
  float: ["getFloat32", 4], float32: ["getFloat32", 4], double: ["getFloat64", 8], float64: ["getFloat64", 8],
  uchar: ["getUint8", 1], uint8: ["getUint8", 1], char: ["getInt8", 1], int8: ["getInt8", 1],
  short: ["getInt16", 2], int16: ["getInt16", 2], ushort: ["getUint16", 2], uint16: ["getUint16", 2],
  int: ["getInt32", 4], int32: ["getInt32", 4], uint: ["getUint32", 4], uint32: ["getUint32", 4],
};

export function parsePly(buffer) {
  const bytes = new Uint8Array(buffer);
  const marker = "end_header\n";
  let end = -1;
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 65536)));
  const at = head.indexOf(marker);
  if (at < 0) throw new Error("Not a PLY file");
  end = new TextEncoder().encode(head.slice(0, at + marker.length)).length;
  const lines = head.slice(0, at).split("\n");
  if (!lines.some(l => l.includes("binary_little_endian"))) throw new Error("Only binary little-endian PLY files are supported");
  let count = 0, inVertex = false;
  const props = [];
  for (const line of lines) {
    const p = line.trim().split(/\s+/);
    if (p[0] === "element") { inVertex = p[1] === "vertex"; if (inVertex) count = parseInt(p[2]); }
    else if (p[0] === "property" && inVertex) {
      if (p[1] === "list") throw new Error("Unsupported PLY vertex property list");
      props.push({ name: p[2], type: PLY_TYPES[p[1]] });
    }
  }
  const offsets = {};
  let stride = 0;
  for (const p of props) { offsets[p.name] = [stride, p.type[0]]; stride += p.type[1]; }
  const view = new DataView(buffer, end);
  const get = (i, name) => { const [o, fn] = offsets[name]; return view[fn](i * stride + o, true); };
  const data = allocate(count);
  const isSplat = "scale_0" in offsets && "rot_0" in offsets && "opacity" in offsets;
  if (!isSplat && !("x" in offsets)) throw new Error("PLY has no vertex positions");
  // A plain point cloud becomes small round splats sized to its density.
  let pointScale = 0.01;
  if (!isSplat && count > 1) {
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < count; i += Math.max(1, Math.floor(count / 5000))) {
      for (let k = 0; k < 3; k++) { const v = get(i, "xyz"[k]); lo[k] = Math.min(lo[k], v); hi[k] = Math.max(hi[k], v); }
    }
    const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    pointScale = diag / Math.cbrt(count) * 0.35;
  }
  const hasRGB = "red" in offsets;
  for (let i = 0; i < count; i++) {
    data.positions[3 * i] = get(i, "x");
    data.positions[3 * i + 1] = get(i, "y");
    data.positions[3 * i + 2] = get(i, "z");
    if (isSplat) {
      for (let k = 0; k < 3; k++) {
        const c = 0.5 + SH_C0 * get(i, `f_dc_${k}`);
        data.colors[4 * i + k] = Math.max(0, Math.min(255, Math.round(c * 255)));
      }
      data.colors[4 * i + 3] = Math.round(255 / (1 + Math.exp(-get(i, "opacity"))));
      covariance(data.covariances, 6 * i,
        Math.exp(get(i, "scale_0")), Math.exp(get(i, "scale_1")), Math.exp(get(i, "scale_2")),
        get(i, "rot_0"), get(i, "rot_1"), get(i, "rot_2"), get(i, "rot_3"));
    } else {
      for (let k = 0; k < 3; k++) data.colors[4 * i + k] = hasRGB ? get(i, ["red", "green", "blue"][k]) : 200;
      data.colors[4 * i + 3] = 255;
      covariance(data.covariances, 6 * i, pointScale, pointScale, pointScale, 1, 0, 0, 0);
    }
  }
  data.kind = isSplat ? "splats" : "points";
  return data;
}

export function parseFile(name, buffer) {
  const lower = name.toLowerCase();
  if (lower.endsWith(".splat")) return parseSplat(buffer);
  if (lower.endsWith(".ply")) return parsePly(buffer);
  throw new Error("Unsupported file: choose a .ply or .splat");
}
