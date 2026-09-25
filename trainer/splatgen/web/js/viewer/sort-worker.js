// Back-to-front depth sort for the splat viewer (16-bit counting sort).
let positions = null;
let count = 0;

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.positions) {
    positions = msg.positions;
    count = positions.length / 3;
    return;
  }
  if (!msg.view || !positions) return;
  const v = msg.view; // column-major world->camera; camera z is row 2
  const depths = new Float32Array(count);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < count; i++) {
    const d = v[2] * positions[3 * i] + v[6] * positions[3 * i + 1] + v[10] * positions[3 * i + 2] + v[14];
    depths[i] = d;
    if (d < min) min = d;
    if (d > max) max = d;
  }
  const buckets = 65536;
  const scale = (buckets - 1) / Math.max(1e-9, max - min);
  const keys = new Uint16Array(count);
  const counts = new Uint32Array(buckets);
  for (let i = 0; i < count; i++) {
    // Far first: invert so the largest depth gets the smallest key.
    const k = (buckets - 1) - Math.floor((depths[i] - min) * scale);
    keys[i] = k;
    counts[k]++;
  }
  const starts = new Uint32Array(buckets);
  for (let k = 1; k < buckets; k++) starts[k] = starts[k - 1] + counts[k - 1];
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[starts[keys[i]]++] = i;
  self.postMessage({ order, token: msg.token }, [order.buffer]);
};
