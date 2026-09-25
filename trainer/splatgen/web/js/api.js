// Thin client for the local SplatGen API.

async function request(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch { /* keep status */ }
    throw new Error(message);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response;
}

const q = (params) => new URLSearchParams(params).toString();

export const api = {
  system: () => request("GET", "/api/system"),
  settings: () => request("GET", "/api/settings"),
  saveSettings: (data) => request("PUT", "/api/settings", data),
  roots: () => request("GET", "/api/fs/roots"),
  list: (path) => request("GET", `/api/fs/list?${q({ path })}`),
  dataset: (path) => request("GET", `/api/dataset?${q({ path })}`),
  datasetPoints: async (path) => new Float32Array(await (await request("GET", `/api/dataset/points?${q({ path })}`)).arrayBuffer()),
  thumbUrl: (path, index, size = 320) => `/api/dataset/image?${q({ path, index, size })}`,
  presets: () => request("GET", "/api/presets"),
  runs: () => request("GET", "/api/runs"),
  run: (id) => request("GET", `/api/runs/${id}`),
  metrics: (id) => request("GET", `/api/runs/${id}/metrics`),
  startRun: (data) => request("POST", "/api/runs", data),
  pause: (id) => request("POST", `/api/runs/${id}/pause`),
  resume: (id, extraSteps = 0) => request("POST", `/api/runs/${id}/resume`, { extra_steps: extraSteps }),
  stop: (id) => request("POST", `/api/runs/${id}/stop`),
  remove: (id) => request("DELETE", `/api/runs/${id}`),
  rename: (id, name) => request("PATCH", `/api/runs/${id}`, { name }),
  exportRun: (id, format, destination) => request("POST", `/api/runs/${id}/export`, { format, destination }),
  importPly: (path, name) => request("POST", "/api/import", { path, name }),
  splatBuffer: async (id) => (await request("GET", `/api/runs/${id}/splat?t=${Date.now()}`)).arrayBuffer(),
  previewUrl: (id, camera, size = 720) => `/api/runs/${id}/preview?${q({ camera, size, t: Date.now() })}`,
  fileUrl: (id, name) => `/api/runs/${id}/files/${encodeURIComponent(name)}`,
};
