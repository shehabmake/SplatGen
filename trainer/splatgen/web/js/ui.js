// Small UI toolkit: DOM helper, toasts, modals, folder picker, chart, formats.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "html") el.innerHTML = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export const icons = {
  folder: '<svg viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
  file: '<svg viewBox="0 0 24 24"><path d="M6 3h8l4 4v14H6z M14 3v4h4"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 5v14l12-7z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>',
  stop: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>',
  download: '<svg viewBox="0 0 24 24"><path d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14"/></svg>',
  upload: '<svg viewBox="0 0 24 24"><path d="M12 20V9m0 0-4 4m4-4 4 4M5 4h14"/></svg>',
  cube: '<svg viewBox="0 0 24 24"><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z M4 7.5 12 12l8-4.5 M12 12v9"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13"/></svg>',
  edit: '<svg viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16z"/></svg>',
  refresh: '<svg viewBox="0 0 24 24"><path d="M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5"/></svg>',
  target: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2"/></svg>',
  camera: '<svg viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  image: '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 16 5-5 5 5 3-3 5 5"/></svg>',
  spark: '<svg viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/></svg>',
};

export function iconButton(icon, label, onClick, cls = "btn", title) {
  const button = h("button", { class: cls, onclick: onClick, title: title || label });
  button.innerHTML = `${icons[icon] || ""}${label ? `<span>${label}</span>` : ""}`;
  return button;
}

export function toast(message, kind = "info", ms = 4000) {
  const el = h("div", { class: `toast ${kind}` }, message);
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), ms);
}

export function modal({ title, body, footer, onClose, wide }) {
  const root = document.getElementById("modal-root");
  const close = () => { backdrop.remove(); document.removeEventListener("keydown", esc); if (onClose) onClose(); };
  const esc = (e) => { if (e.key === "Escape") close(); };
  const box = h("div", { class: "modal", style: wide ? { width: "min(1000px, calc(100vw - 32px))" } : {} },
    h("div", { class: "modal-head" }, h("h2", {}, title), h("div", { class: "grow" }),
      iconButton("", "✕", close, "btn ghost icon")),
    h("div", { class: "modal-body" }, body),
    footer ? h("div", { class: "modal-foot" }, footer) : null);
  const backdrop = h("div", { class: "modal-backdrop", onclick: (e) => { if (e.target === backdrop) close(); } }, box);
  root.append(backdrop);
  document.addEventListener("keydown", esc);
  return { close, box };
}

export function confirmDialog(title, message, action = "Confirm", danger = false) {
  return new Promise((resolve) => {
    let done = false;
    const m = modal({
      title, body: h("p", { class: "muted" }, message),
      footer: [
        h("button", { class: "btn", onclick: () => { done = true; m.close(); resolve(false); } }, "Cancel"),
        h("button", { class: `btn ${danger ? "danger" : "primary"}`, onclick: () => { done = true; m.close(); resolve(true); } }, action),
      ],
      onClose: () => { if (!done) resolve(false); },
    });
  });
}

export function promptDialog(title, label, value = "") {
  return new Promise((resolve) => {
    const input = h("input", { class: "input", value });
    let done = false;
    const submit = () => { done = true; m.close(); resolve(input.value.trim()); };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
    const m = modal({
      title, body: h("div", { class: "field" }, h("label", {}, label), input),
      footer: [h("button", { class: "btn", onclick: () => { done = true; m.close(); resolve(null); } }, "Cancel"),
        h("button", { class: "btn primary", onclick: submit }, "Save")],
      onClose: () => { if (!done) resolve(null); },
    });
    setTimeout(() => input.select(), 30);
  });
}

// Server-side folder / file picker (the browser cannot see local paths).
export function pickPath({ title = "Choose a folder", mode = "folder", start, api, hint }) {
  return new Promise((resolve) => {
    let current = null;
    let resolved = false;
    const list = h("div", { class: "list" });
    const crumbs = h("div", { class: "crumbs" });
    const pathInput = h("input", { class: "input mono", placeholder: "Paste a path…" });
    const status = h("span", { class: "muted small" });
    const choose = h("button", { class: "btn primary" }, mode === "folder" ? "Select this folder" : "Select file");
    const finish = (value) => { resolved = true; m.close(); resolve(value); };
    choose.onclick = () => {
      if (mode === "folder" && current) finish(current);
    };
    choose.disabled = mode !== "folder";

    async function open(path) {
      list.replaceChildren(h("div", { class: "empty" }, h("span", { class: "spin" })));
      try {
        const data = await api.list(path);
        current = data.path;
        pathInput.value = data.path;
        renderCrumbs(data.path);
        status.textContent = data.dataset ? "This folder is a dataset" : "";
        if (data.dataset) status.innerHTML = '<span class="badge">Dataset</span>';
        const items = [];
        if (data.parent && data.parent !== data.path) {
          items.push(row("folder", "..", "Up one level", () => open(data.parent)));
        }
        for (const entry of data.entries) {
          if (entry.dir) {
            const item = row("folder", entry.name, entry.dataset ? "SplatGen / COLMAP dataset" : "", () => open(entry.path));
            if (entry.dataset) item.querySelector(".title").append(" ", h("span", { class: "badge" }, "Dataset"));
            items.push(item);
          } else if (mode === "file") {
            items.push(row("file", entry.name, "", () => finish(entry.path)));
          }
        }
        if (!items.length) items.push(h("div", { class: "empty" }, "Empty folder"));
        list.replaceChildren(...items);
      } catch (error) {
        list.replaceChildren(h("div", { class: "alert err" }, error.message));
      }
    }

    function row(icon, name, sub, onclick) {
      const iconEl = h("div", { class: "icon", html: icons[icon] });
      return h("div", { class: "list-item", onclick }, iconEl,
        h("div", { class: "grow ellipsis" }, h("div", { class: "title ellipsis" }, name), sub ? h("div", { class: "sub" }, sub) : null));
    }

    function renderCrumbs(path) {
      const sep = path.includes("\\") ? "\\" : "/";
      const parts = path.split(/[\\/]/).filter(Boolean);
      const out = [];
      let acc = path.startsWith("/") ? "/" : "";
      parts.forEach((part, i) => {
        if (i === 0) acc = acc + part + (sep === "\\" ? sep : "");
        else acc = acc.endsWith(sep) ? acc + part : acc + sep + part;
        const target = acc;
        out.push(h("button", { onclick: () => open(target) }, part));
        if (i < parts.length - 1) out.push(h("span", { class: "faint" }, "›"));
      });
      crumbs.replaceChildren(...out);
    }

    pathInput.addEventListener("keydown", (e) => { if (e.key === "Enter") open(pathInput.value); });
    const rootsRow = h("div", { class: "row wrap" });
    api.roots().then(({ roots, recent }) => {
      rootsRow.replaceChildren(...roots.map(r => h("button", { class: "btn sm", onclick: () => open(r.path) }, r.name)),
        ...(recent || []).slice(0, 4).map(p => h("button", { class: "btn sm ghost", title: p, onclick: () => open(p) },
          "↺ " + p.split(/[\\/]/).filter(Boolean).slice(-2).join("/"))));
      open(start || (recent && recent[0]) || roots[0].path);
    });
    const m = modal({
      title,
      body: h("div", { class: "stack" },
        hint ? h("div", { class: "alert info" }, hint) : null,
        rootsRow, pathInput, crumbs, list),
      footer: [status, h("div", { class: "grow" }), h("button", { class: "btn", onclick: () => finish(null) }, "Cancel"),
        mode === "folder" ? choose : null],
      onClose: () => { if (!resolved) resolve(null); },
    });
  });
}

// -- formatting ---------------------------------------------------------------------
export const fmt = {
  int: (n) => (n === undefined || n === null) ? "–" : Math.round(n).toLocaleString(),
  num: (n, d = 2) => (n === undefined || n === null || Number.isNaN(n)) ? "–" : Number(n).toFixed(d),
  bytes: (b) => { if (!b && b !== 0) return "–"; const u = ["B", "KB", "MB", "GB"]; let i = 0; while (b >= 1024 && i < 3) { b /= 1024; i++; } return `${b.toFixed(i ? 1 : 0)} ${u[i]}`; },
  duration: (s) => {
    if (s === undefined || s === null || !Number.isFinite(s)) return "–";
    s = Math.max(0, Math.round(s));
    const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
    return hh ? `${hh}h ${mm}m` : mm ? `${mm}m ${ss}s` : `${ss}s`;
  },
  date: (t) => t ? new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "–",
  short: (path) => path ? path.split(/[\\/]/).filter(Boolean).slice(-3).join("/") : "",
};

// "scene · 2026-09-25 01:36" for SplatGen builds, else the folder name.
export function datasetName(path) {
  if (!path) return "";
  const parts = path.split(/[\\/]/).filter(Boolean);
  const i = parts.findIndex(p => p.startsWith("SplatGen_"));
  if (i >= 0) {
    const project = parts[i].slice("SplatGen_".length);
    const stamp = parts[i + 1] && /^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}/.test(parts[i + 1])
      ? parts[i + 1].slice(0, 16).replace("_", " ").replace(/-(\d{2})-(\d{2})$/, " $1:$2").replace(/ (\d{2})-(\d{2})/, " $1:$2") : "";
    return stamp ? `${project} · ${stamp}` : project;
  }
  const last = parts[parts.length - 1] || path;
  return last === "Dataset(Default)" ? (parts[parts.length - 2] || last) : last;
}

export function pill(status) {
  return h("span", { class: `pill ${status}` }, status);
}

// -- line chart ---------------------------------------------------------------------
export class LineChart {
  constructor(canvas, series) {
    this.canvas = canvas;
    this.series = series; // [{key, label, color, axis: 'left'|'right'}]
    this.data = [];
  }

  set(data) { this.data = data; this.draw(); }

  draw() {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, hgt = c.clientHeight;
    if (!w || !hgt) return;
    c.width = w * dpr; c.height = hgt * dpr;
    const ctx = c.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, hgt);
    const pad = { l: 44, r: 44, t: 12, b: 22 };
    const iw = w - pad.l - pad.r, ih = hgt - pad.t - pad.b;
    const data = this.data;
    ctx.font = "11px Inter, Segoe UI, sans-serif";
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.fillStyle = "#6f7a8c";
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + ih * i / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iw, y); ctx.stroke();
    }
    if (data.length < 2) {
      ctx.fillText("Waiting for data…", pad.l + 8, pad.t + 18);
      return;
    }
    const xs = data.map(d => d.step);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const X = (v) => pad.l + (v - x0) / Math.max(1, x1 - x0) * iw;
    this.series.forEach((s, si) => {
      const values = data.map(d => d[s.key]).filter(v => Number.isFinite(v));
      if (!values.length) return;
      let lo = Math.min(...values), hi = Math.max(...values);
      if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
      const Y = (v) => pad.t + ih - (v - lo) / (hi - lo) * ih;
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.8; ctx.beginPath();
      data.forEach((d, i) => { const v = d[s.key]; if (!Number.isFinite(v)) return; i ? ctx.lineTo(X(d.step), Y(v)) : ctx.moveTo(X(d.step), Y(v)); });
      ctx.stroke();
      ctx.fillStyle = s.color;
      const left = s.axis !== "right";
      ctx.textAlign = left ? "right" : "left";
      const ax = left ? pad.l - 6 : pad.l + iw + 6;
      const f = s.format || ((v) => v.toFixed(3));
      ctx.fillText(f(hi), ax, pad.t + 8);
      ctx.fillText(f(lo), ax, pad.t + ih);
      ctx.textAlign = "left";
      ctx.fillText(s.label, pad.l + 8 + si * 90, pad.t + ih + 16);
    });
    ctx.fillStyle = "#6f7a8c"; ctx.textAlign = "right";
    ctx.fillText(`step ${x1.toLocaleString()}`, pad.l + iw, pad.t + ih + 16);
  }
}
