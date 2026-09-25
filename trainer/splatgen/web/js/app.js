// SplatGen web app: router, shared state and the sidebar status.
import { api } from "./api.js";
import { fmt, h, pickPath, toast } from "./ui.js";
import * as home from "./pages/home.js";
import * as dataset from "./pages/dataset.js";
import * as train from "./pages/train.js";
import * as viewer from "./pages/viewer.js";
import * as runs from "./pages/runs.js";
import * as settings from "./pages/settings.js";

const PAGES = { home, dataset, train, viewer, runs, settings };

export const state = {
  system: null,
  datasetPath: localStorage.getItem("splatgen.dataset") || "",
  activeRun: null,
};

export function setDataset(path) {
  state.datasetPath = path || "";
  if (path) localStorage.setItem("splatgen.dataset", path);
  else localStorage.removeItem("splatgen.dataset");
}

export async function chooseDataset() {
  const path = await pickPath({
    title: "Open a dataset",
    api,
    start: state.datasetPath || undefined,
    hint: "Pick a SplatGen build folder (the timestamped folder with Dataset(Default) inside), " +
      "its Dataset(Default) folder, or any COLMAP dataset. Folders marked Dataset can be opened.",
  });
  if (!path) return null;
  try {
    const summary = await api.dataset(path);
    setDataset(path);
    toast(`Opened ${fmt.short(path)} · ${summary.camera_count} images`, "ok");
    location.hash = "#/dataset";
    return summary;
  } catch (error) {
    toast(error.message, "err", 6000);
    return null;
  }
}

export function navigate(hash) { location.hash = hash; }

let cleanup = null;
async function route() {
  const [name, ...params] = (location.hash.replace(/^#\/?/, "") || "home").split("/");
  const page = PAGES[name] || home;
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.dataset.view === (PAGES[name] ? name : "home")));
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } }
  const main = document.getElementById("main");
  main.replaceChildren();
  main.scrollTop = 0;
  try {
    cleanup = await page.render(main, params.map(decodeURIComponent)) || null;
  } catch (error) {
    console.error(error);
    main.append(h("div", { class: "page" }, h("div", { class: "alert err" }, error.message)));
  }
}

async function refreshStatus() {
  try {
    const [system, list] = await Promise.all([api.system(), api.runs()]);
    state.system = system;
    const badge = document.getElementById("system-badge");
    const gpu = system.gpu ? `<b>${system.gpu}</b>` : `<b>${system.default_device.toUpperCase()}</b>`;
    badge.innerHTML = `${gpu}<br>renderer: ${system.default_backend}${system.gsplat ? "" : " · gsplat not installed"}`;
    const live = list.runs.find(r => r.live);
    state.activeRun = live || null;
    const box = document.getElementById("active-job");
    if (live) {
      const l = live.live, latest = l.latest || {};
      const steps = l.steps || live.config.steps;
      const pct = steps ? (100 * (latest.step || 0) / steps) : 0;
      box.classList.remove("hidden");
      box.onclick = () => navigate(`#/train/${live.id}`);
      box.replaceChildren(
        h("div", { class: "row" }, h("b", { class: "ellipsis grow" }, live.name), h("span", { class: `pill ${l.status}` }, l.status)),
        h("div", { class: "muted" }, `${fmt.int(latest.step || 0)} / ${fmt.int(steps)} · ETA ${fmt.duration(latest.eta)}`),
        h("div", { class: "bar" }, h("i", { style: { width: `${pct}%` } })));
    } else {
      box.classList.add("hidden");
    }
  } catch {
    document.getElementById("system-badge").textContent = "Server not reachable";
  }
}

window.addEventListener("hashchange", route);
refreshStatus();
setInterval(refreshStatus, 2000);
route();
