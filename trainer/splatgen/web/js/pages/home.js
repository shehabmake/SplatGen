import { api } from "../api.js";
import { chooseDataset, navigate, setDataset, state } from "../app.js";
import { datasetName, fmt, h, icons, iconButton, pill } from "../ui.js";

export async function render(root) {
  const recentDatasets = h("div", { class: "list" }, h("div", { class: "empty" }, h("span", { class: "spin" })));
  const recentRuns = h("div", { class: "list" }, h("div", { class: "empty" }, h("span", { class: "spin" })));
  const system = h("div", {}, h("span", { class: "spin" }));

  root.append(h("div", { class: "page" },
    h("div", { class: "hero" },
      h("h1", {}, "Train Gaussian splats from your 3D scenes"),
      h("p", {}, "Open a dataset exported by the SplatGen Blender add-on — or any COLMAP dataset — " +
        "train it with gsplat, watch it improve live, and export a standard 3DGS PLY."),
      h("div", { class: "row wrap" },
        iconButton("folder", "Open dataset", chooseDataset, "btn primary big"),
        iconButton("cube", "View a splat file", () => navigate("#/viewer"), "btn big"),
        state.datasetPath ? iconButton("spark", "Continue with last dataset", () => navigate("#/dataset"), "btn ghost big") : null)),
    h("div", { class: "spacer" }),
    h("div", { class: "grid three" },
      h("div", { class: "card" }, h("h2", {}, "Recent datasets"), recentDatasets),
      h("div", { class: "card" }, h("h2", {}, "Recent runs"), recentRuns),
      h("div", { class: "card" }, h("h2", {}, "This computer"), system)),
  ));

  const [roots, runs, info] = await Promise.all([api.roots(), api.runs(), api.system()]);
  const datasets = roots.recent || [];
  recentDatasets.replaceChildren(...(datasets.length ? datasets.slice(0, 6).map(path =>
    h("div", { class: "list-item", onclick: () => { setDataset(path); navigate("#/dataset"); }, title: path },
      h("div", { class: "icon", html: icons.folder }),
      h("div", { class: "grow ellipsis" }, h("div", { class: "title ellipsis" }, datasetName(path)),
        h("div", { class: "sub ellipsis" }, path))))
    : [h("div", { class: "empty" }, "No datasets opened yet.")]));

  recentRuns.replaceChildren(...(runs.runs.length ? runs.runs.slice(0, 6).map(run =>
    h("div", { class: "list-item", onclick: () => navigate(run.kind === "imported" ? `#/viewer/${run.id}` : `#/train/${run.id}`) },
      h("div", { class: "icon", html: icons.cube }),
      h("div", { class: "grow ellipsis" }, h("div", { class: "title ellipsis" }, run.name),
        h("div", { class: "sub" }, `${fmt.date(run.created)} · ${fmt.int(run.results?.gaussians)} splats`)),
      pill(run.live ? run.live.status : run.status)))
    : [h("div", { class: "empty" }, "No training runs yet.")]));

  const rows = [
    ["Device", info.gpu || info.default_device.toUpperCase()],
    ["Renderer", info.default_backend === "gsplat" ? "gsplat (CUDA)" : "PyTorch (slow)"],
    ["gsplat", info.gsplat || "not installed"],
    ["PyTorch", info.torch],
    ["App version", info.version],
  ];
  system.replaceChildren(...[
    h("dl", { class: "kv" }, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
    info.default_backend !== "gsplat" ? h("div", { class: "alert warn", style: { marginTop: "12px" } },
      info.cuda ? "CUDA is available but gsplat is not installed — install it for 50–100× faster training (see README)."
        : "No NVIDIA GPU detected. Training will use the PyTorch renderer, which is only practical for small scenes.") : null].filter(Boolean));
}
