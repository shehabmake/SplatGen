import { api } from "../api.js";
import { fmt, h, iconButton, pickPath, toast } from "../ui.js";

export async function render(root) {
  const [settings, system] = await Promise.all([api.settings(), api.system()]);
  const runsDir = h("input", { class: "input mono", value: settings.runs_dir });
  const select = (value, options) => {
    const el = h("select", { class: "input" }, options.map(([v, l]) => h("option", { value: v }, l)));
    el.value = value;
    return el;
  };
  const device = select(settings.device, [["auto", "Automatic"], ["cuda", "NVIDIA GPU (CUDA)"], ["cpu", "CPU"], ["mps", "Apple GPU (MPS)"]]);
  const backend = select(settings.backend, [["auto", "Automatic"], ["gsplat", "gsplat (CUDA)"], ["torch", "PyTorch"]]);

  async function save() {
    try {
      await api.saveSettings({ runs_dir: runsDir.value.trim(), device: device.value, backend: backend.value });
      toast("Settings saved", "ok");
    } catch (error) { toast(error.message, "err"); }
  }

  const rows = [
    ["App version", system.version], ["PyTorch", system.torch], ["CUDA", system.cuda ? "available" : "not available"],
    ["GPU", system.gpu || "–"], ["Apple MPS", system.mps ? "available" : "–"], ["gsplat", system.gsplat || "not installed"],
    ["Default device", system.default_device], ["Default renderer", system.default_backend],
  ];
  root.append(h("div", { class: "page" },
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Settings"), h("p", {}, "Where runs are stored and how previews are rendered."))),
    h("div", { class: "grid two" },
      h("div", { class: "card stack" },
        h("div", { class: "field" }, h("label", {}, "Runs folder"),
          h("div", { class: "row" }, runsDir, iconButton("folder", "", async () => {
            const p = await pickPath({ title: "Runs folder", api, start: runsDir.value });
            if (p) runsDir.value = p;
          }, "btn icon"))),
        h("div", { class: "fields" },
          h("div", { class: "field" }, h("label", {}, "Preview device"), device),
          h("div", { class: "field" }, h("label", {}, "Preview renderer"), backend)),
        h("div", { class: "hint faint small" }, "Training device and renderer are chosen per run on the Train page."),
        h("div", {}, h("button", { class: "btn primary", onclick: save }, "Save settings"))),
      h("div", { class: "card" }, h("h2", {}, "System"),
        h("dl", { class: "kv" }, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, String(v))])),
        !system.gsplat ? h("div", { class: "alert info", style: { marginTop: "12px" } },
          "Install gsplat for fast GPU training: pip install gsplat (needs an NVIDIA GPU and a CUDA build of PyTorch). See the README.") : null)),
  ));
}
