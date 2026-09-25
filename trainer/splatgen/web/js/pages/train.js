import { api } from "../api.js";
import { chooseDataset, navigate, state } from "../app.js";
import { datasetName, fmt, h, icons, iconButton, LineChart, pickPath, pill, promptDialog, toast } from "../ui.js";

// Advanced settings, grouped. Every key is a TrainConfig field.
const GROUPS = [
  ["Schedule", [
    ["steps", "Training steps", "int", "More steps = sharper, slower."],
    ["sh_degree", "View-dependent color (SH degree)", "select", "0 = flat color, 3 = full reflections.", [[0, "0"], [1, "1"], [2, "2"], [3, "3"]]],
    ["sh_degree_interval", "Raise SH degree every", "int", "steps"],
  ]],
  ["Images", [
    ["downscale", "Image resolution", "select", "Lower resolution trains much faster.", [[1, "Full"], [2, "1/2"], [4, "1/4"], [8, "1/8"]]],
    ["test_every", "Hold out every Nth image", "int", "Used to measure quality on unseen views. 0 = train on all."],
    ["background", "Background", "select", "", [["black", "Black"], ["white", "White"], ["random", "Random"]]],
    ["mask_mode", "Masks", "select", "Ignore = only pixels inside the mask count.", [["none", "Not used"], ["ignore", "Ignore background"]]],
  ]],
  ["Starting splats", [
    ["init", "Initialize from", "select", "", [["points", "Sparse points (points3D)"], ["random", "Random points"], ["ply", "Existing .ply"]]],
    ["init_ply", "PLY to start from", "path", "Only for 'Existing .ply'."],
    ["init_random_count", "Random points", "int", ""],
    ["init_opacity", "Initial opacity", "float", ""],
  ]],
  ["Densification", [
    ["densify_start", "Start at step", "int", ""],
    ["densify_stop", "Stop at step", "int", ""],
    ["densify_every", "Every N steps", "int", ""],
    ["densify_grad_threshold", "Gradient threshold", "float", "Lower = more splats."],
    ["reset_opacity_every", "Reset opacity every", "int", "Removes floaters."],
    ["prune_opacity", "Prune below opacity", "float", ""],
    ["max_gaussians", "Maximum splats", "int", "Growth stops here. 0 = unlimited."],
  ]],
  ["Learning rates", [
    ["lr_means", "Position", "float", "Multiplied by the scene size."],
    ["lr_means_final", "Position final factor", "float", ""],
    ["lr_scales", "Scale", "float", ""],
    ["lr_quats", "Rotation", "float", ""],
    ["lr_opacities", "Opacity", "float", ""],
    ["lr_sh0", "Color", "float", ""],
    ["lr_shN", "View-dependent color", "float", ""],
    ["ssim_weight", "SSIM loss weight", "float", "0.2 in the 3DGS paper."],
  ]],
  ["System", [
    ["device", "Device", "select", "", [["auto", "Automatic"], ["cuda", "NVIDIA GPU (CUDA)"], ["cpu", "CPU"], ["mps", "Apple GPU (MPS)"]]],
    ["backend", "Renderer", "select", "gsplat needs an NVIDIA GPU.", [["auto", "Automatic"], ["gsplat", "gsplat (CUDA)"], ["torch", "PyTorch (any device)"]]],
    ["seed", "Random seed", "int", ""],
    ["checkpoint_every", "Checkpoint every", "int", "0 = only at the end."],
  ]],
];

// Settings of the direct build from raw data. Every key is a ConstructConfig field.
const CONSTRUCT_GROUPS = [
  ["Detail", [
    ["color_threshold", "Color detail threshold", "float", "Split where colors vary more than this. Lower = more, smaller splats."],
    ["normal_threshold", "Curvature threshold", "float", "Split where the surface bends more than this."],
    ["pixel_scale", "Smallest splat (pixels)", "float", "Splits stop at this many pixel footprints."],
    ["edge_split", "Split along color edges", "select", "Two thin splats across sharp borders and text.", [[true, "On"], [false, "Off"]]],
    ["edge_contrast", "Edge contrast", "float", "Color change that counts as an edge."],
  ]],
  ["Check & repeat", [
    ["rounds", "Rounds", "int", "Render, compare and split where the error is high."],
    ["error_threshold", "Error limit", "float", "Cells with a larger mean error split in the next round."],
    ["correction_passes", "Color solve passes", "int", "Per round."],
    ["check_views", "Views checked per round", "int", ""],
  ]],
  ["Shape & color", [
    ["coverage", "Coverage", "float", "Splat size relative to its area. Higher = smoother, blurrier."],
    ["opacity", "Opacity", "float", ""],
    ["sh_degree", "View-dependent color (SH degree)", "select", "", [[0, "0"], [1, "1"], [2, "2"], [3, "3"]]],
    ["sh_regularization", "SH smoothing", "float", "Higher = less view-dependent color."],
    ["roughness_regularization", "Extra smoothing on rough materials", "float", "Uses the roughness pass."],
  ]],
  ["Samples & background", [
    ["max_samples", "Maximum surface samples", "int", "Pixels are skipped evenly above this."],
    ["background", "Sky / world shell", "select", "", [[true, "On"], [false, "Off"]]],
    ["background_resolution", "Shell resolution", "int", "Cells per cube face."],
    ["device", "Device", "select", "", [["auto", "Automatic"], ["cuda", "NVIDIA GPU (CUDA)"], ["cpu", "CPU"], ["mps", "Apple GPU (MPS)"]]],
  ]],
];

function fieldGroups(groups, values, inputs, onChange) {
  return groups.map(([title, fields], gi) => h("details", { class: "group", open: gi === 0 ? true : undefined },
    h("summary", {}, title),
    h("div", { class: "fields" }, fields.map(([key, label, type, hint, options]) => {
      let input;
      if (type === "select") {
        input = h("select", { class: "input" }, options.map(([v, l]) => h("option", { value: v }, l)));
      } else if (type === "path") {
        input = h("input", { class: "input", placeholder: "choose…" });
        input.addEventListener("click", async () => {
          const path = await pickPath({ title: "Choose a .ply", mode: "file", api });
          if (path) { input.value = path; values[key] = path; }
        });
      } else {
        input = h("input", { class: "input", type: "number", step: type === "float" ? "any" : "1" });
      }
      input.value = values[key];
      input.addEventListener("change", () => {
        const raw = input.value;
        const kind = options ? typeof options[0][0] : null;
        values[key] = type === "int" ? parseInt(raw || "0") : type === "float" ? parseFloat(raw || "0")
          : kind === "number" ? parseInt(raw) : kind === "boolean" ? raw === "true" : raw;
        onChange();
      });
      inputs[key] = input;
      return h("div", { class: "field", style: type === "path" ? { gridColumn: "1 / -1" } : {} },
        h("label", {}, label), input, hint ? h("div", { class: "hint" }, hint) : null);
    }))));
}

export async function render(root, params) {
  const runId = params[0];
  if (runId) return monitor(root, runId);
  if (state.activeRun) return monitor(root, state.activeRun.id);
  return setup(root);
}

// -- setup ----------------------------------------------------------------------

async function setup(root) {
  const page = h("div", { class: "page" });
  root.append(page);
  if (!state.datasetPath) {
    page.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Train"), h("p", {}, "Choose a dataset to train."))),
      h("div", { class: "card empty" }, h("div", { html: icons.folder }), h("p", {}, "No dataset open."),
        iconButton("folder", "Open dataset", chooseDataset, "btn primary")));
    return;
  }
  const [data, presets, system] = await Promise.all([
    api.dataset(state.datasetPath).catch(e => ({ error: e.message })), api.presets(), api.system()]);
  if (data.error) {
    page.append(h("div", { class: "alert err" }, data.error), h("div", { class: "spacer" }),
      iconButton("folder", "Open another dataset", chooseDataset, "btn primary"));
    return;
  }
  let presetName = localStorage.getItem("splatgen.preset") || "standard";
  let values = { ...presets.presets[presetName].values };
  const inputs = {};
  const nameInput = h("input", { class: "input", placeholder: "Run name (optional)" });

  const presetButtons = Object.entries(presets.presets).map(([key, p]) =>
    h("button", { class: `preset ${key === presetName ? "on" : ""}`, onclick: () => choosePreset(key) },
      h("b", {}, p.label), h("span", {}, p.description)));

  function choosePreset(key) {
    presetName = key;
    localStorage.setItem("splatgen.preset", key);
    values = { ...presets.presets[key].values };
    presetButtons.forEach((b, i) => b.classList.toggle("on", Object.keys(presets.presets)[i] === key));
    for (const [k, el] of Object.entries(inputs)) el.value = values[k];
    updateEstimate();
  }

  const groups = fieldGroups(GROUPS, values, inputs, updateEstimate);
  const construct = { ...presets.construct };
  const constructGroups = fieldGroups(CONSTRUCT_GROUPS, construct, {}, () => {});

  // Method: standard training, or the direct build when the raw data is there.
  const hasRaw = !!data.extras.raw_dataset;
  let method = hasRaw ? (localStorage.getItem("splatgen.method") || "construct_train") : "train";
  if (!presets.methods[method]) method = "train";
  const polishSteps = h("input", { class: "input", type: "number", step: "1", value: localStorage.getItem("splatgen.polish") || "2000" });
  const methodButtons = Object.entries(presets.methods).map(([key, m]) =>
    h("button", { class: `preset ${key === method ? "on" : ""}`, disabled: key !== "train" && !hasRaw ? true : undefined,
      onclick: () => chooseMethod(key) }, h("b", {}, m.label), h("span", {}, m.description)));
  const trainPanel = h("div", {}, h("h3", {}, "Preset"), h("div", { class: "presets" }, presetButtons),
    h("div", { class: "spacer" }), ...groups);
  const buildPanel = h("div", {},
    h("div", { class: "field polish-steps" }, h("label", {}, "Polish training steps"), polishSteps,
      h("div", { class: "hint" }, "A short training run that starts from the built splats.")),
    h("h3", {}, "Build settings"), ...constructGroups);
  function chooseMethod(key) {
    method = key;
    if (hasRaw) localStorage.setItem("splatgen.method", key);
    methodButtons.forEach((b, i) => b.classList.toggle("on", Object.keys(presets.methods)[i] === key));
    trainPanel.hidden = key !== "train";
    buildPanel.hidden = key === "train";
    buildPanel.querySelector(".polish-steps").hidden = key !== "construct_train";
    startButton.innerHTML = `${icons.play}<span>${key === "train" ? "Start training" : key === "construct" ? "Build splats" : "Build & polish"}</span>`;
    updateEstimate();
  }
  polishSteps.addEventListener("change", () => localStorage.setItem("splatgen.polish", polishSteps.value));

  const estimate = h("div", { class: "muted small" });
  function updateEstimate() {
    const [w, hgt] = (data.resolution.split(" ")[0] || "0x0").split("x").map(Number);
    const ds = values.downscale || 1;
    const px = w && hgt ? `${Math.round(w / ds)}×${Math.round(hgt / ds)} px` : "";
    const train = values.test_every > 1 ? data.camera_count - Math.ceil(data.camera_count / values.test_every) : data.camera_count;
    const what = method === "train" ? `${fmt.int(values.steps)} steps`
      : method === "construct" ? "no training" : `build + ${fmt.int(parseInt(polishSteps.value) || 0)} polish steps`;
    estimate.textContent = `${what} · ${train} images ${px ? "at " + px : ""} · ` +
      `renderer ${values.backend === "auto" ? system.default_backend : values.backend}`;
  }

  const startButton = iconButton("play", "Start training", start, "btn primary big");
  async function start() {
    startButton.disabled = true;
    try {
      const config = { ...values };
      if (method === "construct_train") config.steps = parseInt(polishSteps.value) || 2000;
      const run = await api.startRun({ dataset: state.datasetPath, preset: presetName, config, method, construct,
        name: nameInput.value.trim() || undefined });
      toast(method === "train" ? `Training started: ${run.name}` : `Building: ${run.name}`, "ok");
      navigate(`#/train/${run.id}`);
    } catch (error) {
      toast(error.message, "err", 6000);
      startButton.disabled = false;
    }
  }

  const slow = system.default_backend !== "gsplat";
  page.append(
    h("div", { class: "page-head" },
      h("div", {}, h("h1", {}, "Train"), h("p", {}, "Choose a preset, adjust anything you like, and start.")),
      h("div", { class: "actions" }, iconButton("refresh", "Runs", () => navigate("#/runs"), "btn ghost"))),
    h("div", { class: "grid side" },
      h("div", { class: "stack" },
        h("div", { class: "card" }, h("h3", {}, "Dataset"),
          h("div", { class: "row" }, h("div", { class: "grow ellipsis" },
            h("b", {}, datasetName(data.extras.build_folder || data.root)),
            h("div", { class: "muted small" }, `${data.camera_count} images · ${data.resolution} · ${fmt.int(data.point_count)} points`)),
            iconButton("folder", "", chooseDataset, "btn icon ghost", "Choose another dataset"))),
        slow ? h("div", { class: "alert warn" }, system.cuda
          ? "gsplat is not installed: training will use the slow PyTorch renderer. Install gsplat for GPU speed."
          : "No NVIDIA GPU: training uses the PyTorch renderer. Use the Preview preset and 1/4 resolution for anything but tiny scenes.") : null,
        h("div", { class: "card" }, h("h3", {}, "Name"), nameInput)),
      h("div", { class: "card" },
        h("h3", {}, "Method"),
        h("div", { class: "presets" }, methodButtons),
        hasRaw ? null : h("div", { class: "hint", style: { marginTop: "6px" } },
          "Building directly needs the raw data: export with the Raw dataset option in the Blender add-on."),
        h("div", { class: "spacer" }),
        trainPanel, buildPanel,
        h("div", { class: "spacer" }),
        h("div", { class: "row wrap" }, startButton, estimate))));
  chooseMethod(method);
}

// -- monitor ----------------------------------------------------------------------

async function monitor(root, runId) {
  const page = h("div", { class: "page" });
  root.append(page);
  let run;
  try { run = await api.run(runId); } catch (error) {
    page.append(h("div", { class: "alert err" }, error.message));
    return;
  }
  let data = null;
  try { data = await api.dataset(run.dataset); } catch { /* dataset may have moved */ }
  const cameras = data ? data.cameras : [];
  let camera = cameras.length ? Math.max(0, cameras.findIndex((_, i) => run.config.test_every > 1 && i % run.config.test_every === 0)) : 0;

  const title = h("h1", {}, run.name);
  const status = h("span");
  const subtitle = h("p", { class: "small muted" }, datasetName(run.dataset), " · ", h("span", { class: "mono faint" }, run.dataset));
  const actions = h("div", { class: "actions" });
  const bar = h("i");
  const progressText = h("div", { class: "muted small" });
  const tiles = {};
  const tile = (key, label) => (tiles[key] = { value: h("div", { class: "value" }, "–"), sub: h("div", { class: "sub" }) },
    h("div", { class: "stat" }, h("div", { class: "label" }, label), tiles[key].value, tiles[key].sub));
  const lossCanvas = h("canvas", { class: "chart" });
  const splatCanvas = h("canvas", { class: "chart" });
  const lossChart = new LineChart(lossCanvas, [
    { key: "loss", label: "loss", color: "#b5a3ff" },
    { key: "psnr", label: "train PSNR", color: "#36b3ff", axis: "right", format: v => v.toFixed(1) }]);
  const splatChart = new LineChart(splatCanvas, [
    { key: "gaussians", label: "splats", color: "#3ccf91", format: v => fmt.int(v) }]);
  const previewImg = h("img", { alt: "render" });
  const truthImg = h("img", { alt: "ground truth" });
  const camSelect = h("select", { class: "input", style: { width: "auto" } },
    cameras.map((c, i) => h("option", { value: i }, `${i + 1}. ${c.name}${run.config.test_every > 1 && i % run.config.test_every === 0 ? " (held out)" : ""}`)));
  camSelect.value = camera;
  camSelect.onchange = () => { camera = parseInt(camSelect.value); refreshPreview(true); };
  const autoPreview = h("input", { type: "checkbox", checked: true });
  const log = h("div", { class: "log" });
  const errorBox = h("div");

  page.append(
    h("div", { class: "page-head" }, h("div", {}, h("div", { class: "row" }, title, status), subtitle), actions),
    errorBox,
    h("div", { class: "card" },
      h("div", { class: "row" }, h("div", { class: "grow" }, progressText)),
      h("div", { style: { margin: "8px 0 2px" } }, h("div", { class: "bar" }, bar))),
    h("div", { class: "spacer" }),
    h("div", { class: "stats" },
      tile("step", "Step"), tile("loss", "Loss"), tile("psnr", "Train PSNR"), tile("test", "Test PSNR"),
      tile("gaussians", "Splats"), tile("speed", "Speed"), tile("time", "Elapsed")),
    h("div", { class: "spacer" }),
    h("div", { class: "grid two" },
      h("div", { class: "card" },
        h("div", { class: "row" }, h("h2", { class: "grow", style: { margin: 0 } }, "Live preview"),
          camSelect,
          iconButton("refresh", "", () => refreshPreview(true), "btn icon", "Refresh preview"),
          h("label", { class: "check small" }, autoPreview, "auto")),
        h("div", { class: "spacer" }),
        cameras.length ? h("div", { class: "compare" },
          h("figure", {}, h("div", { class: "preview-frame" }, previewImg), h("figcaption", {}, "Splats")),
          h("figure", {}, h("div", { class: "preview-frame" }, truthImg), h("figcaption", {}, "Ground truth")))
          : h("div", { class: "alert warn" }, "The dataset for this run cannot be found, so previews are unavailable."),
        h("div", { class: "spacer" }),
        h("h3", {}, "Log"), log),
      h("div", { class: "stack" },
        h("div", { class: "card" }, h("h2", {}, "Loss & PSNR"), lossCanvas),
        h("div", { class: "card" }, h("h2", {}, "Splat count"), splatCanvas))),
    h("div", { class: "spacer" }),
    h("details", { class: "card" }, h("summary", { class: "muted", style: { cursor: "pointer" } }, "Configuration"),
      h("div", { class: "spacer" }),
      h("pre", { class: "log", style: { maxHeight: "320px" } }, JSON.stringify(run.config, null, 2))));

  let lastPreview = 0;
  let previewBusy = false;
  function refreshPreview(force = false) {
    if (!cameras.length || previewBusy) return;
    if (!force && (!autoPreview.checked || Date.now() - lastPreview < 3000)) return;
    previewBusy = true;
    lastPreview = Date.now();
    const img = new Image();
    img.onload = () => { previewImg.src = img.src; previewBusy = false; };
    img.onerror = () => { previewBusy = false; };
    img.src = api.previewUrl(runId, camera, 720);
    truthImg.src = api.thumbUrl(run.dataset, camera, 720);
  }

  function setActions(r, live) {
    const st = live ? live.status : r.status;
    const buttons = [];
    if (live && st === "building") buttons.push(iconButton("stop", "Stop", () => api.stop(runId).catch(e => toast(e.message, "err")), "btn danger"));
    if (live && ["training", "loading"].includes(st)) buttons.push(iconButton("pause", "Pause", () => api.pause(runId).catch(e => toast(e.message, "err"))));
    if (live && st === "paused") buttons.push(iconButton("play", "Resume", () => api.resume(runId).catch(e => toast(e.message, "err")), "btn primary"));
    if (live && ["training", "paused", "loading"].includes(st)) buttons.push(iconButton("stop", "Stop & save", () => api.stop(runId).catch(e => toast(e.message, "err")), "btn danger"));
    if (!live && ["stopped", "finished", "failed"].includes(st) && r.file_sizes && r.file_sizes["checkpoint.pt"]) {
      buttons.push(iconButton("play", st === "stopped" ? "Continue" : "Train more", async () => {
        let extra = 0;
        if (st !== "stopped") {
          const answer = await promptDialog("Train more", "Additional steps", "3000");
          if (!answer) return;
          extra = parseInt(answer) || 0;
        }
        try { await api.resume(runId, extra); toast("Training continues", "ok"); } catch (e) { toast(e.message, "err"); }
      }));
    }
    if (!live && st === "finished" && r.method === "construct" && !(r.file_sizes && r.file_sizes["checkpoint.pt"])) {
      buttons.push(iconButton("play", "Polish with training", async () => {
        const answer = await promptDialog("Polish with training", "Training steps", "2000");
        if (!answer) return;
        try { await api.resume(runId, parseInt(answer) || 2000); toast("Polishing started", "ok"); } catch (e) { toast(e.message, "err"); }
      }, "btn primary"));
    }
    buttons.push(iconButton("cube", "Open in 3D viewer", () => navigate(`#/viewer/${runId}`), "btn"));
    if (r.files && r.files.ply) buttons.push(h("a", { class: "btn primary", href: api.fileUrl(runId, r.files.ply) }, h("span", { html: icons.download }), "Download PLY"));
    actions.replaceChildren(...buttons);
  }

  let actionKey = "";
  let metricsAt = 0;
  async function poll() {
    let r;
    try { r = await api.run(runId); } catch { return; }
    const live = r.live;
    const st = live ? live.status : r.status;
    status.replaceChildren(pill(st));
    const rawLatest = (live && live.latest && Object.keys(live.latest).length) ? live.latest : (r.results?.train || {});
    const building = st === "building" || (rawLatest.stage && rawLatest.step === undefined);
    const latest = building ? {} : rawLatest;
    const steps = (live && live.steps) || r.config.steps;
    const step = latest.step || r.step || 0;
    bar.style.width = building ? `${Math.min(100, 100 * (rawLatest.progress || 0))}%`
      : `${Math.min(100, 100 * step / Math.max(1, steps))}%`;
    const where = live && live.device ? ` · ${live.device} / ${live.backend}` : r.device ? ` · ${r.device} / ${r.backend}` : "";
    const stage = { samples: "reading surface samples", detail: "measuring detail", split: "splitting detailed areas",
      shape: "fitting splat shapes", colour: "solving colours", check: "rendering and correcting",
      evaluate: "scoring held-out views", done: "writing files" }[rawLatest.stage] || "starting";
    const round = rawLatest.round ? ` · round ${rawLatest.round} of ${rawLatest.rounds}` : "";
    progressText.textContent = building ? `Building from raw data · ${stage}${round}`
      : st === "built" ? "Built · preparing the polish training…"
      : st === "training"
      ? `Training · step ${fmt.int(step)} of ${fmt.int(steps)} · about ${fmt.duration(latest.eta)} left${where}`
      : st === "loading" ? "Loading images and preparing the starting splats…"
      : st === "saving" ? "Evaluating and writing the exports…"
      : st === "paused" ? `Paused at step ${fmt.int(step)}`
      : r.method === "construct" && !step ? `${st[0].toUpperCase() + st.slice(1)} · built without training${
        r.results?.construct?.time ? ` in ${fmt.duration(r.results.construct.time)}` : ""}`
      : `${st[0].toUpperCase() + st.slice(1)} at step ${fmt.int(step)} of ${fmt.int(steps)}${where}`;
    if (building || (r.method === "construct" && !step)) {
      const rounds = r.results?.construct?.rounds || [];
      const lastRound = rounds[rounds.length - 1] || {};
      tiles.step.value.textContent = building ? (rawLatest.round ? `${rawLatest.round}/${rawLatest.rounds}` : "–") : `${rounds.length}`;
      tiles.step.sub.textContent = "build rounds";
      tiles.loss.value.textContent = "–";
      tiles.psnr.value.textContent = (rawLatest.psnr || lastRound.check_psnr) ? `${fmt.num(rawLatest.psnr || lastRound.check_psnr, 2)} dB` : "–";
      tiles.psnr.sub.textContent = "build views";
      tiles.gaussians.value.textContent = fmt.int(rawLatest.splats || r.results?.gaussians);
      tiles.gaussians.sub.textContent = "";
      tiles.speed.value.textContent = "–";
      tiles.speed.sub.textContent = "no training";
      tiles.time.value.textContent = fmt.duration(r.results?.construct?.time);
      tiles.time.sub.textContent = "";
      const test = r.results?.eval;
      tiles.test.value.textContent = test && test.psnr ? `${fmt.num(test.psnr, 2)} dB` : "–";
      tiles.test.sub.textContent = test && test.ssim ? `SSIM ${fmt.num(test.ssim, 3)}` : "measured at the end";
    } else {
      tiles.step.value.textContent = fmt.int(step);
      tiles.step.sub.textContent = `of ${fmt.int(steps)}`;
      tiles.loss.value.textContent = fmt.num(latest.loss, 4);
      tiles.psnr.value.textContent = latest.psnr ? `${fmt.num(latest.psnr, 2)} dB` : "–";
      const test = (live && live.eval && live.eval.psnr) ? live.eval : r.results?.eval;
      tiles.test.value.textContent = test && test.psnr ? `${fmt.num(test.psnr, 2)} dB` : "–";
      tiles.test.sub.textContent = test && test.ssim ? `SSIM ${fmt.num(test.ssim, 3)}` : "measured at the end";
      tiles.gaussians.value.textContent = fmt.int(latest.gaussians || r.results?.gaussians);
      tiles.gaussians.sub.textContent = latest.sh_degree !== undefined ? `SH degree ${latest.sh_degree}` : "";
      tiles.speed.value.textContent = latest.rate ? `${fmt.num(latest.rate, 1)}` : "–";
      tiles.speed.sub.textContent = "steps / second";
      tiles.time.value.textContent = fmt.duration(latest.elapsed);
      tiles.time.sub.textContent = st === "training" ? `ETA ${fmt.duration(latest.eta)}` : "";
    }
    errorBox.replaceChildren(...(r.error ? [h("div", { class: "alert err", style: { marginBottom: "12px" } }, r.error)] : []));
    if (live && live.log) log.textContent = live.log.map(l => l.text).join("\n");
    else if (!live && !log.textContent) log.textContent = r.error ? `Failed: ${r.error}` : `Run ${st}.`;
    const key = `${st}|${JSON.stringify(r.files)}|${Object.keys(r.file_sizes || {}).join()}`;
    if (key !== actionKey) { actionKey = key; setActions(r, live); }
    if (Date.now() - metricsAt > 3000 || !live) {
      metricsAt = Date.now();
      const { metrics } = await api.metrics(runId);
      const trained = metrics.filter(m => !m.build);
      lossChart.set(trained);
      splatChart.set(trained);
    }
    const hasModel = ["training", "paused", "saving", "finished", "stopped", "imported", "built"].includes(st)
      || (building && rawLatest.stage === "check");
    if (hasModel && !previewShown) { previewShown = true; refreshPreview(true); }
    if (live && ["training", "paused", "building"].includes(st)) refreshPreview();
  }

  let previewShown = false;
  await poll();
  const timer = setInterval(poll, 1000);
  const onResize = () => { lossChart.draw(); splatChart.draw(); };
  window.addEventListener("resize", onResize);
  return () => { clearInterval(timer); window.removeEventListener("resize", onResize); };
}
