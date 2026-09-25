import { api } from "../api.js";
import { navigate } from "../app.js";
import { parseFile, parseSplat } from "../viewer/loaders.js";
import { SplatViewer } from "../viewer/splat-viewer.js";
import { fmt, h, icons, iconButton, pickPath, toast } from "../ui.js";

export async function render(root, params) {
  const page = h("div", { class: "page full" });
  const wrap = h("div", { class: "viewer-wrap viewer-full" });
  page.append(wrap);
  root.append(page);
  root.style.overflow = "hidden";

  const info = h("div", { class: "viewer-info glass" }, "No splats loaded");
  const title = h("b", { class: "ellipsis", style: { maxWidth: "260px" } }, "Viewer");
  const runSelect = h("select", { class: "input", style: { width: "220px" } });
  const camSelect = h("select", { class: "input hidden", style: { width: "200px" } });
  const fileInput = h("input", { type: "file", accept: ".ply,.splat", class: "hidden" });
  const liveToggle = h("input", { type: "checkbox" });
  const liveLabel = h("label", { class: "check small hidden" }, liveToggle, "live");
  const upButtons = {};
  const upGroup = h("div", { class: "segmented", title: "Up axis" },
    ["+z", "+y", "-y"].map(axis => (upButtons[axis] = h("button", { onclick: () => setUp(axis) }, axis.toUpperCase()))));
  const downloadMenu = h("div", { class: "row hidden" });
  const drop = h("div", { class: "drop-hint" }, h("div", {}, h("b", {}, "Drop a .ply or .splat file"),
    "or pick a training run above"));

  wrap.append(drop,
    h("div", { class: "viewer-overlay" },
      h("div", { class: "glass viewer-toolbar" },
        title,
        runSelect,
        iconButton("upload", "Open file", () => fileInput.click(), "btn sm"),
        iconButton("folder", "Import PLY", importPly, "btn sm ghost"),
        camSelect,
        liveLabel),
      h("div", { class: "grow" }),
      h("div", { class: "glass viewer-toolbar" },
        upGroup,
        iconButton("target", "Fit", () => viewer.fit(), "btn sm ghost"),
        iconButton("image", "Screenshot", screenshot, "btn sm ghost"),
        downloadMenu)),
    h("div", { class: "viewer-help glass" }, "Drag: orbit · Right-drag / Shift: pan · Scroll: zoom · WASD/QE: move · Double-click: fit"),
    h("div", { style: { position: "absolute", right: "12px", bottom: "12px" } }, info),
    fileInput);

  let viewer;
  try {
    viewer = new SplatViewer(wrap, {
      up: localStorage.getItem("splatgen.up") || "+z",
      onInfo: (s) => { if (s.splats) info.textContent = `${fmt.int(s.splats)} splats` + (s.fps >= 1 ? ` · ${Math.round(s.fps)} fps` : ""); },
    });
  } catch (error) {
    wrap.replaceChildren(h("div", { class: "page" }, h("div", { class: "alert err" }, error.message)));
    return;
  }
  function setUp(axis) {
    viewer.setUp(axis);
    localStorage.setItem("splatgen.up", axis);
    Object.entries(upButtons).forEach(([k, b]) => b.classList.toggle("on", k === axis));
    viewer.fit();
  }
  Object.entries(upButtons).forEach(([k, b]) => b.classList.toggle("on", k === (localStorage.getItem("splatgen.up") || "+z")));

  let currentRun = null;
  let cameras = [];
  let liveTimer = null;

  async function loadRun(id, keepView = false) {
    if (!id) return;
    try {
      const run = await api.run(id);
      currentRun = run;
      info.textContent = "Loading splats…";
      const buffer = await api.splatBuffer(id);
      viewer.setSplats(parseSplat(buffer), { keepView });
      drop.classList.add("hidden");
      title.textContent = run.name;
      if (!keepView) await loadCameras(run);
      const live = run.live && ["training", "paused", "loading"].includes(run.live.status);
      liveLabel.classList.toggle("hidden", !live);
      downloadMenu.replaceChildren(
        ...Object.entries(run.files || {}).map(([fmtName, file]) =>
          h("a", { class: "btn sm", href: api.fileUrl(id, file), title: `Download ${file}` }, h("span", { html: icons.download }), fmtName.toUpperCase())));
      downloadMenu.classList.toggle("hidden", !Object.keys(run.files || {}).length);
      if (!keepView) history.replaceState(null, "", `#/viewer/${id}`);
    } catch (error) {
      toast(error.message, "err");
      info.textContent = "No splats loaded";
    }
  }

  async function loadCameras(run) {
    cameras = [];
    camSelect.classList.add("hidden");
    if (!run.dataset) { viewer.setCameras(null); return; }
    try {
      const data = await api.dataset(run.dataset);
      cameras = data.cameras;
      const axis = data.up_axis[2] > 0.9 ? "+z" : data.up_axis[1] > 0.9 ? "+y" : data.up_axis[1] < -0.9 ? "-y" : null;
      if (axis) { setUp(axis); }
      viewer.showCameras = false;
      viewer.setCameras(cameras);
      camSelect.replaceChildren(h("option", { value: "" }, "Free camera"),
        ...cameras.map((c, i) => h("option", { value: i }, `${i + 1}. ${c.name}`)));
      camSelect.classList.remove("hidden");
    } catch { /* dataset may be gone; viewing still works */ }
  }

  camSelect.onchange = () => {
    const i = camSelect.value;
    if (i === "") { viewer.fixedView = null; viewer.dirty = true; return; }
    viewer.viewFromCamera(cameras[parseInt(i)]);
  };

  liveToggle.onchange = () => {
    clearInterval(liveTimer);
    if (liveToggle.checked && currentRun) liveTimer = setInterval(() => loadRun(currentRun.id, true), 8000);
  };

  async function loadLocal(file) {
    try {
      info.textContent = `Reading ${file.name}…`;
      const data = parseFile(file.name, await file.arrayBuffer());
      viewer.setSplats(data);
      viewer.setCameras(null);
      currentRun = null;
      title.textContent = file.name;
      runSelect.value = "";
      camSelect.classList.add("hidden");
      downloadMenu.classList.add("hidden");
      liveLabel.classList.add("hidden");
      drop.classList.add("hidden");
      if (data.kind === "points") toast("This PLY is a plain point cloud; points are shown as small splats.");
    } catch (error) {
      toast(error.message, "err", 6000);
    }
  }

  fileInput.onchange = () => { if (fileInput.files[0]) loadLocal(fileInput.files[0]); fileInput.value = ""; };
  const dragOn = (e) => { e.preventDefault(); wrap.classList.add("dragging"); drop.classList.remove("hidden"); };
  const dragOff = () => { wrap.classList.remove("dragging"); if (viewer.splats) drop.classList.add("hidden"); };
  wrap.addEventListener("dragover", dragOn);
  wrap.addEventListener("dragleave", dragOff);
  wrap.addEventListener("drop", (e) => {
    e.preventDefault(); dragOff();
    const file = e.dataTransfer.files[0];
    if (file) loadLocal(file);
  });

  async function importPly() {
    const path = await pickPath({ title: "Import a Gaussian splat .ply", mode: "file", api,
      hint: "The file is copied into your runs folder so it can be viewed, exported or trained further." });
    if (!path) return;
    try {
      const run = await api.importPly(path);
      toast(`Imported ${run.name}`, "ok");
      await fillRuns(run.id);
      loadRun(run.id);
    } catch (error) { toast(error.message, "err", 6000); }
  }

  function screenshot() {
    const a = h("a", { href: viewer.snapshot(), download: `${title.textContent || "splats"}.png` });
    a.click();
  }

  async function fillRuns(selected) {
    const { runs } = await api.runs();
    const usable = runs.filter(r => (r.files && r.files.splat) || r.live);
    runSelect.replaceChildren(h("option", { value: "" }, usable.length ? "Choose a run…" : "No runs yet"),
      ...usable.map(r => h("option", { value: r.id }, `${r.name} · ${r.live ? r.live.status : r.status}`)));
    if (selected) runSelect.value = selected;
  }
  runSelect.onchange = () => loadRun(runSelect.value);

  await fillRuns(params[0]);
  if (params[0]) loadRun(params[0]);

  return () => {
    clearInterval(liveTimer);
    viewer.destroy();
    root.style.overflow = "";
  };
}
