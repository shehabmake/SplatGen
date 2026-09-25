import { api } from "../api.js";
import { chooseDataset, navigate, state } from "../app.js";
import { SplatViewer } from "../viewer/splat-viewer.js";
import { datasetName, fmt, h, icons, iconButton, modal, toast } from "../ui.js";

export async function render(root) {
  if (!state.datasetPath) {
    root.append(h("div", { class: "page" },
      h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Dataset"), h("p", {}, "No dataset open."))),
      h("div", { class: "card empty" },
        h("div", { html: icons.folder }),
        h("p", {}, "Open a SplatGen build folder or a COLMAP dataset to inspect it."),
        iconButton("folder", "Open dataset", chooseDataset, "btn primary"))));
    return;
  }
  const page = h("div", { class: "page" }, h("div", { class: "empty" }, h("span", { class: "spin" }), " Loading dataset…"));
  root.append(page);
  let data;
  try {
    data = await api.dataset(state.datasetPath);
  } catch (error) {
    page.replaceChildren(h("div", { class: "alert err" }, error.message), h("div", { class: "spacer" }),
      iconButton("folder", "Open another dataset", chooseDataset, "btn primary"));
    return;
  }

  const name = datasetName(data.extras.build_folder || data.root);
  const viewerBox = h("div", { class: "viewer-wrap", style: { height: "520px" } });
  const thumbs = h("div", { class: "thumbs" });
  const stat = (label, value, sub) => h("div", { class: "stat" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), sub ? h("div", { class: "sub" }, sub) : null);

  page.replaceChildren(...[
    h("div", { class: "page-head" },
      h("div", {}, h("h1", {}, name), h("p", { class: "mono small" }, data.path)),
      h("div", { class: "actions" },
        iconButton("folder", "Change", chooseDataset),
        iconButton("play", "Train this dataset", () => navigate("#/train"), "btn primary"))),
    h("div", { class: "stats" },
      stat("Images", fmt.int(data.camera_count), data.masks ? `${data.masks} with masks` : "no masks"),
      stat("Resolution", data.resolution),
      stat("Sparse points", fmt.int(data.point_count), "used to place the first splats"),
      stat("Scene size", fmt.num(data.extent, 2), "camera rig radius"),
      stat("Format", data.format === "splatgen-legacy" ? "SplatGen" : "COLMAP",
        data.extras.raw_dataset ? "raw data available" : "legacy dataset")),
    h("div", { class: "spacer" }),
    ...data.warnings.map(w => h("div", { class: "alert warn", style: { marginBottom: "8px" } }, w)),
    data.extras.raw_dataset ? h("div", { class: "alert info", style: { marginBottom: "8px" } },
      "This build also has Dataset(Raw). This version trains from the legacy dataset; raw-data training comes in a later version.") : null,
    h("div", { class: "grid two" },
      h("div", { class: "card" }, h("h2", {}, "Cameras & sparse points"),
        h("p", { class: "muted small", style: { marginTop: "-6px" } }, "Drag to orbit, right-drag to pan, scroll to zoom. Click an image to look through its camera."),
        viewerBox),
      h("div", { class: "card" }, h("h2", {}, `Images (${data.camera_count})`),
        h("div", { style: { maxHeight: "560px", overflow: "auto" } }, thumbs)))].filter(Boolean));

  let viewer = null;
  try {
    viewer = new SplatViewer(viewerBox, { up: data.up_axis[2] > 0.9 ? "+z" : data.up_axis[1] > 0.9 ? "+y" : data.up_axis[1] < -0.9 ? "-y" : "+z" });
    viewer.pointSize = 2;
    api.datasetPoints(state.datasetPath).then(points => {
      viewer.setPoints(points);
      viewer.setCameras(data.cameras);
    });
  } catch (error) {
    viewerBox.replaceChildren(h("div", { class: "alert err" }, error.message));
  }

  let selected = -1;
  data.cameras.forEach((cam, i) => {
    const el = h("div", { class: "thumb", title: cam.name, onclick: () => {
      selected = i;
      thumbs.querySelectorAll(".thumb").forEach((t, j) => t.classList.toggle("on", j === i));
      if (viewer) { viewer.setCameras(data.cameras, i); viewer.viewFromCamera(cam); }
    }, ondblclick: () => {
      modal({ title: cam.name, wide: true, body: h("img", { src: api.thumbUrl(state.datasetPath, i, 1600), style: { width: "100%", borderRadius: "8px" } }) });
    } },
      h("img", { loading: "lazy", src: api.thumbUrl(state.datasetPath, i, 256), alt: cam.name }),
      h("span", {}, cam.name));
    thumbs.append(el);
  });
  if (!data.camera_count) toast("This dataset has no usable images", "err");
  return () => { if (viewer) viewer.destroy(); };
}
