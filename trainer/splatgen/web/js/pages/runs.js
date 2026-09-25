import { api } from "../api.js";
import { navigate } from "../app.js";
import { confirmDialog, datasetName, fmt, h, icons, iconButton, pickPath, pill, promptDialog, toast } from "../ui.js";

export async function render(root) {
  const body = h("div", {}, h("div", { class: "empty" }, h("span", { class: "spin" })));
  root.append(h("div", { class: "page" },
    h("div", { class: "page-head" },
      h("div", {}, h("h1", {}, "Runs"), h("p", {}, "Every training run and imported splat, with its exports.")),
      h("div", { class: "actions" },
        iconButton("upload", "Import PLY", importPly),
        iconButton("plus", "New training", () => navigate("#/train"), "btn primary"))),
    h("div", { class: "card", style: { padding: "0", overflow: "auto" } }, body)));

  async function importPly() {
    const path = await pickPath({ title: "Import a Gaussian splat .ply", mode: "file", api });
    if (!path) return;
    try { const run = await api.importPly(path); toast(`Imported ${run.name}`, "ok"); load(); }
    catch (error) { toast(error.message, "err", 6000); }
  }

  async function exportTo(run, format) {
    const folder = await pickPath({ title: `Save ${format.toUpperCase()} to…`, api });
    if (!folder) return;
    try { const out = await api.exportRun(run.id, format, folder); toast(`Saved ${out.path}`, "ok", 6000); }
    catch (error) { toast(error.message, "err", 6000); }
  }

  async function load() {
    const { runs } = await api.runs();
    if (!runs.length) {
      body.replaceChildren(h("div", { class: "empty" }, h("div", { html: icons.cube }),
        h("p", {}, "No runs yet. Train a dataset or import a PLY.")));
      return;
    }
    const rows = runs.map(run => {
      const live = run.live;
      const status = live ? live.status : run.status;
      const steps = live?.latest?.step ?? run.step;
      const test = run.results?.eval?.psnr;
      const files = run.files || {};
      const actions = h("div", { class: "row", style: { justifyContent: "flex-end", gap: "4px" } },
        run.kind !== "imported" ? iconButton("spark", "", () => navigate(`#/train/${run.id}`), "btn sm ghost", "Training details") : null,
        (files.splat || live) ? iconButton("cube", "", () => navigate(`#/viewer/${run.id}`), "btn sm ghost", "View in 3D") : null,
        files.ply ? h("a", { class: "btn sm", href: api.fileUrl(run.id, files.ply), title: "Download PLY" }, "PLY") : null,
        files.splat ? h("a", { class: "btn sm", href: api.fileUrl(run.id, files.splat), title: "Download .splat" }, "SPLAT") : null,
        files.ply ? iconButton("folder", "", () => exportTo(run, "ply"), "btn sm ghost", "Save PLY to a folder…") : null,
        iconButton("edit", "", async () => {
          const name = await promptDialog("Rename run", "Name", run.name);
          if (name) { await api.rename(run.id, name); load(); }
        }, "btn sm ghost", "Rename"),
        live ? null : iconButton("trash", "", async () => {
          if (await confirmDialog("Delete run", `Delete "${run.name}" and all its files? This cannot be undone.`, "Delete", true)) {
            try { await api.remove(run.id); toast("Run deleted"); load(); } catch (e) { toast(e.message, "err"); }
          }
        }, "btn sm ghost", "Delete"));
      return h("tr", {},
        h("td", {}, h("div", { class: "title" }, h("b", {}, run.name)),
          h("div", { class: "faint small ellipsis", style: { maxWidth: "340px" }, title: run.dataset }, run.kind === "imported" ? "imported PLY" : datasetName(run.dataset))),
        h("td", {}, pill(status)),
        h("td", { class: "num", style: { whiteSpace: "nowrap" } }, run.kind === "imported" ? "–" : `${fmt.int(steps)} / ${fmt.int(run.config?.steps)}`),
        h("td", { class: "num" }, fmt.int(live?.latest?.gaussians ?? run.results?.gaussians)),
        h("td", { class: "num" }, test ? `${fmt.num(test, 2)} dB` : "–"),
        h("td", { class: "muted", style: { whiteSpace: "nowrap" } }, fmt.date(run.created)),
        h("td", {}, actions));
    });
    body.replaceChildren(h("table", { class: "table" },
      h("thead", {}, h("tr", {}, ["Run", "Status", "Steps", "Splats", "Test PSNR", "Created", ""].map((t, i) =>
        h("th", { class: i >= 2 && i <= 4 ? "num" : "" }, t)))),
      h("tbody", {}, rows)));
  }
  await load();
  const timer = setInterval(load, 3000);
  return () => clearInterval(timer);
}
