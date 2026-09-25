## SplatGen Prepare — Blender add-on

**Download `SplatGen_Prepare_<version>.zip` below (under Assets). Do not unzip it.**

### Install (Blender 5.3)
1. Disable any older SplatGen Prepare or SplatGen Pro add-on.
2. Edit → Preferences → Add-ons → ⌄ menu (top right) → **Install from Disk…** → pick the zip.
3. Enable **SplatGen Prepare**. In the 3D View press **N** → **SplatGen** tab.
4. Save your .blend, add cameras, press **Build dataset**.

### What it writes
Each build folder contains the unchanged legacy `Dataset(Default)` and the new
`Dataset(Raw)`: render passes (combined, lighting split, clay), geometry,
per-pixel material parameters, object/material ids, motion and denoising
passes, cameras, a scene mesh, collision voxels, and world/probe panoramas.
Files are named `frame_NNNN_<pass>.exr`. Format: `docs/RAW_DATASET.md`
(schema `splatgen-raw-dataset-v2`).

Raw export settings: gear icon next to **Build dataset** → *Raw data*.

### Status
Tested headlessly on Blender 5.0.1 with Cycles. Not yet tested inside the
Blender 5.3 interface or with EEVEE — please report any error messages.
