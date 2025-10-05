# Plan M — GPU Barnes-Hut Octree (The Best)

This repository hosts Plan M: a GPU-resident Barnes-Hut octree implementation for interactive N-body simulation and visualization.

Overview
 - Plan M implements a 3D octree (64^3 L0 grid) stored in 2D textures using Z-slice stacking.
 - Forces are computed in GLSL using a 3D Barnes-Hut traversal with Plummer softening.
 - The app is designed as a browser-based demo built with Three.js and raw WebGL2 shaders.

Quick start (development)

```powershell
npm install
npm run build
npm start
```

Open http://localhost:8333 in your browser. The landing page runs Plan M by default.

Controls
 - + / = : increase Plan M timestep (faster)
 - - / _ : decrease Plan M timestep (slower)
 - 0 : reset Plan M timestep

Project Layout (important)
 - `src/plan-m/` — Plan M implementation (shaders, pipeline, integrator)
 - `src/index.js` — application entry point (now runs Plan M only)
 - `public/` — static landing page and built JS

Legacy plans
 - Older experimental plans (Plan A, C, D) have been archived under `legacy/` and are no longer part of the primary UI. They remain available for reference and historical comparison.

Future
 - The codebase can be converted to an npm package or library in the future; current focus is to keep a production-ready demo app.

License: MIT