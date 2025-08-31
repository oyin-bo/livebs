Plan A — Texture-Based Brute Force with Stochastic Sampling
=============================================================

This folder contains the Plan A baseline: GPU-resident particle state stored in textures,
with a fragment shader `update.frag` that advances positions and velocities. The update
shader implements a stochastic neighbor sampling strategy to reduce per-frame work.

Files added by the initial scaffold:
- `index.js` — API stub and capability probe (class `PlanA`)
- `gpu/update.frag` — fragment shader header + sampling loop
- `gpu/render.vert` — vertex shader that renders particles by sampling the positions texture
- `bench.js` — minimal bench harness to collect frame-time statistics

Quick start (developer):
1. Ensure dependencies from repository root are installed (see project README).
2. Import `PlanA` into your app and construct with a WebGL2 rendering context:

```js
import PlanA from './src/plan-a/index.js';
const plan = new PlanA(gl, { particleCount: 100000 });
```

3. Upload initial positions using `PlanA.createInitialPositions` and set up textures.
4. Each frame call `plan.step(dt)` then `plan.render(projectionView)`; use `Bench` to record timings.

Design notes and next steps:
- The shader headers in `gpu/` show exact uniform lists and index->UV formulas; copy them into
  your concrete shader compilation/wrapper code and wire uniforms accordingly.
- Add FBO/texture creation in `index.js` where the `...existing code...` comment is placed.
- Implement the packed RGBA8 fallback path for devices without float FBO support.
- Add smoke tests that run a few frames, perform a `gl.readPixels`, and validate expected changes.
