# Scaling GPU Social-Graph Forces in WebGL2 — Problem & Options

_Last updated: 30 Aug 2025_

## Context
We’re visualizing millions of “accounts” as particles in 3D and applying forces that combine:
- **Global gravity** (e.g., toward a central mass),
- **Graph-driven pairwise forces** (attraction/repulsion along social ties),
- Optional **conversation-/topic-driven** ephemeral ties.

Two hard constraints emerge at scale:
1) **Memory for neighbor lists** (external or precomputed edges).  
2) **Bandwidth / texture-read bottleneck** when evaluating neighbors each frame.

Below is a compact problem statement with practical options, caveats, and an implementation plan suitable for WebGL2 (with notes on WebGPU).

---

## Quick sizing (why it explodes)
Let \(N\) = nodes, average degree \(k\) (directed). Each edge stores **(neighbor index, weight)**.

**Memory ≈ \(N · k · B\)**, where **B** is bytes per edge.

Typical numbers (k ≈ 56 from demo ranges):

| N (nodes) | Edges (N·k) | Float pairs (8 B/edge) | Quantized (4 B/edge) |
|---:|---:|---:|---:|
| 100,000 | 5.6 M | **44.8 MB** | **22.4 MB** |
| 1,000,000 | 56 M | **448 MB** | **224 MB** |
| 2,000,000 | 112 M | **896 MB** | **448 MB** |
| 10,000,000 | 560 M | **4.48 GB** | **2.24 GB** |

Per-frame read cost (position fetches) ≈ **N·k** samples. With RGBA16F positions (8 B/sample):
- 1 M × 56 ≈ **56 M reads** → ~**0.45 GB/frame** (before any other textures); at 60 FPS this implies ~**27 GB/s** just for these reads.
- 2 M × 56 ≈ **112 M reads** → ~**0.9 GB/frame**; at 60 FPS → ~**54 GB/s**.  
On many GPUs this saturates your fragment shader before arithmetic matters.

**Takeaway:** both **memory** and **memory bandwidth** are first-order constraints.

---

## CRUX #1 — Neighbor-list memory
**Goal:** store many edges compactly; accept and stream updates when data comes from external sources.

### Options
1) **Quantize indices & weights (compact path).**  
   - Pack index into 24 bits (RGB) and weight into 8 bits (A) → **RGBA8** (4 B/edge).  
   - Max distinct nodes ≈ 16.7 M (fits 24-bit).  
   - Pros: 2× smaller vs float; widely supported; fast NEAREST fetch.  
   - Cons: weight precision is 8-bit; index decode needed in-shader.

2) **CSR textures (Compressed Sparse Row).**  
   - `rowPtr` (N+1 entries) + `neighbors` (E entries).  
   - Exact fit, no per-row padding.  
   - Layout both as 2D textures with NEAREST; map linear offsets → (x,y) via division/mod.  
   - Pros: suits external data; supports wide degree variation.  
   - Cons: random-access addressing logic in-shader.

3) **Tile + 16-bit offsets.**  
   - Partition nodes into tiles of 65,536; store a 16-bit local offset + a small tile id.  
   - Keeps 4 B/edge; good when neighbors are mostly local after layout.

4) **Fixed-K arrays (simple but wasteful).**  
   - Preallocate K slots per node (texture array).  
   - Pros: trivial indexing.  
   - Cons: pays for max K even for low-degree nodes; can balloon to multi-GB.

5) **Undirected edge sharing.**  
   - If relationships are symmetric, halve storage by keeping one edge and applying force to both endpoints.

6) **Streaming updates.**  
   - Append new edges via `texSubImage2D` into a ring buffer + small per-frame delta list; rebuild CSR periodically (e.g., every 1–5 s) rather than per event.

**Notes / gotchas**
- Always use **NEAREST** filtering for index/text data.  
- Respect **maxTextureSize** (~8–16k); choose texture **width** ≤ max; compute height accordingly.  
- Consider **endian & packing** if you later port to WebGPU (integers become native and easier there).

---

## CRUX #2 — Read bandwidth per frame
**Goal:** reduce the number of neighbor position reads while keeping visual behavior faithful.

### Options
1) **Stochastic edge sampling (frame-dithering).**  
   - Visit a fraction **p** of edges each frame; scale force by **1/p** (unbiased).  
   - p = 0.25 → 4× fewer reads; motion still smooth as the integrator runs every frame.  
   - Caveat: increases force variance; use mild damping.

2) **Degree caps with weighted reservoir sampling.**  
   - Cap effective degree per node to **K** (e.g., 64–128), sampled by weight.  
   - Preserves high-weight neighbors; keeps reads bounded.  
   - Caveat: truncates the long tail unless compensated via far-field approximations.

3) **Update scheduling by degree.**  
   - Update hubs every frame; long-tail nodes every 2–4 frames; reuse cached acceleration for skipped frames.  
   - Caveat: slight temporal lag for low-degree nodes.

4) **Community/cluster forces (coarsening).**  
   - Precompute communities; each node reads a small set of **community centroids** + a small K of explicit neighbors.  
   - Captures global structure with O(K + C) reads; C ≪ 1,000.  
   - Caveat: need periodic community refresh if the graph evolves.

5) **Hierarchical approximation (Barnes–Hut-like for graphs).**  
   - Build a cluster tree; use cluster aggregates for far regions, expand near ones.  
   - Caveat: tree build/update cost; careful thresholding to avoid artifacts.

6) **Cache neighbor aggregates.**  
   - Per node, maintain a **cached force** texture updated every **R** frames; each frame use cached_acc + a small sampled delta.  
   - Caveat: consistency across frames; choose R based on motion speed.

7) **Edge-parallel accumulation with additive blending.**  
   - Draw one primitive per edge into an accumulation FBO; use blending to scatter contributions to endpoints.  
   - Pro: lets you amortize edges across frames naturally.  
   - Caveat: needs `EXT_color_buffer_float`; `EXT_float_blend` improves RGBA32F blending; use **RGBA16F** if 32F blend is unavailable.

8) **Tiling / spatial hashing (your “tiles” idea).**  
   - Compute a **tileId** per particle from position (GPU pass).  
   - In force pass: only process neighbors whose tile is within radius **r** (e.g., 3×3 tiles), **skip** others or approximate them via tile centroids.  
   - Anneal **r**: start large, shrink as clusters stabilize.  
   - Optional: maintain **tile centroids & counts** (GPU blend pass or CPU every R frames) for a cheap far-field.

9) **Full tile lists (max perf).**  
   - Sort particle indices by tile (GPU/CPU), build `tileRowPtr`, loop only tile±r spans.  
   - Caveat: needs sorting infrastructure; rebuild frequency trade-off.

---

## Conversation-driven forces (ephemeral instead of stateful)
1) **Time-decayed interaction weights.**  
   \( w_{ij}(t) = \sum_{e \in msgs_{ij}} e^{-(t - t_e)/\tau} \). Keep only edges above a threshold in the current window.  
   - Pros: edge set remains small and current; naturally streams.  
   - Caveat: choose \(\tau\) (minutes/hours/days) to match dynamics.

2) **Topic/embedding anchors.**  
   - Compute text embeddings; derive **M** topic anchors (e.g., k-means).  
   - Per node, store small weights to anchors; forces pull to anchors + small K explicit ties.  
   - Pros: O(M + K) reads; semantic layout without all-pairs.  
   - Caveat: anchor maintenance; embedding refresh cadence.

3) **Event-sourced “springs”.**  
   - Short-lived springs for recent replies/mentions, updated via a tiny events texture; decay over seconds/minutes.  
   - Pros: high responsiveness; negligible memory.

---

## External data ingestion (exact ties)
- **CSR textures** with two modes:
  - **Accurate**: `RG32F` (8 B/edge), quick to prototype and validate.
  - **Compact**: `RGBA8` (4 B/edge) with 24-bit indices + 8-bit weight.  
- **Streaming**: ring-buffer deltas + periodic CSR rebuilds.  
- **Sampling-ready**: precompute alias tables or cumulative bins per node offline when you need weighted random neighbor picks (constant-time in-shader sampling).

---

## WebGL2 vs WebGPU
- **WebGL2**: works today; rely on textures + fragment passes, NEAREST fetches, optional float blending extensions.  
- **WebGPU**: storage buffers, 16/32-bit integer indices, atomics, compute workgroups → cleaner CSR, cheaper scatter/gather, much better for multi-million-node graphs with large degrees.

---

## Practical plan for the current demo
1) **Add CSR neighbor support** (both RG32F and compact RGBA8).
2) **Edge sampling knob** (e.g., 25–100%).
3) **Community centroids pass** (100–500 centroids) + per-node **K=64** explicit neighbors.
4) **TileId pass** + **gated neighbor loops** (only tile±r); optional **tile centroids** as far-field.
5) Optional **edge-parallel accumulation** path when extensions allow; otherwise stick to node-parallel loops.
6) UI controls: sampling %, K cap, tile size & radius, far-tiles sample count, anneal toggle, reseed.

---

## Caveats & engineering notes
- **Precision**: Favor **RGBA16F** for position/velocity; use damping; avoid accumulating tiny deltas for hours (reseed or rebase occasionally).  
- **Determinism**: GPU blends and parallel order vary; if reproducibility matters, fix seeds, avoid race-prone scatter unless acceptable.  
- **Extensions**: Check `EXT_color_buffer_float`, `EXT_float_blend`; fall back to RGBA16F targets.  
- **Caps**: Mind `MAX_TEXTURE_SIZE` and `MAX_DRAW_BUFFERS`; clamp sim texture dims; consider half-floats for state to save VRAM.  
- **Mobile**: Tread carefully—memory & bandwidth are much tighter; default to smaller K and stronger sampling.

---

## TL;DR
- The bottlenecks are **(1) edge storage** and **(2) per-frame neighbor reads**.  
- Use **compact CSR** + **quantization** to store edges.  
- Cut reads with **sampling**, **degree caps**, **community & hierarchical approximations**, and your **tile gating** (with optional far-field via centroids).  
- For dynamic conversation forces, prefer **time-decayed edges** + **topic anchors** so the working set stays small and meaningful.  
- If exact, massive graphs are mandatory, **WebGPU** is the pragmatic long-term path.
