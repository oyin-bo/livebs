# Plan A — Texture-Based Brute Force with Stochastic Sampling
Status: Implementation-ready with complete specifications
Summary
-------
Plan A is the safe, baseline implementation: keep particle state in GPU-accessible textures and perform a mostly brute-force force calculation, while reducing average work using stochastic neighbor sampling and temporal subsampling. This plan is designed to be the first playable baseline and a fallback for all other plans.

This document provides complete implementation specifications including exact API contracts, shader interfaces, data layouts, capability detection, fallback strategies, test requirements, and edge case handling.

TL;DR
-----
- Keep particle state in GPU textures and update with a fullscreen fragment pass (ping-pong FBOs).
- Use stochastic neighbor sampling + temporal accumulation to avoid full O(N²) work every frame.
- Prefer RGBA32F when available; fall back to 8-bit packing when not.
- Target: baseline of 100k particles at ~30 FPS on mid-range desktop as the initial success metric; higher targets are aspirational and depend heavily on samplingFraction and hardware.

Goals / Success Criteria
------------------------
- Implement a GPU-driven particle system where positions (and optionally velocities) live in textures.
- Achieve smooth, debuggable baseline behavior at 100k particles on a mid-range desktop GPU.
- Provide a simple, repeatable benchmark harness (FPS, particle count, memory usage).
- Provide clear hooks for migrating to more advanced plans (C/D/M) later.

What "done" looks like
----------------------
- A working demo that: loads particles into a texture, updates positions each frame with a fragment shader pass, renders as points; UI to change particle count and sampling fraction; records FPS and memory usage.
- Tests and small README that explain how to run and measure the demo.

High-level architecture
-----------------------
- Host UI / app (Three.js) — existing `src/index.js` will drive and host the demo.
- Plan-A module (under `src/plan-a/`) that contains:
  - an initialization module that creates textures, framebuffers, and shaders
  - an update pass (fragment shader) that computes new positions/velocities
  - a render pass that reads positions and draws points
  - 8-bit packing/unpacking helpers for devices without float-texture support

Files and locations
-------------------
- `src/plan-a/index.js` — top-level Plan A class (instantiation, start/stop/update hooks)
- `src/plan-a/gpu/` — shaders and GPGPU helper code
  - `src/plan-a/gpu/update.frag` — fragment shader used for the update pass
  - `src/plan-a/gpu/pass.vert` — full-screen quad vertex shader
  - `src/plan-a/gpu/render.vert` — point rendering vertex shader
  - `src/plan-a/gpu/render.frag` — point rendering fragment shader
- `src/plan-a/utils/gpu-texture.js` — helper to create float/encoded textures and FBOs
- `src/plan-a/docs/1-plan.md` — this file (implementation plan)
- `src/plan-a/bench.js` — lightweight measurement harness (FPS, frame timings)
- `src/plan-a/README.md` — quick start and notes

Three.js integration notes
--------------------------
Prefer using `THREE.RawShaderMaterial` (or `THREE.ShaderMaterial`) with a `THREE.Points` or an instanced triangle to render particles. Key points:

- Create a buffer attribute `a_index` sized `particleCount` containing indices [0..P-1] and use `gl.drawArrays` / `THREE.Points` to issue draws.
- Bind the `positions` texture to the material as a `sampler2D` uniform and pass `u_texSize` so the vertex shader can fetch per-particle position by index.
- Use a `THREE.DataTexture` only for initial CPU uploads; prefer letting the GPGPU pass render to an FBO and use the resulting GL texture directly as the sampler bound to the shader material.
- Keep the render pass separate from the update pass: update writes to ping-pong FBOs, then render reads from the current position texture.

Performance caveats (practical targets)
-------------------------------------
The acceptance table lists aspirational targets for high-end hardware. For the initial baseline, treat the Desktop Mid / 500k 60fps goal as aspirational: validate performance empirically and tune `samplingFraction` and `point` rendering quality. Expect that raw brute-force force computation—even stochastic—will be bounded by memory bandwidth and shader arithmetic; measure and iterate from the 100k baseline first.

## Complete API Contract

### PlanA Class Interface
```javascript
class PlanA {
  constructor(gl, options = {})
  // options: {
  //   particleCount: 100000,
  //   samplingFraction: 0.25,
  //   dt: 0.016,
  //   integrationMethod: 'euler' | 'semi-implicit',
  //   wrapMode: 'wrap' | 'clamp',
  //   worldBounds: { min: [-10,-10,-10], max: [10,10,10] },
  //   enableVelocityTexture: true,
  //   seed: 12345
  // }

  async init()                    // Returns Promise, throws on unrecoverable errors
  step()                          // Run one update + render frame
  resize(newParticleCount)        // Safe reallocation and state migration
  setSamplingFraction(fraction)   // Update runtime parameter [0.01, 1.0]
  readback(maxCount = 1000)       // Read first N particles for testing (slow)
  getMetrics()                    // Returns { fps, frameTime, memoryMB, particleCount }
  dispose()                       // Free all GPU resources
}
```

### Shader Interface Specifications

#### Update Fragment Shader (`update.frag`)
```glsl
// Uniforms (must be bound by host)
uniform sampler2D u_positions;     // Current positions (RGBA32F)
uniform sampler2D u_velocities;    // Current velocities (RGBA32F, optional)
uniform float u_time;              // Global time (seconds)
uniform float u_dt;                // Delta time (seconds)
uniform int u_frameCount;          // Frame counter for deterministic RNG
uniform float u_samplingFraction;  // Fraction of neighbors to sample [0,1]
uniform vec2 u_texSize;            // Texture dimensions (width, height)
uniform int u_particleCount;       // Active particle count
uniform vec3 u_worldMin;           // World bounds minimum
uniform vec3 u_worldMax;           // World bounds maximum
uniform uint u_seed;               // RNG seed for reproducibility
uniform int u_integrationMethod;   // 0=euler, 1=semi-implicit
uniform int u_wrapMode;            // 0=wrap, 1=clamp

// Outputs
out vec4 fragColor;                // New position (xyz) + mass (w)
```

#### Render Vertex Shader (`render.vert`)
```glsl
// Attributes
attribute float a_index;           // Particle index

// Uniforms
uniform sampler2D u_positions;     // Position texture
uniform vec2 u_texSize;            // Texture dimensions
uniform mat4 u_projectionView;     // Combined projection-view matrix
uniform float u_pointSize;         // Point size in pixels

// Outputs
varying vec3 v_color;              // Color based on velocity/force
```

Functional blocks
-----------------
1. Capability probe
	- Detect support for `OES_texture_float` / `EXT_color_buffer_float` and `EXT_float_blend` (if used).
	- Choose float textures (RGBA32F) when available; otherwise fall back to 8-bit encoding for positions.

2. State storage
	- Two RGBA float textures ping-ponged each frame: `positions` and `velocities` (optional).
	- Texture layout: texture width × height = smallest square ≥ particle count. Each texel stores a single particle in RGBA.
	  - RGBA meaning (float mode): R = x, G = y, B = z, A = mass or a flag.
	  - If using velocity texture: RG = vx,vy, B = vz, A = padding.

3. Initialization
  - Initialize a deterministic position array (random or seeded) on the host and upload to the `positions` texture for GPU processing.
  - Optionally keep a tiny, offline CPU reference used only for unit tests / debugging comparisons (not used as a runtime fallback).

4. Update pass (GPGPU)
	- A fullscreen quad draw executes `update.frag` which for each output texel reads N neighbors (or a stochastic subset) and accumulates forces.
  - Stochastic sampling: per-frame sample a subset of neighbors per-particle (e.g., 25% of neighbors) and use temporal accumulation to approximate full force over multiple frames. See "Temporal accumulation" below for implementation details and an efficient accumulation pattern that avoids unbounded state growth.
	- Integration: simple Euler or semi-implicit Euler integration in shader.
	- Edge handling: wrap or clamp to world bounds. Provide damping and max-speed clamps.

5. Render pass
	- Use a `THREE.Points` or an instanced mesh that reads positions via `texture2D` in vertex shader.
	- For each vertex instance read the particle's position from the `positions` texture and output clip-space position.

6. Compatibility fallback
  - If float textures are not available, encode positions into 8-bit RGBA (position packing). CPU-based update fallbacks are intentionally omitted — this project targets GPU-capable devices only.

## Detailed Implementation Specifications

### 1. Capability Detection (Exact Implementation)
```javascript
// Required checks with fallback decisions
const capabilities = {
  webgl2: !!gl.getParameter,
  float_texture: !!gl.getExtension('EXT_color_buffer_float'),
  float_blend: !!gl.getExtension('EXT_float_blend'),
  max_texture_size: gl.getParameter(gl.MAX_TEXTURE_SIZE)
};

// Decision matrix:
if (capabilities.float_texture && capabilities.float_blend) {
  // Use RGBA32F textures (preferred path)
  textureFormat = gl.RGBA32F;
  precision = 'high';
} else {
  // Use RGBA8 with position packing
  textureFormat = gl.RGBA;
  precision = 'medium'; // ~±1e-3 for worldScale=10
}

// Texture size validation
const requiredSize = Math.ceil(Math.sqrt(particleCount));
if (requiredSize > capabilities.max_texture_size) {
  throw new Error(`Particle count ${particleCount} requires texture size ${requiredSize}, but max is ${capabilities.max_texture_size}`);
}
```

### 2. Texture Layout and Index↔UV Mapping
```javascript
// Texture dimensions
const W = Math.ceil(Math.sqrt(particleCount));
const H = W;

// Index to texture coordinates
function indexToUV(index, W, H) {
  const x = index % W;
  const y = Math.floor(index / W);
  return [(x + 0.5) / W, (y + 0.5) / H];
}

// UV to index (for readback)
function uvToIndex(u, v, W, H) {
  const x = Math.floor(u * W);
  const y = Math.floor(v * H);
  return y * W + x;
}

// Texel handling for particles beyond count
// Texels with index >= particleCount should be zeroed/ignored
```

### 3. Stochastic Sampling Strategy
```javascript
// Deterministic pseudo-random sampling
// In fragment shader:
uint hash(uint x) {
  x += (x << 10u);
  x ^= (x >> 6u);
  x += (x << 3u);
  x ^= (x >> 11u);
  x += (x << 15u);
  return x;
}

float random(uint seed, uint index, uint frame) {
  return float(hash(seed + index * 1009u + frame * 2039u)) / 4294967295.0;
}

// Sampling logic:
// NOTE: many GPUs limit dynamic loop iteration counts. See "Shader Loop Limits" below.
int sampleCount = max(1, int(float(u_particleCount) * u_samplingFraction));
for (int i = 0; i < sampleCount; i++) {
  float r = random(u_seed, gl_FragCoord.x + gl_FragCoord.y * u_texSize.x, u_frameCount + i);
  int neighborIndex = int(r * float(u_particleCount));
  // Process neighbor at neighborIndex
}
```

### 4. 8-bit Position Packing (Fallback Implementation)
```glsl
// Pack world position to RGBA8
vec4 packPosition(vec3 worldPos, vec3 worldMin, vec3 worldScale) {
  vec3 normalized = (worldPos - worldMin) / worldScale;
  normalized = clamp(normalized, 0.0, 1.0);
  
  // Use 10-10-10-2 bit packing for higher precision
  uint packed = uint(normalized.x * 1023.0) |
               (uint(normalized.y * 1023.0) << 10u) |
               (uint(normalized.z * 1023.0) << 20u);
  
  return vec4(
    float((packed >> 0u) & 0xFFu) / 255.0,
    float((packed >> 8u) & 0xFFu) / 255.0,
    float((packed >> 16u) & 0xFFu) / 255.0,
    float((packed >> 24u) & 0xFFu) / 255.0
  );
}

// Unpack RGBA8 to world position
vec3 unpackPosition(vec4 packed, vec3 worldMin, vec3 worldScale) {
  uint reconstructed = uint(packed.x * 255.0) |
                      (uint(packed.y * 255.0) << 8u) |
                      (uint(packed.z * 255.0) << 16u) |
                      (uint(packed.w * 255.0) << 24u);
  
  vec3 normalized = vec3(
    float((reconstructed >> 0u) & 0x3FFu) / 1023.0,
    float((reconstructed >> 10u) & 0x3FFu) / 1023.0,
    float((reconstructed >> 20u) & 0x3FFu) / 1023.0
  );
  
  return worldMin + normalized * worldScale;
}

// Expected precision: for worldScale=10, resolution ≈ 10/1023 ≈ 0.01 units

Shader Loop Limits
------------------
Many mobile and integrated GPUs impose strict limits on dynamic fragment-shader loop iterations (commonly <= 255). Relying on a dynamic `sampleCount` that scales with `u_particleCount` can therefore break on some hardware. Use one of these robust patterns in `update.frag`:

- Fixed maximum with early exit: choose a safe MAX_SAMPLES (e.g. 256) and iterate that many times, using a uniform or computed `effectiveSamples = min(sampleCount, MAX_SAMPLES)` and use `if (i >= effectiveSamples) break;` inside the loop. This keeps loop count bounded while supporting variable sampling fractions.

- Tiled or texture-driven indexing: precompute a small sampling-lookup texture or index buffer on the host that supplies neighbor indices per-sample. The loop then iterates a fixed small number and fetches neighbor indices from the lookup texture.

- Two-phase approach: for high sampling fractions, split into multiple passes where each pass samples a bounded subset.

Use these patterns in preference to unbounded dynamic loops.

Temporal Accumulation (stochastic sampling)
-----------------------------------------
To approximate the full N×N force over time while sampling only a fraction each frame, accumulate partial force estimators across frames. Implementation pattern:

1. Each frame the shader computes `partialForce` from the sampled neighbors.
2. The shader writes the new velocity/position based on `partialForce * (1.0 / u_samplingFraction)` to unbiasedly estimate the full-sum when sampling is uniform.
3. Optionally apply an exponential moving average (EMA) on forces or velocities to reduce variance:

  newVel = mix(oldVel, oldVel + partialForce * (1.0 / u_samplingFraction) * dt, alpha);

  where alpha is a small blending factor (e.g. 0.2) chosen experimentally.

4. Ensure numeric stability by clamping updates and using small dt or lower samplingFraction when needed.

This avoids maintaining extra GPU-side accumulation buffers while producing a consistent long-term approximation.

Force Model Clarification
-------------------------
The example `computePairForce` uses an inverse-square formula (mass1 * mass2 / d^2) and is intended as a gravitational-like interaction for demonstration. The exact force law is configurable; typical options:

- Gravitational (attractive): F = G * m1 * m2 / (d^2 + eps)
- Repulsive (electrostatic-like): F = k * q1 * q2 / (d^2 + eps)
- Short-range softening: multiply by a falloff or cutoff for large distances

Make the force constants (G, k, softening, cutoff) configurable via uniforms so tuning can be done at runtime.

GLSL Bitwise and WebGL Compatibility Note
----------------------------------------
Bitwise operations and integer/uint types used in packing code require GLSL ES 3.00 / WebGL2 and may still vary across drivers. If bitwise `uint` operations prove incompatible on target devices, provide a fallback packing/unpacking path implemented in JS using Float32Array uploads and RGBA byte slicing, or use 4×8-bit channel packing arithmetic that avoids `uint` shifts (e.g. multiply/add with 255.0). Detect support at runtime and choose the appropriate path.
```

### 5. Integration Methods
```glsl
// Euler integration
vec3 integrateEuler(vec3 position, vec3 velocity, vec3 force, float mass, float dt) {
  vec3 acceleration = force / mass;
  velocity += acceleration * dt;
  position += velocity * dt;
  return position;
}

// Semi-implicit Euler (more stable)
vec3 integrateSemiImplicit(vec3 position, vec3 velocity, vec3 force, float mass, float dt) {
  vec3 acceleration = force / mass;
  velocity += acceleration * dt;  // Update velocity first
  position += velocity * dt;      // Then position with new velocity
  return position;
}

// Numerical stability limits
const float MAX_VELOCITY = 100.0;
const float MAX_FORCE = 1000.0;
velocity = clamp(velocity, vec3(-MAX_VELOCITY), vec3(MAX_VELOCITY));
force = clamp(force, vec3(-MAX_FORCE), vec3(MAX_FORCE));
```

### 6. World Bounds Handling
```glsl
// Wrap mode
vec3 applyWrapBounds(vec3 position, vec3 minBounds, vec3 maxBounds) {
  vec3 size = maxBounds - minBounds;
  position = mod(position - minBounds, size) + minBounds;
  return position;
}

// Clamp mode with velocity reflection
vec3 applyClampBounds(vec3 position, inout vec3 velocity, vec3 minBounds, vec3 maxBounds) {
  if (position.x < minBounds.x) { position.x = minBounds.x; velocity.x = abs(velocity.x) * 0.8; }
  if (position.x > maxBounds.x) { position.x = maxBounds.x; velocity.x = -abs(velocity.x) * 0.8; }
  // Repeat for y, z
  return position;
}
```

Data shapes and sizes
---------------------
- For P particles choose W = ceil(sqrt(P)), H = W and place particles in first P texels.
- Using RGBA32F: one texel per particle, 16 bytes/particle. For 100k particles ~1.6MB texture memory (plus ping-pong), which is acceptable.
- For 1M particles ~16MB per texture (two textures ~32MB) — still in range for modern GPUs but watch VRAM.

### Memory Requirements Table
| Particle Count | Texture Size | RGBA32F Memory | RGBA8 Memory | Total (Float) |
|----------------|--------------|----------------|--------------|---------------|
| 100,000        | 316×316      | 1.6MB          | 0.4MB        | 6.4MB         |
| 500,000        | 708×708      | 8.0MB          | 2.0MB        | 32.0MB        |
| 1,000,000      | 1000×1000    | 16.0MB         | 4.0MB        | 64.0MB        |
| 2,000,000      | 1415×1415    | 32.0MB         | 8.0MB        | 128.0MB       |

Note: "Total (Float)" includes ping-pong textures (positions + velocities) × 2 (i.e., 4 textures). These are conservative estimates and do not include additional GL buffers, program binaries, or driver overhead.

## Performance Monitoring and Instrumentation

### PerformanceMonitor Class
```javascript
class PerformanceMonitor {
  constructor() {
    this.frameCount = 0;
    this.startTime = performance.now();
    this.frameTimes = [];
    this.gpuTimerExt = null; // EXT_disjoint_timer_query_webgl2
  }

  beginFrame() {
    this.frameStart = performance.now();
    // GPU timer query if available
  }

  endFrame() {
    const frameTime = performance.now() - this.frameStart;
    this.frameTimes.push(frameTime);
    if (this.frameTimes.length > 60) this.frameTimes.shift();
    this.frameCount++;
  }

  getMetrics() {
    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    return {
      fps: 1000 / avgFrameTime,
      frameTimeMs: avgFrameTime,
      frameTimeP95: this.percentile(this.frameTimes, 0.95),
      totalFrames: this.frameCount,
      uptimeSeconds: (performance.now() - this.startTime) / 1000
    };
  }

  estimateMemoryUsage(particleCount, useFloat) {
    const bytesPerTexel = useFloat ? 16 : 4; // RGBA32F vs RGBA8
    const textureSize = Math.ceil(Math.sqrt(particleCount));
    const texelsPerTexture = textureSize * textureSize;
    const textureCount = 4; // positions, velocities × 2 (ping-pong)
    return {
      textureSizePx: textureSize,
      totalTexels: texelsPerTexture * textureCount,
      memoryMB: (texelsPerTexture * textureCount * bytesPerTexel) / (1024 * 1024)
    };
  }
}
```

Shader sketches
---------------
- `pass.vert` — full-screen quad vertex shader (standard)

- `update.frag` (complete implementation template)
```glsl
#version 300 es
precision highp float;

// Input uniforms (see API contract above)
uniform sampler2D u_positions;
uniform sampler2D u_velocities;
uniform float u_time;
uniform float u_dt;
uniform int u_frameCount;
uniform float u_samplingFraction;
uniform vec2 u_texSize;
uniform int u_particleCount;
uniform vec3 u_worldMin;
uniform vec3 u_worldMax;
uniform uint u_seed;
uniform int u_integrationMethod;
uniform int u_wrapMode;

out vec4 fragColor;

// RNG implementation
uint hash(uint x) {
  x += (x << 10u);
  x ^= (x >> 6u);
  x += (x << 3u);
  x ^= (x >> 11u);
  x += (x << 15u);
  return x;
}

float random(uint seed, uint index, uint frame) {
  return float(hash(seed + index * 1009u + frame * 2039u)) / 4294967295.0;
}

// Index/UV conversion
vec2 indexToUV(int index) {
  int x = index % int(u_texSize.x);
  int y = index / int(u_texSize.x);
  return (vec2(x, y) + 0.5) / u_texSize;
}

// Force calculation
vec3 computePairForce(vec3 pos1, vec3 pos2, float mass1, float mass2) {
  vec3 dir = pos2 - pos1;
  float d2 = dot(dir, dir) + 1e-6; // Softening
  float d = sqrt(d2);
  float force = mass1 * mass2 / d2;
  return force * normalize(dir);
}

void main() {
  // Current particle index
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int myIndex = coord.y * int(u_texSize.x) + coord.x;
  
  if (myIndex >= u_particleCount) {
    fragColor = vec4(0.0);
    return;
  }

  // Read current state
  vec2 myUV = (vec2(coord) + 0.5) / u_texSize;
  vec4 posData = texture(u_positions, myUV);
  vec4 velData = texture(u_velocities, myUV);
  
  vec3 myPos = posData.xyz;
  float myMass = posData.w;
  vec3 myVel = velData.xyz;
  
  // Accumulate forces from neighbors
  vec3 totalForce = vec3(0.0);
  int sampleCount = max(1, int(float(u_particleCount) * u_samplingFraction));
  
  for (int i = 0; i < sampleCount; i++) {
    float r = random(u_seed, uint(myIndex), uint(u_frameCount + i));
    int neighborIndex = int(r * float(u_particleCount));
    
    if (neighborIndex == myIndex) continue;
    
    vec2 neighborUV = indexToUV(neighborIndex);
    vec4 neighborPosData = texture(u_positions, neighborUV);
    vec3 neighborPos = neighborPosData.xyz;
    float neighborMass = neighborPosData.w;
    
    totalForce += computePairForce(myPos, neighborPos, myMass, neighborMass);
  }
  
  // Integration
  vec3 acceleration = totalForce / myMass;
  vec3 newVel, newPos;
  
  if (u_integrationMethod == 0) { // Euler
    newVel = myVel + acceleration * u_dt;
    newPos = myPos + newVel * u_dt;
  } else { // Semi-implicit Euler
    newVel = myVel + acceleration * u_dt;
    newPos = myPos + newVel * u_dt;
  }
  
  // Apply bounds
  if (u_wrapMode == 0) { // Wrap
    vec3 size = u_worldMax - u_worldMin;
    newPos = mod(newPos - u_worldMin, size) + u_worldMin;
  } else { // Clamp with bounce
    newPos = clamp(newPos, u_worldMin, u_worldMax);
  }
  
  // Stability limits
  newVel = clamp(newVel, vec3(-100.0), vec3(100.0));
  
  fragColor = vec4(newPos, myMass);
}
```

- `render.vert` (complete implementation template)
```glsl
#version 300 es

in float a_index;

uniform sampler2D u_positions;
uniform vec2 u_texSize;
uniform mat4 u_projectionView;
uniform float u_pointSize;

out vec3 v_color;

vec2 indexToUV(float index) {
  float x = mod(index, u_texSize.x);
  float y = floor(index / u_texSize.x);
  return (vec2(x, y) + 0.5) / u_texSize;
}

void main() {
  vec2 uv = indexToUV(a_index);
  vec4 posData = texture(u_positions, uv);
  vec3 worldPos = posData.xyz;
  
  gl_Position = u_projectionView * vec4(worldPos, 1.0);
  gl_PointSize = u_pointSize;
  
  // Color based on position for visualization
  v_color = normalize(worldPos) * 0.5 + 0.5;
}
```

Limitations & mitigations
-------------------------
- Limitation: Brute force is O(N²) in exact form — mitigated by stochastic sampling (draws partial neighborhood per frame) and temporal accumulation.
  - Mitigation: expose sampling fraction UI; start with 25% sampling and progressive increase if device can handle it.
- Limitation: precision on devices without float textures.
  - Mitigation: implement compact encoding (pack floats into 8-bit channels) as the compatibility fallback.
- Limitation: memory for large particle counts.
  - Mitigation: allow adjustable particle count at runtime and display memory estimate in HUD; recommend testing first at 100k.

Testing, verification and benchmarks
-----------------------------------
### Comprehensive Test Plan

#### 1. Smoke Tests (Basic Functionality)
```javascript
// Test: Basic initialization and rendering
async function testSmoke() {
  const gl = getWebGL2Context();
  const planA = new PlanA(gl, { particleCount: 1000 });
  await planA.init();
  
  // Should render without GL errors
  planA.step();
  assert(gl.getError() === gl.NO_ERROR, "No GL errors during step");
  
  // Positions should change over time
  const positions1 = await planA.readback(10);
  planA.step();
  const positions2 = await planA.readback(10);
  assert(!arraysEqual(positions1, positions2), "Positions should change");
  
  planA.dispose();
}
```

#### 2. Deterministic Physics Test
```javascript
// Test: Validate against CPU reference implementation
async function testDeterministic() {
  const seed = 12345;
  const particleCount = 100;
  
  // GPU implementation
  const planA = new PlanA(gl, { particleCount, seed, samplingFraction: 1.0 });
  await planA.init();
  
  for (let i = 0; i < 10; i++) planA.step();
  const gpuPositions = await planA.readback(particleCount);
  
  // CPU reference (simplified)
  const cpuPositions = runCPUReference(particleCount, seed, 10);
  
  // Compare within tolerance
  for (let i = 0; i < particleCount; i++) {
    const gpuPos = gpuPositions[i];
    const cpuPos = cpuPositions[i];
    const distance = Math.sqrt(
      (gpuPos.x - cpuPos.x) ** 2 +
      (gpuPos.y - cpuPos.y) ** 2 +
      (gpuPos.z - cpuPos.z) ** 2
    );
    assert(distance < 0.01, `Particle ${i} position error: ${distance}`);
  }
}
```

#### 3. Sampling Coverage Test
```javascript
// Test: Ensure stochastic sampling covers all neighbors over time
async function testSamplingCoverage() {
  const planA = new PlanA(gl, { 
    particleCount: 1000, 
    samplingFraction: 0.1,
    seed: 42 
  });
  await planA.init();
  
  // Track which neighbors are sampled for particle 0
  const sampledNeighbors = new Set();
  const maxFrames = 100;
  
  for (let frame = 0; frame < maxFrames; frame++) {
    // Hook into shader to track sampling (test-only)
    const neighbors = getFrameSampledNeighbors(planA, 0);
    neighbors.forEach(n => sampledNeighbors.add(n));
    planA.step();
  }
  
  // Should sample significant portion of particle space
  const coverageRatio = sampledNeighbors.size / 1000;
  assert(coverageRatio > 0.8, `Coverage too low: ${coverageRatio}`);
}
```

#### 4. Precision Test (8-bit Fallback)
```javascript
// Test: Validate 8-bit packing precision
function testPackingPrecision() {
  const worldBounds = { min: [-10, -10, -10], max: [10, 10, 10] };
  const testPositions = [
    [0, 0, 0],
    [5.5, -3.2, 8.1],
    [-9.9, 9.9, -9.9]
  ];
  
  testPositions.forEach(pos => {
    const packed = packPosition(pos, worldBounds);
    const unpacked = unpackPosition(packed, worldBounds);
    const error = distance(pos, unpacked);
    assert(error < 0.02, `Packing error too high: ${error} for ${pos}`);
  });
}
```

#### 5. Performance Regression Test
```javascript
// Test: Ensure performance meets baseline requirements
async function testPerformanceBaseline() {
  const planA = new PlanA(gl, { particleCount: 100000 });
  await planA.init();
  
  // Warm up
  for (let i = 0; i < 10; i++) planA.step();
  
  // Measure
  const startTime = performance.now();
  const frameCount = 60;
  for (let i = 0; i < frameCount; i++) {
    planA.step();
  }
  const totalTime = performance.now() - startTime;
  const avgFrameTime = totalTime / frameCount;
  const fps = 1000 / avgFrameTime;
  
  assert(fps >= 30, `FPS too low: ${fps} (target: 30+ fps for 100k particles)`);
  assert(avgFrameTime <= 33.3, `Frame time too high: ${avgFrameTime}ms`);
}
```

#### 6. Resource Leak Test
```javascript
// Test: Ensure proper cleanup
async function testResourceLeaks() {
  const initialInfo = gl.getExtension('WEBGL_debug_renderer_info');
  let planA;
  
  for (let i = 0; i < 5; i++) {
    planA = new PlanA(gl, { particleCount: 10000 });
    await planA.init();
    planA.step();
    planA.dispose();
  }
  
  // Check for obvious leaks (textures, programs, buffers)
  const programs = gl.getParameter(gl.ACTIVE_PROGRAM);
  const textures = gl.getParameter(gl.ACTIVE_TEXTURE);
  // Note: WebGL doesn't expose direct resource counts, but we can
  // check that context is still valid and no GL errors accumulated
  assert(gl.getError() === gl.NO_ERROR, "GL errors indicate resource issues");
}
```

### Performance Acceptance Criteria
| Hardware Class | Particle Count | Target FPS | Max Frame Time |
|----------------|----------------|------------|----------------|
| Desktop High   | 1,000,000      | 60 fps     | 16.7ms        |
| Desktop Mid    | 500,000        | 60 fps     | 16.7ms        |
| Desktop Low    | 100,000        | 30 fps     | 33.3ms        |
| Mobile High    | 50,000         | 30 fps     | 33.3ms        |
| Mobile Low     | 10,000         | 30 fps     | 33.3ms        |

### Quality Validation Criteria
- **Numerical stability**: Simulation should run for 1000+ frames without particles escaping bounds or energy growing unbounded
- **Visual quality**: No obvious artifacts from stochastic sampling at recommended sampling fractions (0.25+)
- **Determinism**: Same seed and parameters should produce identical results across runs
- **Precision**: 8-bit fallback should maintain ±0.02 unit accuracy for world scales up to 20 units

Developer workflow and milestones (recommended)
---------------------------------------------
### Detailed Implementation Timeline

#### Week 0.5 — Foundation & Scaffolding
**Goal**: Basic project structure and capability detection
- [ ] Create `src/plan-a/index.js` with PlanA class skeleton
- [ ] Implement capability detection (float texture support, max texture size)
- [ ] Create `src/plan-a/utils/gpu-texture.js` with texture creation helpers
- [ ] Add basic error handling and WebGL context validation
- [ ] Set up project build integration

**Deliverable**: PlanA class can be instantiated, detects capabilities, throws meaningful errors

#### Week 1 — Core GPU Pipeline (Float Path)
**Goal**: Working particle simulation with RGBA32F textures
- [ ] Implement texture allocation and ping-pong framebuffers
- [ ] Create and compile update fragment shader with basic force calculation
- [ ] Create render vertex shader that reads positions and draws points
- [ ] Wire shader uniform bindings and attribute setup
- [ ] Add particle initialization (deterministic seeded positions)
- [ ] Basic integration with Three.js rendering pipeline

**Deliverable**: 1000 particles moving and rendering smoothly at 60fps

#### Week 1.5 — Stochastic Sampling & UI Controls
**Goal**: Scalable sampling and user controls
- [ ] Implement deterministic RNG in fragment shader
- [ ] Add stochastic neighbor sampling with configurable fraction
- [ ] Create UI sliders for particle count and sampling fraction
- [ ] Add performance monitoring (FPS, frame time display)
- [ ] Implement dynamic particle count adjustment
- [ ] Add basic HUD showing current parameters

**Deliverable**: 100k particles at 30+ fps with adjustable sampling (0.1-1.0)

#### Week 2 — Fallbacks, Testing & Polish
**Goal**: Production-ready implementation with full test coverage
- [ ] Implement 8-bit position packing for devices without float textures
- [ ] Create comprehensive test suite (smoke, deterministic, precision, performance)
- [ ] Add CPU reference implementation for validation
- [ ] Implement proper resource cleanup and disposal
- [ ] Add memory usage estimation and display
- [ ] Create detailed README with setup and usage instructions
- [ ] Performance optimization and profiling

**Deliverable**: Robust implementation passing all tests, with fallbacks, ready for production

### Daily Checkpoint Structure
Each development day should include:
1. **Morning**: Run smoke test, check FPS baseline
2. **Development**: Implement feature with unit test
3. **Evening**: Run full test suite, commit working state

### Integration Milestones
- **Day 3**: Integration with main app UI (button to launch Plan A)
- **Day 7**: Performance baseline established (100k particles, benchmark data)
- **Day 10**: Fallback path tested on devices without float textures
- **Day 14**: Full test suite green, documentation complete

Quick run instructions
----------------------
From the project root:

```cmd
npm install
npm run build
npm start
```

Open http://localhost:8333 (or the server address) and select Plan A via HUD (1 or click the button).

Notes on WebGL features and progressive migration
-------------------------------------------------
- Plan A is intentionally conservative: it uses only features available in WebGL2 / standard extensions. Prefer float-texture (EXT_color_buffer_float) but provide fallback.
- As soon as Plan A is stable it becomes a testbed: you can replace the update pass with Plan C's additive-grid passes or Plan M's pyramid passes without touching the render pipeline.

## Edge Cases and Implementation Pitfalls

### Critical WebGL Considerations
1. **Fragment Shader Loop Limits**: Many GPUs have maximum loop iteration limits (often 255). For large particle counts with high sampling fractions, unroll loops or use texture-driven indexing instead of dynamic loops.

2. **Precision Loss in Reductions**: Float addition is not associative. When sampling many neighbors, accumulate forces in a consistent order or use higher-precision intermediate calculations.

3. **Framebuffer Completeness**: Not all devices support all texture formats for rendering. Always check `gl.checkFramebufferStatus()` after FBO creation and provide graceful fallbacks.

4. **Memory Bandwidth**: Large textures can exceed GPU memory bandwidth. Monitor performance as particle count increases and consider texture compression or tiling for very large simulations.

### Error Recovery Strategies
```javascript
// Critical error handling patterns
class PlanA {
  async init() {
    try {
      this.createTextures();
      this.compileShaders();
      this.setupFramebuffers();
    } catch (error) {
      this.dispose(); // Always cleanup on failure
      if (error.name === 'InsufficientMemory') {
        throw new Error(`Not enough GPU memory for ${this.particleCount} particles. Try reducing particle count.`);
      } else if (error.name === 'UnsupportedFormat') {
        throw new Error('Device does not support required WebGL features for Plan A');
      }
      throw error;
    }
  }

  handleRuntimeError(error) {
    // Graceful degradation strategies
    if (error.includes('out of memory')) {
      this.resize(Math.floor(this.particleCount * 0.75));
      console.warn('Reduced particle count due to memory pressure');
    } else if (error.includes('shader')) {
      this.fallbackToCPU = true;
      console.error('GPU shader error, falling back to CPU simulation');
    }
  }
}
```

### Browser Compatibility Matrix
| Browser | Float Textures | Float Blending | Max Texture Size | Notes |
|---------|----------------|----------------|------------------|-------|
| Chrome 90+ | ✅ | ✅ | 16384 | Full support |
| Firefox 85+ | ✅ | ✅ | 16384 | Full support |
| Safari 14+ | ✅ | ⚠️ | 8192 | Limited float blending |
| Edge 90+ | ✅ | ✅ | 16384 | Full support |
| Mobile Chrome | ✅ | ✅ | 4096 | Reduced texture limits |
| Mobile Safari | ⚠️ | ❌ | 4096 | Requires 8-bit fallback |

### Performance Degradation Patterns
1. **Texture Thrashing**: When texture size approaches GPU memory limits, performance drops dramatically. Monitor memory usage and warn users before hitting limits.

2. **Branch Divergence**: Different particles taking different code paths in shaders causes severe performance penalties. Minimize conditional logic in fragment shaders.

3. **Cache Misses**: Random neighbor sampling creates poor memory access patterns. Consider blue noise or structured sampling patterns for better cache locality.

Appendix: Implementation Checklist for First PR
----------------------------------------------
### Core Implementation Files
- [ ] `src/plan-a/index.js` — Complete PlanA class with all methods from API contract
- [ ] `src/plan-a/gpu/update.frag` — Fragment shader with exact uniform interface specified
- [ ] `src/plan-a/gpu/pass.vert` — Full-screen quad vertex shader 
- [ ] `src/plan-a/gpu/render.vert` — Point rendering vertex shader with texture sampling
- [ ] `src/plan-a/gpu/render.frag` — Point rendering fragment shader
- [ ] `src/plan-a/utils/gpu-texture.js` — Texture and framebuffer creation utilities
- [ ] `src/plan-a/bench.js` — Performance monitoring with metrics from spec above

### Test Files
- [ ] `src/plan-a/tests/smoke.test.js` — Basic functionality tests
- [ ] `src/plan-a/tests/deterministic.test.js` — CPU reference comparison
- [ ] `src/plan-a/tests/precision.test.js` — 8-bit packing validation
- [ ] `src/plan-a/tests/performance.test.js` — Performance regression tests
- [ ] `src/plan-a/tests/resources.test.js` — Memory leak detection

### Documentation
- [ ] `src/plan-a/README.md` — Quick start guide and usage examples
- [ ] Performance benchmarks documented for target hardware
- [ ] API documentation with all public methods and options
- [ ] Known limitations and workarounds documented

### Integration Requirements
- [ ] Integrated with main app UI (button/menu to launch Plan A)
- [ ] HUD displays particle count, FPS, memory usage, sampling fraction
- [ ] UI controls for runtime parameter adjustment
- [ ] Graceful error handling with user-friendly error messages
- [ ] Resource cleanup on app shutdown/navigation

### Validation Criteria for PR Approval
1. **Functionality**: All smoke tests pass, particles render and move correctly
2. **Performance**: Meets baseline performance targets for 100k particles
3. **Compatibility**: Works on both float-texture and 8-bit fallback paths
4. **Quality**: Deterministic tests pass with <1% error tolerance
5. **Robustness**: Handles edge cases (zero particles, max texture size, out of memory)
6. **Documentation**: Complete setup instructions and API reference

### File Size Estimates (for review scope)
- Core implementation: ~800-1200 lines of JavaScript + GLSL
- Tests: ~400-600 lines of JavaScript
- Documentation: ~200-300 lines of Markdown
- Total: ~1400-2100 lines (manageable PR size)

References and Technical Background
-----------------------------------
### Core Technologies
- **WebGL2 Specification**: [Khronos WebGL2 Spec](https://www.khronos.org/registry/webgl/specs/latest/2.0/)
- **GPGPU Particle Systems**: Classic ping-pong texture techniques for particle simulation
- **Stochastic Sampling**: Monte Carlo methods for approximating expensive computations
- **Texture Packing**: Encoding floating-point data in 8-bit textures for compatibility

### Key WebGL Extensions
- `EXT_color_buffer_float`: Enables rendering to floating-point textures
- `EXT_float_blend`: Enables blending operations on floating-point render targets  
- `EXT_disjoint_timer_query_webgl2`: GPU timing for performance profiling
- `WEBGL_debug_renderer_info`: Device capability detection

### Algorithm References
- **N-body Simulation**: Classical gravitational/electrostatic force calculations
- **Verlet Integration**: Numerical integration methods for particle systems
- **Spatial Hashing**: Optimization techniques for neighbor finding (future Plans C/D/M)
- **Barnes-Hut Algorithm**: Hierarchical force approximation (Plan M reference)

### Performance Optimization Patterns
- **Texture Memory Layout**: Optimal data organization for GPU cache efficiency
- **Fragment Shader Optimization**: Minimizing branches and memory access patterns
- **WebGL Best Practices**: Resource management and error handling patterns

---

End of Complete Plan A Implementation Specification