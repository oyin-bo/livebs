import * as THREE from 'three';

// Plan A - API stub for GPU-driven particle system (texture-based)
// This file provides a developer-friendly starting point. It intentionally
// leaves low-level GL texture/FBO creation to the integrator (TODO markers).

export default class PlanA {
  // Flexible constructor: accept (renderer, scene, options) or (gl, options) or (scene, renderer)
  constructor(a, b, c) {
    this._objects = [];

    // Normalize arguments
    this.scene = null;
    this.renderer = null;
    this.gl = null;
    this.options = {};

    if (a && a.getContext) {
      // a is a canvas-like or raw GL context
      this.gl = a;
      this.options = b || {};
      this.scene = c || null;
    } else if (a && a.isWebGLRenderer) {
      // three.js renderer passed as first arg
      this.renderer = a;
      this.scene = b || null;
      this.gl = this.renderer.getContext();
      this.options = c || {};
    } else {
      // assume (scene, renderer)
      this.scene = a || null;
      this.renderer = b || null;
      this.gl = (this.renderer && this.renderer.getContext && this.renderer.getContext()) || null;
      this.options = c || {};
    }

    this.options = Object.assign({
      particleCount: 100000,
      samplingFraction: 0.25,
      dt: 1 / 60,
      worldBounds: { min: [-1, -1, -1], max: [1, 1, 1] }
    }, this.options);

    this.textures = { positions: null, velocities: null };
    this.framebuffers = { ping: null, pong: null };
    this.stats = { frameTimes: [], lastFrameTime: 0 };

    if (this.gl) {
      this.capabilities = PlanA.capabilityProbe(this.gl);
    } else {
      // conservative defaults if no GL is available at construction time
      this.capabilities = { supportsFloat: false, supportsFloatFBO: false, maxTextureSize: 4096 };
    }

    this.usePacked = !this.capabilities.supportsFloatFBO;

    // Initialize texture layout (will throw if MAX_TEXTURE_SIZE constraint hit)
    this._initLayout();
  }

  static capabilityProbe(gl) {
    const extColorFloat = !!gl.getExtension('EXT_color_buffer_float');
    const extTexFloat = !!gl.getExtension('OES_texture_float');
    const extTexFloatLinear = !!gl.getExtension('OES_texture_float_linear');
    const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);

    return {
      supportsFloat: !!extTexFloat || !!extColorFloat,
      supportsFloatFBO: !!extColorFloat,
      supportsFloatLinear: !!extTexFloatLinear,
      maxTextureSize: maxTexSize,
      suggestedMaxParticles: Math.floor(maxTexSize * maxTexSize)
    };
  }

  _initLayout() {
    const P = this.options.particleCount;
    const W = Math.ceil(Math.sqrt(P));
    const H = W;
    if (W > this.capabilities.maxTextureSize) {
      throw new Error(`particleCount too large for MAX_TEXTURE_SIZE (${this.capabilities.maxTextureSize})`);
    }
    this.texWidth = W;
    this.texHeight = H;
    this.particleCount = P;

    // TODO: create RGBA32F textures and FBOs (ping/pong) here if GL is available
    // this.textures.positions = createTexture(gl, W, H, { internalFormat: gl.RGBA32F, type: gl.FLOAT });
    // this.framebuffers.ping = createFramebufferForTexture(this.textures.positions);
  }

  // Utility to convert index -> uv (center of texel)
  static indexToUV(index, texWidth, texHeight) {
    const x = (index % texWidth) + 0.5;
    const y = (Math.floor(index / texWidth)) + 0.5;
    return [x / texWidth, y / texHeight];
  }

  // Start update loop (caller may implement its own loop)
  start() {
    this.running = true;
  }

  stop() {
    this.running = false;
  }

  // Single GPU update step (advance simulation by dt)
  step(dt = this.options.dt) {
    const gl = this.gl;
    const t0 = (typeof performance !== 'undefined') ? performance.now() : Date.now();

    if (!gl) {
      // fallback: if no GL, keep existing Three.js point demo behavior
      if (this.scene && this._objects.length === 0) this._createFallbackPoints();
      const t1 = (typeof performance !== 'undefined') ? performance.now() : Date.now();
      this.stats.lastFrameTime = t1 - t0;
      return;
    }

    // TODO: bind ping framebuffer, set viewport to texWidth/texHeight,
    // use compiled update program (gpu/update.frag) and draw a full-screen quad
    // The shader will read from this.textures.positions and this.textures.velocities
    // and write the new positions into the bound framebuffer.

    const t1 = (typeof performance !== 'undefined') ? performance.now() : Date.now();
    this.stats.lastFrameTime = t1 - t0;
    this.stats.frameTimes.push(this.stats.lastFrameTime);
    if (this.stats.frameTimes.length > 200) this.stats.frameTimes.shift();
  }

  // Render pass: render points using positions texture (Three.js integration)
  render(projectionViewMatrix) {
    if (!this.renderer || !this.scene) return;

    // If the user provided a scene/renderer, leave rendering to Three.js; if integration
    // with GPU textures is desired, create a BufferGeometry with an `a_index` attribute
    // and a custom shaderMaterial that samples the positions texture in the vertex shader.
    // This method is intentionally minimal; see `src/plan-a/gpu/render.vert` for a vertex shader example.
  }

  getFrameStats() {
    const frames = this.stats.frameTimes;
    if (!frames.length) return { median: 0, mean: 0 };
    const sum = frames.reduce((a, b) => a + b, 0);
    const mean = sum / frames.length;
    const sorted = frames.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    return { mean, median, last: this.stats.lastFrameTime };
  }

  // Utility: create initial position Float32Array (CPU-side) using seed
  static createInitialPositions(particleCount, seed = 1337) {
    const arr = new Float32Array(particleCount * 4); // RGBA texel layout
    let rng = seed & 0xffffffff;
    function rand() {
      // simple xorshift32
      rng ^= rng << 13;
      rng ^= rng >>> 17;
      rng ^= rng << 5;
      return (rng >>> 0) / 4294967295;
    }
    for (let i = 0; i < particleCount; i++) {
      const base = i * 4;
      arr[base + 0] = (rand() * 2 - 1) * 0.5; // x
      arr[base + 1] = (rand() * 2 - 1) * 0.5; // y
      arr[base + 2] = (rand() * 2 - 1) * 0.1; // z
      arr[base + 3] = 1.0; // mass or padding
    }
    return arr;
  }

  // Fallback simple points cloud for environments without GL setup
  _createFallbackPoints() {
    const geom = new THREE.BufferGeometry();
    const count = Math.min(1000, this.options.particleCount);
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3 + 0] = (Math.random() - 0.5) * 4;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 4;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 4;
    }
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: 0x66ccff, size: 0.02 });
    const points = new THREE.Points(geom, mat);
    if (this.scene) {
      this.scene.add(points);
      this._objects.push(points);
    }
  }
}
