/**
 * Plan M: "The Menace" — GPU-side Dynamic Quadtree
 * 
 * Implements a GPU-resident quadtree using WebGL2 fragment shaders for large-scale
 * particle simulation with O(N log N) complexity via Barnes-Hut algorithm.
 */

import * as THREE from 'three';

// Externalized shader sources and utilities
import fsQuadVert from './shaders/fullscreen.vert.js';
import reductionFrag from './shaders/reduction.frag.js';
import aggregationVert from './shaders/aggregation.vert.js';
import aggregationFrag from './shaders/aggregation.frag.js';
import traversalFrag from './shaders/traversal.frag.js';
import renderVert from './shaders/render.vert.js';
import renderFrag from './shaders/render.frag.js';
import velIntegrateFrag from './shaders/vel_integrate.frag.js';
import posIntegrateFrag from './shaders/pos_integrate.frag.js';
import { unbindAllTextures as dbgUnbindAllTextures, checkGl as dbgCheckGl, checkFBO as dbgCheckFBO } from './utils/debug.js';
import { aggregateParticlesIntoL0 as aggregateL0 } from './pipeline/aggregator.js';
import { runReductionPass as pyramidReduce } from './pipeline/pyramid.js';
import { calculateForces as pipelineCalculateForces } from './pipeline/traversal.js';
import { integratePhysics as pipelineIntegratePhysics } from './pipeline/integrator.js';
import { updateWorldBoundsFromTexture as pipelineUpdateBounds } from './pipeline/bounds.js';
import { renderParticles as pipelineRenderParticles } from './pipeline/renderer.js';

export default class PlanM {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.gl = renderer ? renderer.getContext() : null;
    
    // Configuration
    this.options = {
      particleCount: 50000,
      dt: 10 / 60,
      gravityStrength: 0.0003,
      theta: 0.5,
      softening: 0.05,
      pointSize: 10.0,
      worldBounds: {
        min: [-5, -5, -5],
        max: [5, 5, 5]
      }
    };
    
    // Internal state
    this._objects = [];
    this.isInitialized = false;
    this.frameCount = 0;
    this.time = 0;
    this.running = false;
    this._cameraInfoLogged = false;
    
    // GPU resources
    this.levelTextures = [];
    this.levelFramebuffers = [];
    this.positionTextures = null;
    this.velocityTextures = null;
    this.forceTexture = null;
    this.programs = {};
    this.quadVAO = null;
    this.particleVAO = null;
    this.textureWidth = 0;
    this.textureHeight = 0;
    this.numLevels = 0;
    this.L0Size = 0;
    this._disableFloatBlend = false;
    this._quadtreeDisabled = false;
    this._orthoCam = null;
    this._lastBoundsUpdateFrame = -1;
  }

  // Debug helper: unbind all textures on commonly used units to avoid feedback loops
  unbindAllTextures() {
    dbgUnbindAllTextures(this.gl);
  }

  // Debug helper: log gl errors with a tag
  checkGl(tag) {
    return dbgCheckGl(this.gl, tag);
  }

  // Debug helper: check FBO completeness and tag
  checkFBO(tag) {
    dbgCheckFBO(this.gl, tag);
  }

  async init() {
    if (!this.gl) {
      throw new Error('WebGL2 context not available');
    }
    
    console.log('Initializing Plan M: GPU-side Dynamic Quadtree...');
    
    try {
      this.checkWebGL2Support();
      this.calculateTextureDimensions();
      this.createShaderPrograms();
      this.createTextures();
      this.createGeometry();
      this.initializeParticles();
      pipelineUpdateBounds(this, 512);
      
      this.isInitialized = true;
      console.log('Plan M initialized successfully');
      
    } catch (error) {
      console.error('Plan M initialization failed:', error);
      this.dispose();
      throw error;
    }
  }

  checkWebGL2Support() {
    const gl = this.gl;
    
    const colorBufferFloat = gl.getExtension('EXT_color_buffer_float');
    const floatBlend = gl.getExtension('EXT_float_blend');
    
    if (!colorBufferFloat) {
      throw new Error('EXT_color_buffer_float extension not supported');
    }
    
    if (!floatBlend) {
      throw new Error('EXT_float_blend extension not supported: required for additive blending to float textures');
    }
    
    const caps = {
      maxVertexTextureUnits: gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS),
      maxTextureUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
      maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS),
    };
    console.log('WebGL caps:', caps);

    console.log('WebGL2 extensions check passed');
  }

  calculateTextureDimensions() {
    // Octree configuration: 64³ voxels at L0
    this.octreeGridSize = 64; // 64×64×64 = 262,144 voxels
    this.numLevels = 7; // L0(64³) → L1(32³) → L2(16³) → L3(8³) → L4(4³) → L5(2³) → L6(1³)
    
    // Calculate texture sizes for each octree level
    // Store 3D grid as 2D texture by stacking Z-slices
    // For 64³: arrange 64 slices of 64×64 in an 8×8 grid → 512×512 texture
    this.levelGridSizes = [64, 32, 16, 8, 4, 2, 1]; // Grid size at each level
    const slicesPerRow = 8; // How many Z-slices per row in texture
    
    // L0: 512×512 to hold 64 slices of 64×64
    this.L0Size = this.octreeGridSize * slicesPerRow;
    
    // Particle texture dimensions (unchanged)
    this.textureWidth = Math.ceil(Math.sqrt(this.options.particleCount));
    this.textureHeight = Math.ceil(this.options.particleCount / this.textureWidth);
    this.actualTextureSize = this.textureWidth * this.textureHeight;
    
    console.log(`Octree: L0=${this.octreeGridSize}³ voxels (${this.L0Size}×${this.L0Size} texture), ${this.numLevels} levels`);
    console.log(`  Level sizes: ${this.levelGridSizes.map((s, i) => `L${i}=${s}³`).join(', ')}`);
    console.log(`Position texture: ${this.textureWidth}×${this.textureHeight} for ${this.options.particleCount} particles (${this.actualTextureSize} total texels)`);
  }

  createShaderPrograms() {
    const gl = this.gl;
    
    this.programs.aggregation = this.createProgram(aggregationVert, aggregationFrag, { 'a_particleIndex': 0 });
    this.programs.reduction = this.createProgram(fsQuadVert, reductionFrag);
    this.programs.traversal = this.createProgram(fsQuadVert, traversalFrag);
    this.programs.velIntegrate = this.createProgram(fsQuadVert, velIntegrateFrag);
    this.programs.posIntegrate = this.createProgram(fsQuadVert, posIntegrateFrag);
    this.programs.render = this.createProgram(renderVert, renderFrag);
    
    console.log('Shader programs created successfully');
  }

  createProgram(vertexSource, fragmentSource, bindAttribs = null) {
    const gl = this.gl;
    
    const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fragmentSource);
    
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    
    // Bind attribute locations before linking if specified
    if (bindAttribs) {
      for (const [name, location] of Object.entries(bindAttribs)) {
        gl.bindAttribLocation(program, location, name);
      }
    }
    
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Shader program link failed: ${info}`);
    }
    
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    
    return program;
  }

  createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile failed: ${info}\nSource:\n${source}`);
    }
    
    return shader;
  }

  createTextures() {
    const gl = this.gl;
    
    this.levelTextures = [];
    this.levelFramebuffers = [];
    
    // Create octree level textures with appropriate sizes
    // Each level's texture size is calculated based on 3D grid mapped to 2D
    const slicesPerRow = 8;
    for (let i = 0; i < this.numLevels; i++) {
      const gridSize = this.levelGridSizes[i]; // e.g., 64, 32, 16, 8, 4, 2, 1
      const textureSize = gridSize * slicesPerRow; // e.g., 512, 256, 128, 64, 32, 16, 8
      
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, textureSize, textureSize, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      
      this.levelTextures.push({texture, size: textureSize, gridSize});
      this.levelFramebuffers.push(framebuffer);
      
      console.log(`  Created octree level ${i}: grid=${gridSize}³, texture=${textureSize}×${textureSize}`);
    }
    
    this.positionTextures = this.createPingPongTextures(this.textureWidth, this.textureHeight);
    this.velocityTextures = this.createPingPongTextures(this.textureWidth, this.textureHeight);
    this.forceTexture = this.createRenderTexture(this.textureWidth, this.textureHeight);
    
    console.log(`Created ${this.numLevels} octree level textures and particle textures`);
  }

  createPingPongTextures(width, height) {
    const gl = this.gl;
    const textures = [];
    const framebuffers = [];
    
    for (let i = 0; i < 2; i++) {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      
      textures.push(texture);
      framebuffers.push(framebuffer);
    }
    
    return {
      textures,
      framebuffers,
      currentIndex: 0,
      getCurrentTexture: function() { return this.textures[this.currentIndex]; },
      getTargetTexture: function() { return this.textures[1 - this.currentIndex]; },
      getTargetFramebuffer: function() { return this.framebuffers[1 - this.currentIndex]; },
      swap: function() { this.currentIndex = 1 - this.currentIndex; }
    };
  }

  createRenderTexture(width, height) {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

    return { texture, framebuffer };
  }

  createGeometry() {
    const gl = this.gl;
    
    const quadVertices = new Float32Array([
      -1, -1,  1, -1,  -1, 1,  1, 1
    ]);
    
    this.quadVAO = gl.createVertexArray();
    gl.bindVertexArray(this.quadVAO);
    
    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    
    const particleIndices = new Float32Array(this.options.particleCount);
    for (let i = 0; i < this.options.particleCount; i++) {
      particleIndices[i] = i;
    }
    
    this.particleVAO = gl.createVertexArray();
    gl.bindVertexArray(this.particleVAO);
    
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, particleIndices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
    
    gl.bindVertexArray(null);
    
    // Create minimal VAO for aggregation (no attributes, just to trigger vertex count)
    this.aggregationVAO = gl.createVertexArray();
    gl.bindVertexArray(this.aggregationVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);  // Bind buffer but don't enable attributes
    gl.bindVertexArray(null);
  }

  initializeParticles() {
    const positions = new Float32Array(this.actualTextureSize * 4);
    const velocities = new Float32Array(this.actualTextureSize * 4);
    
    const bounds = this.options.worldBounds;
    const center = [
      (bounds.min[0] + bounds.max[0]) / 2,
      (bounds.min[1] + bounds.max[1]) / 2,
      (bounds.min[2] + bounds.max[2]) / 2
    ];
    const speed = (this.options.initialSpeed !== undefined) ? this.options.initialSpeed : 0.05;
    
    for (let i = 0; i < this.options.particleCount; i++) {
      const base = i * 4;
      
      // Uniform spherical distribution using Marsaglia's method
      let x, y, z, r2;
      do {
        x = Math.random() * 2 - 1;
        y = Math.random() * 2 - 1;
        z = Math.random() * 2 - 1;
        r2 = x*x + y*y + z*z;
      } while (r2 > 1 || r2 < 0.01);
      
      // Scale to desired radius (0 to 4 units like before)
      const radius = Math.random() * 3 + Math.random() * 1;
      const scale = radius / Math.sqrt(r2);
      
      positions[base + 0] = center[0] + x * scale;
      positions[base + 1] = center[1] + y * scale;
      positions[base + 2] = center[2] + z * scale;
      positions[base + 3] = 0.5 + Math.random() * 1.5; // mass: random range 0.5 to 2.0
      velocities[base + 0] = (Math.random() - 0.5) * 2.0 * speed;
      velocities[base + 1] = (Math.random() - 0.5) * 2.0 * speed;
      velocities[base + 2] = (Math.random() - 0.5) * 2.0 * speed;
      velocities[base + 3] = 0.0;
    }
    
    for (let i = this.options.particleCount; i < this.actualTextureSize; i++) {
      const base = i * 4;
      positions[base + 0] = 0;
      positions[base + 1] = 0;
      positions[base + 2] = 0;
      positions[base + 3] = 0;
      velocities[base + 0] = 0;
      velocities[base + 1] = 0;
      velocities[base + 2] = 0;
      velocities[base + 3] = 0;
    }
    
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < this.options.particleCount; i++) {
      const base = i * 4;
      const x = positions[base + 0];
      const y = positions[base + 1];
      const z = positions[base + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    const padX = Math.max(0.5, 0.1 * Math.max(1e-6, (maxX - minX)));
    const padY = Math.max(0.5, 0.1 * Math.max(1e-6, (maxY - minY)));
    const padZ = Math.max(0.5, 0.1 * Math.max(1e-6, (maxZ - minZ)));
    this.options.worldBounds = {
      min: [minX - padX, minY - padY, minZ - padZ],
      max: [maxX + padX, maxY + padY, maxZ + padZ]
    };

    this.uploadTextureData(this.positionTextures.textures[0], positions);
    this.uploadTextureData(this.positionTextures.textures[1], positions);
    this.uploadTextureData(this.velocityTextures.textures[0], velocities);
    this.uploadTextureData(this.velocityTextures.textures[1], velocities);
    
    console.log(`Particle data initialized: ${this.options.particleCount} particles in ${this.actualTextureSize} texels`);
    console.log(`Initial particle 0: pos=[${positions[0].toFixed(2)}, ${positions[1].toFixed(2)}, ${positions[2].toFixed(2)}] mass=${positions[3]}`);
    console.log(`Initial velocity 0: vel=[${velocities[0].toFixed(3)}, ${velocities[1].toFixed(3)}, ${velocities[2].toFixed(3)}]`);
    console.log(`World bounds after init:`, this.options.worldBounds);
    console.log(`Bounds range: X=[${minX.toFixed(2)}, ${maxX.toFixed(2)}] Y=[${minY.toFixed(2)}, ${maxY.toFixed(2)}] Z=[${minZ.toFixed(2)}, ${maxZ.toFixed(2)}]`);
  }

  uploadTextureData(texture, data) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.textureWidth, this.textureHeight, gl.RGBA, gl.FLOAT, data);
  }

  start() {
    this.running = true;
    
    if (!this.isInitialized && this.gl) {
      this.init().catch(error => {
        console.error('Plan M initialization failed:', error);
        this.createFallbackVisualization();
      });
    } else if (!this.gl) {
      this.createFallbackVisualization();
    }
    
    console.log('Plan M started');
  }

  stop() {
    this.running = false;
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
    console.log('Plan M stopped');
  }

  update() {
    if (!this.running) return;
    
    if (this.isInitialized) {
      this.step();
      if ((this.frameCount % 10) === 0) {
        pipelineUpdateBounds(this, 256);
      }
      
      // Debug: log first particle position periodically
      if (this.frameCount % 60 === 0 && this.frameCount < 300) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.positionTextures.framebuffers[this.positionTextures.currentIndex]);
        const px = new Float32Array(4);
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        console.log(`Frame ${this.frameCount}: P0 pos=[${px[0].toFixed(2)}, ${px[1].toFixed(2)}, ${px[2].toFixed(2)}] mass=${px[3].toFixed(2)}`);
      }
      
      pipelineRenderParticles(this);
    }
  }

  step() {
    if (this.options.debugSkipQuadtree || this._quadtreeDisabled) {
      this.clearForceTexture();
    } else {
      this.buildQuadtree();
      pipelineCalculateForces(this);
    }
    pipelineIntegratePhysics(this);
    
    this.time += this.options.dt;
    this.frameCount++;
  }

  buildQuadtree() {
    const gl = this.gl;
    this.unbindAllTextures();

    gl.disable(gl.SCISSOR_TEST);
    for (let i = 0; i < this.numLevels; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.levelFramebuffers[i]);
      gl.viewport(0, 0, this.levelTextures[i].size, this.levelTextures[i].size);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    
    aggregateL0(this);
    if (this._quadtreeDisabled) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return;
    }
    
    for (let level = 0; level < this.numLevels - 1; level++) {
      pyramidReduce(this, level, level + 1);
    }
  }

  clearForceTexture() {
    const gl = this.gl;
    this.unbindAllTextures();
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.forceTexture.framebuffer);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.textureWidth, this.textureHeight);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  getCameraFromScene() {
    if (!this.scene) return null;
    
    let camera = null;
    this.scene.traverse((child) => {
      if (child.isCamera && !camera) {
        camera = child;
      }
    });
    
    if (!camera && window.camera && window.camera.isCamera) {
      camera = window.camera;
      if (!this._cameraInfoLogged) {
        console.log('Plan M: Found global camera');
        this._cameraInfoLogged = true;
      }
    }
    
    if (!camera) {
      camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(0, 0, 15);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();
      camera.updateProjectionMatrix();
      if (!this._cameraInfoLogged) {
        console.log('Plan M: Using fallback camera');
        this._cameraInfoLogged = true;
      }
    } else {
      if (!this._cameraInfoLogged) {
        console.log('Plan M: Found camera in scene:', camera.constructor.name);
        this._cameraInfoLogged = true;
      }
    }
    
    return camera;
  }

  createFallbackVisualization() {
    if (!this.scene) return;
    
    console.log('Creating Plan M fallback visualization');
    
    const group = new THREE.Group();
    
    const particleGeometry = new THREE.BufferGeometry();
    const count = Math.min(this.options.particleCount, 50000);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 8 + Math.random() * 2;
      const height = (Math.random() - 0.5) * 4;
      
      positions[i * 3 + 0] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = Math.sin(angle) * radius;
      positions[i * 3 + 2] = height;
      
      const dist = Math.sqrt(positions[i * 3] ** 2 + positions[i * 3 + 1] ** 2 + positions[i * 3 + 2] ** 2);
      colors[i * 3 + 0] = 0.2 + 0.8 * (1 - dist / 10);
      colors[i * 3 + 1] = 0.4 + 0.6 * (1 - dist / 10);
      colors[i * 3 + 2] = 0.8;
    }
    
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const particleMaterial = new THREE.PointsMaterial({ 
      size: 0.05,
      transparent: true,
      opacity: 0.8,
      vertexColors: true,
      blending: THREE.AdditiveBlending
    });
    
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    group.add(particles);
    
    const levels = Math.min(6, this.numLevels);
    for (let i = 0; i < levels; i++) {
      const size = Math.pow(2, levels - i) * 0.1;
      const geo = new THREE.PlaneGeometry(size, size);
      const hue = i / levels;
      const mat = new THREE.MeshBasicMaterial({ 
        color: new THREE.Color().setHSL(hue, 0.6, 0.5), 
        side: THREE.DoubleSide, 
        transparent: true, 
        opacity: 0.1 
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(0, 0, -i * 0.03);
      mesh.rotation.x = -Math.PI / 2;
      group.add(mesh);
    }
    
    this.scene.add(group);
    this._objects.push(group);
    
    console.log(`Plan M fallback: ${count} particles with ${levels} quadtree levels`);
  }

  dispose() {
    if (!this.gl) return;
    
    const gl = this.gl;
    
    this.levelTextures.forEach(level => gl.deleteTexture(level.texture));
    this.levelFramebuffers.forEach(fbo => gl.deleteFramebuffer(fbo));
    
    if (this.positionTextures) {
      this.positionTextures.textures.forEach(tex => gl.deleteTexture(tex));
      this.positionTextures.framebuffers.forEach(fbo => gl.deleteFramebuffer(fbo));
    }
    if (this.velocityTextures) {
      this.velocityTextures.textures.forEach(tex => gl.deleteTexture(tex));
      this.velocityTextures.framebuffers.forEach(fbo => gl.deleteFramebuffer(fbo));
    }
    if (this.forceTexture) {
      gl.deleteTexture(this.forceTexture.texture);
      gl.deleteFramebuffer(this.forceTexture.framebuffer);
    }
    
    Object.values(this.programs).forEach(program => gl.deleteProgram(program));
    
    if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
    if (this.particleVAO) gl.deleteVertexArray(this.particleVAO);
    
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
    
    this.isInitialized = false;
    console.log('Plan M disposed');
  }
}
