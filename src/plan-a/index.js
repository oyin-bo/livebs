/**
 * Plan A - Texture-Based Brute Force with Stochastic Sampling
 * 
 * GPU-driven particle system using ping-pong textures for state storage
 * and stochastic neighbor sampling to reduce computational complexity.
 */

import * as THREE from 'three';
import { GPUTexture, generateParticlePositions, generateParticleVelocities } from './utils/gpu-texture.js';
import { PerformanceMonitor } from './bench.js';

// Import shader sources as strings
const passVertSource = `#version 300 es
precision highp float;

in vec2 a_position;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const updateFragSource = `#version 300 es
precision highp float;

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

const float MAX_VELOCITY = 100.0;
const float MAX_FORCE = 1000.0;
const int MAX_SAMPLES = 256;
const float SOFTENING = 1e-6;
const float DAMPING = 0.8;

out vec4 fragColor;

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

vec2 indexToUV(int index) {
  int x = index % int(u_texSize.x);
  int y = index / int(u_texSize.x);
  return (vec2(x, y) + 0.5) / u_texSize;
}

vec3 computePairForce(vec3 pos1, vec3 pos2, float mass1, float mass2) {
  vec3 dir = pos2 - pos1;
  float d2 = dot(dir, dir) + SOFTENING;
  float force = mass1 * mass2 / d2;
  return force * normalize(dir);
}

vec3 applyWrapBounds(vec3 position, vec3 minBounds, vec3 maxBounds) {
  vec3 size = maxBounds - minBounds;
  position = mod(position - minBounds, size) + minBounds;
  return position;
}

vec3 applyClampBounds(vec3 position, inout vec3 velocity, vec3 minBounds, vec3 maxBounds) {
  if (position.x < minBounds.x) { 
    position.x = minBounds.x; 
    velocity.x = abs(velocity.x) * DAMPING; 
  }
  if (position.x > maxBounds.x) { 
    position.x = maxBounds.x; 
    velocity.x = -abs(velocity.x) * DAMPING; 
  }
  if (position.y < minBounds.y) { 
    position.y = minBounds.y; 
    velocity.y = abs(velocity.y) * DAMPING; 
  }
  if (position.y > maxBounds.y) { 
    position.y = maxBounds.y; 
    velocity.y = -abs(velocity.y) * DAMPING; 
  }
  if (position.z < minBounds.z) { 
    position.z = minBounds.z; 
    velocity.z = abs(velocity.z) * DAMPING; 
  }
  if (position.z > maxBounds.z) { 
    position.z = maxBounds.z; 
    velocity.z = -abs(velocity.z) * DAMPING; 
  }
  return position;
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int myIndex = coord.y * int(u_texSize.x) + coord.x;
  
  if (myIndex >= u_particleCount) {
    fragColor = vec4(0.0);
    return;
  }

  vec2 myUV = (vec2(coord) + 0.5) / u_texSize;
  vec4 posData = texture(u_positions, myUV);
  vec4 velData = texture(u_velocities, myUV);
  
  vec3 myPos = posData.xyz;
  float myMass = posData.w;
  vec3 myVel = velData.xyz;
  
  vec3 totalForce = vec3(0.0);
  int sampleCount = max(1, min(MAX_SAMPLES, int(float(u_particleCount) * u_samplingFraction)));
  
  for (int i = 0; i < MAX_SAMPLES; i++) {
    if (i >= sampleCount) break;
    
    float r = random(u_seed, uint(myIndex), uint(u_frameCount + i));
    int neighborIndex = int(r * float(u_particleCount));
    
    if (neighborIndex == myIndex) continue;
    
    vec2 neighborUV = indexToUV(neighborIndex);
    vec4 neighborPosData = texture(u_positions, neighborUV);
    vec3 neighborPos = neighborPosData.xyz;
    float neighborMass = neighborPosData.w;
    
    totalForce += computePairForce(myPos, neighborPos, myMass, neighborMass);
  }
  
  totalForce *= (1.0 / u_samplingFraction);
  totalForce = clamp(totalForce, vec3(-MAX_FORCE), vec3(MAX_FORCE));
  myVel = clamp(myVel, vec3(-MAX_VELOCITY), vec3(MAX_VELOCITY));
  
  vec3 newPos;
  vec3 newVel = myVel;
  
  if (u_integrationMethod == 0) {
    vec3 acceleration = totalForce / myMass;
    newVel += acceleration * u_dt;
    newPos = myPos + newVel * u_dt;
  } else {
    vec3 acceleration = totalForce / myMass;
    newVel += acceleration * u_dt;
    newPos = myPos + newVel * u_dt;
  }
  
  if (u_wrapMode == 0) {
    newPos = applyWrapBounds(newPos, u_worldMin, u_worldMax);
  } else {
    newPos = applyClampBounds(newPos, newVel, u_worldMin, u_worldMax);
  }
  
  newVel = clamp(newVel, vec3(-MAX_VELOCITY), vec3(MAX_VELOCITY));
  fragColor = vec4(newPos, myMass);
}`;

const renderVertSource = `#version 300 es

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
  
  v_color = normalize(worldPos) * 0.5 + 0.5;
}`;

const renderFragSource = `#version 300 es
precision mediump float;

in vec3 v_color;
out vec4 fragColor;

void main() {
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);
  float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
  vec3 color = v_color * (0.8 + 0.2 * (1.0 - dist));
  fragColor = vec4(color, alpha);
}`;

export default class PlanA {
  constructor(a, b, c) {
    this.gl = null;
    this.scene = null;
    this.renderer = null;
    this._objects = [];
    
    // Handle multiple constructor signatures for backward compatibility
    if (a && a.getContext) {
      // New API: (gl, options)
      this.gl = a;
      this.options = b || {};
      this.scene = c || null;
    } else if (a && a.isWebGLRenderer) {
      // (renderer, scene, options)
      this.renderer = a;
      this.scene = b || null;
      this.gl = this.renderer.getContext();
      this.options = c || {};
    } else {
      // Legacy API: (scene, renderer, options)
      this.scene = a || null;
      this.renderer = b || null;
      this.gl = (this.renderer && this.renderer.getContext && this.renderer.getContext()) || null;
      this.options = c || {};
    }
    
    // Configuration with defaults
    this.options = {
      particleCount: this.options.particleCount || 100000,
      samplingFraction: this.options.samplingFraction || 0.25,
      dt: this.options.dt || 0.016,
      integrationMethod: this.options.integrationMethod || 'semi-implicit',
      wrapMode: this.options.wrapMode || 'wrap',
      worldBounds: this.options.worldBounds || { min: [-10, -10, -10], max: [10, 10, 10] },
      enableVelocityTexture: this.options.enableVelocityTexture !== false,
      seed: this.options.seed || 12345,
      pointSize: this.options.pointSize || 2.0
    };

    // Internal state
    this.isInitialized = false;
    this.frameCount = 0;
    this.time = 0;
    this.running = false;
    this._objects = []; // For Three.js integration
    
    // GPU resources
    this.gpuTexture = null;
    this.positionTextures = null;
    this.velocityTextures = null;
    this.updateProgram = null;
    this.renderProgram = null;
    this.quadVAO = null;
    this.particleVAO = null;
    
    // Performance monitoring
    this.performanceMonitor = new PerformanceMonitor();
    
    // For legacy Three.js integration
    this.threeMesh = null;
  }

  async init() {
    try {
      console.log('Initializing Plan A...', this.options);
      
      this.gpuTexture = new GPUTexture(this.gl);
      console.log('GPU capabilities:', this.gpuTexture.capabilities);
      
      this.gpuTexture.validateParticleCount(this.options.particleCount);
      this.performanceMonitor.initGPUTiming(this.gl);
      
      this.createTextures();
      this.createShaderPrograms();
      this.createGeometry();
      this.initializeParticles();
      
      this.isInitialized = true;
      console.log('Plan A initialized successfully');
      
    } catch (error) {
      this.dispose();
      throw error;
    }
  }

  createTextures() {
    const { width, height } = this.gpuTexture.getTextureDimensions(this.options.particleCount);
    console.log(`Creating textures: ${width}x${height} for ${this.options.particleCount} particles`);
    
    this.positionTextures = this.gpuTexture.createPingPongTextures(width, height);
    
    if (this.options.enableVelocityTexture) {
      this.velocityTextures = this.gpuTexture.createPingPongTextures(width, height);
    }
    
    this.textureWidth = width;
    this.textureHeight = height;
  }

  createShaderPrograms() {
    const gl = this.gl;
    
    this.updateProgram = this.createProgram(passVertSource, updateFragSource);
    
    this.updateUniforms = {
      u_positions: gl.getUniformLocation(this.updateProgram, 'u_positions'),
      u_velocities: gl.getUniformLocation(this.updateProgram, 'u_velocities'),
      u_time: gl.getUniformLocation(this.updateProgram, 'u_time'),
      u_dt: gl.getUniformLocation(this.updateProgram, 'u_dt'),
      u_frameCount: gl.getUniformLocation(this.updateProgram, 'u_frameCount'),
      u_samplingFraction: gl.getUniformLocation(this.updateProgram, 'u_samplingFraction'),
      u_texSize: gl.getUniformLocation(this.updateProgram, 'u_texSize'),
      u_particleCount: gl.getUniformLocation(this.updateProgram, 'u_particleCount'),
      u_worldMin: gl.getUniformLocation(this.updateProgram, 'u_worldMin'),
      u_worldMax: gl.getUniformLocation(this.updateProgram, 'u_worldMax'),
      u_seed: gl.getUniformLocation(this.updateProgram, 'u_seed'),
      u_integrationMethod: gl.getUniformLocation(this.updateProgram, 'u_integrationMethod'),
      u_wrapMode: gl.getUniformLocation(this.updateProgram, 'u_wrapMode')
    };

    this.renderProgram = this.createProgram(renderVertSource, renderFragSource);
    
    this.renderUniforms = {
      u_positions: gl.getUniformLocation(this.renderProgram, 'u_positions'),
      u_texSize: gl.getUniformLocation(this.renderProgram, 'u_texSize'),
      u_projectionView: gl.getUniformLocation(this.renderProgram, 'u_projectionView'),
      u_pointSize: gl.getUniformLocation(this.renderProgram, 'u_pointSize')
    };
  }

  createProgram(vertexSource, fragmentSource) {
    const gl = this.gl;
    
    const vertexShader = this.createShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fragmentSource);
    
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
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

  createGeometry() {
    const gl = this.gl;
    
    const quadVertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1
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
    
    const a_index = gl.getAttribLocation(this.renderProgram, 'a_index');
    gl.enableVertexAttribArray(a_index);
    gl.vertexAttribPointer(a_index, 1, gl.FLOAT, false, 0, 0);
    
    gl.bindVertexArray(null);
  }

  initializeParticles() {
    const positions = generateParticlePositions(
      this.options.particleCount, 
      this.options.seed, 
      this.options.worldBounds
    );
    
    this.gpuTexture.uploadParticleData(
      this.positionTextures.textures[0], 
      this.textureWidth, 
      this.textureHeight, 
      positions
    );
    this.gpuTexture.uploadParticleData(
      this.positionTextures.textures[1], 
      this.textureWidth, 
      this.textureHeight, 
      positions
    );
    
    if (this.velocityTextures) {
      const velocities = generateParticleVelocities(
        this.options.particleCount, 
        this.options.seed + 1
      );
      
      this.gpuTexture.uploadParticleData(
        this.velocityTextures.textures[0], 
        this.textureWidth, 
        this.textureHeight, 
        velocities
      );
      this.gpuTexture.uploadParticleData(
        this.velocityTextures.textures[1], 
        this.textureWidth, 
        this.textureHeight, 
        velocities
      );
    }
    
    console.log('Particle data initialized');
  }

  step() {
    if (!this.isInitialized) {
      throw new Error('Plan A not initialized. Call init() first.');
    }

    this.performanceMonitor.beginFrame();
    this.runUpdatePass();
    this.time += this.options.dt;
    this.frameCount++;
    this.performanceMonitor.endFrame();
  }

  runUpdatePass() {
    const gl = this.gl;
    
    gl.useProgram(this.updateProgram);
    
    const targetFramebuffer = this.positionTextures.getTargetFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFramebuffer);
    gl.viewport(0, 0, this.textureWidth, this.textureHeight);
    
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.positionTextures.getCurrentTexture());
    gl.uniform1i(this.updateUniforms.u_positions, 0);
    
    if (this.velocityTextures) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.velocityTextures.getCurrentTexture());
      gl.uniform1i(this.updateUniforms.u_velocities, 1);
    }
    
    gl.uniform1f(this.updateUniforms.u_time, this.time);
    gl.uniform1f(this.updateUniforms.u_dt, this.options.dt);
    gl.uniform1i(this.updateUniforms.u_frameCount, this.frameCount);
    gl.uniform1f(this.updateUniforms.u_samplingFraction, this.options.samplingFraction);
    gl.uniform2f(this.updateUniforms.u_texSize, this.textureWidth, this.textureHeight);
    gl.uniform1i(this.updateUniforms.u_particleCount, this.options.particleCount);
    
    const { min, max } = this.options.worldBounds;
    gl.uniform3f(this.updateUniforms.u_worldMin, min[0], min[1], min[2]);
    gl.uniform3f(this.updateUniforms.u_worldMax, max[0], max[1], max[2]);
    
    gl.uniform1ui(this.updateUniforms.u_seed, this.options.seed);
    gl.uniform1i(this.updateUniforms.u_integrationMethod, this.options.integrationMethod === 'euler' ? 0 : 1);
    gl.uniform1i(this.updateUniforms.u_wrapMode, this.options.wrapMode === 'wrap' ? 0 : 1);
    
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    
    this.positionTextures.swap();
    if (this.velocityTextures) {
      this.velocityTextures.swap();
    }
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
  }

  render(projectionViewMatrix) {
    if (!this.isInitialized) return;
    
    const gl = this.gl;
    
    gl.useProgram(this.renderProgram);
    
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.positionTextures.getCurrentTexture());
    gl.uniform1i(this.renderUniforms.u_positions, 0);
    
    gl.uniform2f(this.renderUniforms.u_texSize, this.textureWidth, this.textureHeight);
    gl.uniformMatrix4fv(this.renderUniforms.u_projectionView, false, projectionViewMatrix);
    gl.uniform1f(this.renderUniforms.u_pointSize, this.options.pointSize);
    
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    gl.bindVertexArray(this.particleVAO);
    gl.drawArrays(gl.POINTS, 0, this.options.particleCount);
    
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  resize(newParticleCount) {
    console.log(`Resizing from ${this.options.particleCount} to ${newParticleCount} particles`);
    
    this.gpuTexture.validateParticleCount(newParticleCount);
    
    const oldPositionTextures = this.positionTextures;
    const oldVelocityTextures = this.velocityTextures;
    
    this.options.particleCount = newParticleCount;
    
    this.createTextures();
    this.createGeometry();
    this.initializeParticles();
    
    if (oldPositionTextures) {
      this.cleanupPingPongTextures(oldPositionTextures);
    }
    if (oldVelocityTextures) {
      this.cleanupPingPongTextures(oldVelocityTextures);
    }
  }

  setSamplingFraction(fraction) {
    this.options.samplingFraction = Math.max(0.01, Math.min(1.0, fraction));
  }

  readback(maxCount = 1000) {
    const count = Math.min(maxCount, this.options.particleCount);
    const data = this.gpuTexture.readTextureData(
      this.positionTextures.getCurrentTexture(),
      this.textureWidth,
      this.textureHeight
    );
    
    const particles = [];
    for (let i = 0; i < count; i++) {
      const base = i * 4;
      particles.push({
        x: data[base + 0],
        y: data[base + 1], 
        z: data[base + 2],
        mass: data[base + 3]
      });
    }
    
    return particles;
  }

  getMetrics() {
    const metrics = this.performanceMonitor.getMetrics();
    const memEstimate = this.gpuTexture.estimateMemoryUsage(this.options.particleCount);
    
    return {
      fps: metrics.fps,
      frameTime: metrics.frameTime,
      memoryMB: memEstimate.memoryMB,
      particleCount: this.options.particleCount,
      samplingFraction: this.options.samplingFraction,
      frameCount: this.frameCount
    };
  }

  cleanupPingPongTextures(pingPong) {
    const gl = this.gl;
    
    pingPong.textures.forEach(texture => gl.deleteTexture(texture));
    pingPong.framebuffers.forEach(fbo => gl.deleteFramebuffer(fbo));
  }

  dispose() {
    if (!this.gl) return;
    
    const gl = this.gl;
    
    if (this.positionTextures) {
      this.cleanupPingPongTextures(this.positionTextures);
    }
    if (this.velocityTextures) {
      this.cleanupPingPongTextures(this.velocityTextures);
    }
    
    if (this.updateProgram) gl.deleteProgram(this.updateProgram);
    if (this.renderProgram) gl.deleteProgram(this.renderProgram);
    
    if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
    if (this.particleVAO) gl.deleteVertexArray(this.particleVAO);
    
    if (this.performanceMonitor) {
      this.performanceMonitor.dispose();
    }
    
    if (this.gpuTexture) {
      this.gpuTexture.dispose();
    }
    
    // Clean up Three.js objects
    if (this.threeMesh && this.scene) {
      this.scene.remove(this.threeMesh);
    }
    
    this.isInitialized = false;
    console.log('Plan A disposed');
  }

  // Legacy Three.js integration methods
  start() {
    this.running = true;
    
    // Auto-initialize if not done yet
    if (!this.isInitialized && this.gl) {
      this.init().catch(error => {
        console.error('Plan A initialization failed:', error);
        this.createFallbackPoints();
      });
    } else if (!this.gl && this.scene) {
      // Fallback to Three.js points if no WebGL2
      this.createFallbackPoints();
    }
    
    console.log('Plan A started');
  }

  stop() {
    this.running = false;
    console.log('Plan A stopped');
  }

  update() {
    if (!this.running) return;
    
    if (this.isInitialized) {
      this.step();
      
      // Update Three.js integration if needed
      if (this.renderer && this.scene) {
        this.renderToThreeJS();
      }
    }
  }

  renderToThreeJS() {
    // This method integrates with Three.js rendering
    // For now, we use the direct WebGL rendering approach
    // More advanced integration could create Three.js materials that sample our textures
    
    if (!this.renderer) return;
    
    // Get camera matrices
    const camera = this.renderer.info?.render?.calls > 0 ? 
      (this.scene.children.find(obj => obj.isCamera) || this.renderer.xr?.getCamera()) : null;
    
    if (camera) {
      const projectionMatrix = camera.projectionMatrix;
      const viewMatrix = camera.matrixWorldInverse;
      const projectionView = projectionMatrix.clone().multiply(viewMatrix);
      
      // Save Three.js state
      const oldViewport = this.gl.getParameter(this.gl.VIEWPORT);
      const oldProgram = this.gl.getParameter(this.gl.CURRENT_PROGRAM);
      
      // Render our particles
      this.render(projectionView.elements);
      
      // Restore Three.js state
      this.gl.viewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
      this.gl.useProgram(oldProgram);
    }
  }

  createFallbackPoints() {
    if (!this.scene) return;
    
    console.log('Creating fallback Three.js points');
    
    const geometry = new THREE.BufferGeometry();
    const count = Math.min(10000, this.options.particleCount);
    const positions = new Float32Array(count * 3);
    
    // Generate random positions
    for (let i = 0; i < count; i++) {
      positions[i * 3 + 0] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const material = new THREE.PointsMaterial({ 
      color: 0x66ccff, 
      size: 0.05,
      transparent: true,
      opacity: 0.8
    });
    
    this.threeMesh = new THREE.Points(geometry, material);
    this.scene.add(this.threeMesh);
    this._objects.push(this.threeMesh);
    
    console.log(`Added ${count} fallback particles to scene`);
  }

  // Legacy compatibility methods
  getFrameStats() {
    const metrics = this.getMetrics();
    return {
      mean: metrics.frameTime,
      median: metrics.frameTime,
      last: metrics.frameTime
    };
  }
}
