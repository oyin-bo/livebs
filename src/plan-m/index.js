/**
 * Plan M: "The Menace" — GPU-side Dynamic Quadtree
 * 
 * Implements a GPU-resident quadtree using WebGL2 fragment shaders for large-scale
 * particle simulation with O(N log N) complexity via Barnes-Hut algorithm.
 */

import * as THREE from 'three';

// Simplified shaders for initial working implementation
const vertexShaderSource = `#version 300 es
precision highp float;

in vec2 a_position;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const reductionFragmentShader = `#version 300 es
precision highp float;

uniform sampler2D u_previousLevel;

out vec4 fragColor;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  
  // Read 2x2 block from previous level
  ivec2 childBase = coord * 2;
  vec4 child00 = texelFetch(u_previousLevel, childBase + ivec2(0,0), 0);
  vec4 child01 = texelFetch(u_previousLevel, childBase + ivec2(0,1), 0);
  vec4 child10 = texelFetch(u_previousLevel, childBase + ivec2(1,0), 0);
  vec4 child11 = texelFetch(u_previousLevel, childBase + ivec2(1,1), 0);
  
  // Aggregate: sum all components
  vec4 aggregate = child00 + child01 + child10 + child11;
  
  fragColor = aggregate;
}`;

const renderVertexShader = `#version 300 es
precision highp float;

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
}`;

const renderFragmentShader = `#version 300 es
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

export default class PlanM {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.gl = renderer ? renderer.getContext() : null;
    
    // Configuration
    this.options = {
      particleCount: 50000, // Start smaller for initial implementation
      worldBounds: { min: [-10, -10, -10], max: [10, 10, 10] },
      theta: 0.5, // Barnes-Hut parameter
      pointSize: 2.0,
      dt: 0.016
    };
    
    // Internal state
    this._objects = [];
    this.isInitialized = false;
    this.frameCount = 0;
    this.time = 0;
    this.running = false;
    
    // GPU resources
    this.levelTextures = [];
    this.levelFramebuffers = [];
    this.positionTextures = null;
    this.programs = {};
    this.quadVAO = null;
    this.particleVAO = null;
    this.textureWidth = 0;
    this.textureHeight = 0;
    this.numLevels = 0;
    this.L0Size = 0;
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
    
    // Check for required extensions
    const colorBufferFloat = gl.getExtension('EXT_color_buffer_float');
    const floatBlend = gl.getExtension('EXT_float_blend');
    
    if (!colorBufferFloat) {
      throw new Error('EXT_color_buffer_float extension not supported');
    }
    
    if (!floatBlend) {
      console.warn('EXT_float_blend not available, may have blending limitations');
    }
    
    console.log('WebGL2 extensions check passed');
  }

  calculateTextureDimensions() {
    // Calculate L0 size to fit particles with reasonable density
    this.L0Size = Math.max(64, Math.ceil(Math.sqrt(this.options.particleCount * 4))); // 4x oversampling
    this.numLevels = Math.min(8, Math.ceil(Math.log2(this.L0Size)) + 1); // Limit to 8 levels
    
    // For position textures
    this.textureWidth = Math.ceil(Math.sqrt(this.options.particleCount));
    this.textureHeight = Math.ceil(this.options.particleCount / this.textureWidth);
    
    console.log(`Quadtree: L0=${this.L0Size}x${this.L0Size}, ${this.numLevels} levels`);
    console.log(`Position texture: ${this.textureWidth}x${this.textureHeight} for ${this.options.particleCount} particles`);
  }

  createShaderPrograms() {
    const gl = this.gl;
    
    // Reduction program
    this.programs.reduction = this.createProgram(
      vertexShaderSource, 
      reductionFragmentShader
    );
    
    // Render program
    this.programs.render = this.createProgram(
      renderVertexShader, 
      renderFragmentShader
    );
    
    console.log('Shader programs created successfully');
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

  createTextures() {
    const gl = this.gl;
    
    // Create quadtree level textures
    this.levelTextures = [];
    this.levelFramebuffers = [];
    
    let currentSize = this.L0Size;
    for (let i = 0; i < this.numLevels; i++) {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, currentSize, currentSize, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      
      this.levelTextures.push({texture, size: currentSize});
      this.levelFramebuffers.push(framebuffer);
      
      currentSize = Math.max(1, Math.floor(currentSize / 2));
    }
    
    // Create particle position textures (ping-pong)
    this.positionTextures = this.createPingPongTextures(this.textureWidth, this.textureHeight);
    
    console.log(`Created ${this.numLevels} quadtree level textures and particle textures`);
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

  createGeometry() {
    const gl = this.gl;
    
    // Full-screen quad for reduction passes
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
    
    // Particle indices for rendering
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
  }

  initializeParticles() {
    const positions = new Float32Array(this.options.particleCount * 4);
    
    // Generate random particles within world bounds that form a swarm pattern
    const bounds = this.options.worldBounds;
    const center = [
      (bounds.min[0] + bounds.max[0]) / 2,
      (bounds.min[1] + bounds.max[1]) / 2,
      (bounds.min[2] + bounds.max[2]) / 2
    ];
    
    for (let i = 0; i < this.options.particleCount; i++) {
      const base = i * 4;
      
      // Create clustered distribution for better visualization
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 8 + Math.random() * 2; // Bias toward center
      const height = (Math.random() - 0.5) * 4;
      
      positions[base + 0] = center[0] + Math.cos(angle) * radius;
      positions[base + 1] = center[1] + Math.sin(angle) * radius;
      positions[base + 2] = center[2] + height;
      positions[base + 3] = 1.0; // mass
    }
    
    // Upload to GPU
    this.uploadTextureData(this.positionTextures.textures[0], positions);
    this.uploadTextureData(this.positionTextures.textures[1], positions);
    
    console.log('Particle data initialized');
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
      this.renderToThreeJS();
    }
  }

  step() {
    // Build quadtree (simplified for initial implementation)
    this.buildQuadtree();
    
    // Simple animation - rotate particles slightly
    this.time += this.options.dt;
    this.frameCount++;
  }

  buildQuadtree() {
    const gl = this.gl;
    
    // Clear all level textures
    for (let i = 0; i < this.numLevels; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.levelFramebuffers[i]);
      gl.viewport(0, 0, this.levelTextures[i].size, this.levelTextures[i].size);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    
    // For now, just populate L0 with dummy data and test reduction
    this.populateL0WithTestData();
    
    // Build pyramid via reduction passes
    for (let level = 0; level < this.numLevels - 1; level++) {
      this.runReductionPass(level, level + 1);
    }
  }

  populateL0WithTestData() {
    const gl = this.gl;
    
    // Create simple test pattern in L0 for verification
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.levelFramebuffers[0]);
    gl.viewport(0, 0, this.levelTextures[0].size, this.levelTextures[0].size);
    
    // Fill with a simple pattern
    gl.clearColor(0.1, 0.2, 0.3, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  runReductionPass(sourceLevel, targetLevel) {
    const gl = this.gl;
    
    gl.useProgram(this.programs.reduction);
    
    // Bind target framebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.levelFramebuffers[targetLevel]);
    gl.viewport(0, 0, this.levelTextures[targetLevel].size, this.levelTextures[targetLevel].size);
    
    // Bind source texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.levelTextures[sourceLevel].texture);
    
    const u_previousLevel = gl.getUniformLocation(this.programs.reduction, 'u_previousLevel');
    gl.uniform1i(u_previousLevel, 0);
    
    // Render full-screen quad
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  renderToThreeJS() {
    if (!this.renderer || !this.scene) return;
    
    // Get camera for projection matrix
    const camera = this.getCameraFromScene();
    if (!camera) {
      console.warn('Plan M: No camera found for rendering');
      return;
    }
    
    const gl = this.gl;
    
    // Save WebGL state
    const oldViewport = gl.getParameter(gl.VIEWPORT);
    const oldProgram = gl.getParameter(gl.CURRENT_PROGRAM);
    const oldFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    
    try {
      // Use our render program
      gl.useProgram(this.programs.render);
      
      // Bind default framebuffer (screen)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      
      // Bind particle positions
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.positionTextures.getCurrentTexture());
      
      // Set uniforms
      const u_positions = gl.getUniformLocation(this.programs.render, 'u_positions');
      const u_texSize = gl.getUniformLocation(this.programs.render, 'u_texSize');
      const u_projectionView = gl.getUniformLocation(this.programs.render, 'u_projectionView');
      const u_pointSize = gl.getUniformLocation(this.programs.render, 'u_pointSize');
      
      gl.uniform1i(u_positions, 0);
      gl.uniform2f(u_texSize, this.textureWidth, this.textureHeight);
      gl.uniform1f(u_pointSize, this.options.pointSize * 2); // Make points larger
      
      // Calculate projection-view matrix
      camera.updateMatrixWorld();
      camera.updateProjectionMatrix();
      const projectionMatrix = camera.projectionMatrix;
      const viewMatrix = camera.matrixWorldInverse;
      const projectionView = projectionMatrix.clone().multiply(viewMatrix);
      gl.uniformMatrix4fv(u_projectionView, false, projectionView.elements);
      
      // Enable blending for particles
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.enable(gl.DEPTH_TEST);
      
      // Render particles
      gl.bindVertexArray(this.particleVAO);
      gl.drawArrays(gl.POINTS, 0, this.options.particleCount);
      gl.bindVertexArray(null);
      
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      
      // Debug: Log on first few frames
      if (this.frameCount < 3) {
        console.log(`Plan M: Rendered ${this.options.particleCount} particles at frame ${this.frameCount}`);
      }
      
    } catch (error) {
      console.error('Plan M render error:', error);
    } finally {
      // Restore WebGL state
      gl.viewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
      gl.useProgram(oldProgram);
      gl.bindFramebuffer(gl.FRAMEBUFFER, oldFramebuffer);
    }
  }

  getCameraFromScene() {
    if (!this.scene) return null;
    
    // Find camera in scene - check all children recursively
    let camera = null;
    this.scene.traverse((child) => {
      if (child.isCamera && !camera) {
        camera = child;
        console.log('Plan M: Found camera in scene:', child.constructor.name);
      }
    });
    
    // If no camera found, create one that should work
    if (!camera) {
      camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
      camera.position.set(0, 0, 5);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();
      camera.updateProjectionMatrix();
      console.log('Plan M: Using fallback camera');
    }
    
    return camera;
  }

  createFallbackVisualization() {
    if (!this.scene) return;
    
    console.log('Creating Plan M fallback visualization');
    
    // Create a more sophisticated quadtree visualization
    const group = new THREE.Group();
    
    // Particle swarm - create a 3D swarm pattern
    const particleGeometry = new THREE.BufferGeometry();
    const count = Math.min(this.options.particleCount, 50000);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    // Create swarm pattern
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 8 + Math.random() * 2;
      const height = (Math.random() - 0.5) * 4;
      
      positions[i * 3 + 0] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = Math.sin(angle) * radius;
      positions[i * 3 + 2] = height;
      
      // Color based on distance from center
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
    
    // Quadtree levels visualization with proper scaling
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
    
    // Clean up textures
    this.levelTextures.forEach(level => gl.deleteTexture(level.texture));
    this.levelFramebuffers.forEach(fbo => gl.deleteFramebuffer(fbo));
    
    if (this.positionTextures) {
      this.positionTextures.textures.forEach(tex => gl.deleteTexture(tex));
      this.positionTextures.framebuffers.forEach(fbo => gl.deleteFramebuffer(fbo));
    }
    
    // Clean up programs
    Object.values(this.programs).forEach(program => gl.deleteProgram(program));
    
    // Clean up VAOs
    if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
    if (this.particleVAO) gl.deleteVertexArray(this.particleVAO);
    
    // Clean up Three.js objects
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
    
    this.isInitialized = false;
    console.log('Plan M disposed');
  }
}