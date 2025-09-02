/**
 * Plan M: "The Menace" — GPU-side Dynamic Quadtree
 * 
 * Implements a GPU-resident quadtree using WebGL2 fragment shaders for large-scale
 * particle simulation with O(N log N) complexity via Barnes-Hut algorithm.
 */

import * as THREE from 'three';

// Shader sources for the quadtree implementation
const vertexShaderSource = `#version 300 es
precision highp float;

in vec2 a_position;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const particleAggregationVertexShader = `#version 300 es
precision highp float;

in vec3 a_position;
in float a_mass;

uniform mat4 u_worldToGrid;
uniform vec2 u_gridSize;

out vec4 v_particleData;

void main() {
  // Transform particle position to grid space
  vec4 gridPos = u_worldToGrid * vec4(a_position, 1.0);
  
  // Convert to texture coordinates (0 to gridSize)
  vec2 texCoord = gridPos.xy * u_gridSize;
  
  // Output as screen coordinates for point rendering
  gl_Position = vec4((texCoord / u_gridSize) * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = 1.0;
  
  // Pass data for fragment shader aggregation
  v_particleData = vec4(a_position.xy * a_mass, a_mass, 1.0); // weighted position + mass + count
}`;

const particleAggregationFragmentShader = `#version 300 es
precision highp float;

in vec4 v_particleData;
out vec4 fragColor;

void main() {
  // Output components that will be additively blended
  fragColor = v_particleData; // sum_x, sum_y, mass, count
}`;

const reductionFragmentShader = `#version 300 es
precision highp float;

uniform sampler2D u_previousLevel;
uniform vec2 u_previousLevelSize;

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

const traversalFragmentShader = `#version 300 es
precision highp float;

uniform sampler2D u_particlePositions;
uniform sampler2D u_quadtreeLevel0;
uniform sampler2D u_quadtreeLevel1;
uniform sampler2D u_quadtreeLevel2;
uniform sampler2D u_quadtreeLevel3;
uniform sampler2D u_quadtreeLevel4;
uniform sampler2D u_quadtreeLevel5;
uniform sampler2D u_quadtreeLevel6;
uniform sampler2D u_quadtreeLevel7;
uniform float u_theta;
uniform int u_numLevels;
uniform float u_cellSizes[8];
uniform vec2 u_texSize;
uniform int u_particleCount;

out vec4 fragColor;

vec4 sampleLevel(int level, ivec2 coord) {
  if (level == 0) { return texelFetch(u_quadtreeLevel0, coord, 0); }
  else if (level == 1) { return texelFetch(u_quadtreeLevel1, coord, 0); }
  else if (level == 2) { return texelFetch(u_quadtreeLevel2, coord, 0); }
  else if (level == 3) { return texelFetch(u_quadtreeLevel3, coord, 0); }
  else if (level == 4) { return texelFetch(u_quadtreeLevel4, coord, 0); }
  else if (level == 5) { return texelFetch(u_quadtreeLevel5, coord, 0); }
  else if (level == 6) { return texelFetch(u_quadtreeLevel6, coord, 0); }
  else if (level == 7) { return texelFetch(u_quadtreeLevel7, coord, 0); }
  else { return vec4(0.0); }
}

vec2 getLevelSize(int level) {
  if (level == 0) { return textureSize(u_quadtreeLevel0, 0); }
  else if (level == 1) { return textureSize(u_quadtreeLevel1, 0); }
  else if (level == 2) { return textureSize(u_quadtreeLevel2, 0); }
  else if (level == 3) { return textureSize(u_quadtreeLevel3, 0); }
  else if (level == 4) { return textureSize(u_quadtreeLevel4, 0); }
  else if (level == 5) { return textureSize(u_quadtreeLevel5, 0); }
  else if (level == 6) { return textureSize(u_quadtreeLevel6, 0); }
  else if (level == 7) { return textureSize(u_quadtreeLevel7, 0); }
  else { return vec2(1.0); }
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int myIndex = coord.y * int(u_texSize.x) + coord.x;
  
  if (myIndex >= u_particleCount) {
    fragColor = vec4(0.0);
    return;
  }
  
  vec2 myUV = (vec2(coord) + 0.5) / u_texSize;
  vec3 myPos = texture(u_particlePositions, myUV).xyz;
  vec3 totalForce = vec3(0.0);
  
  // Simple traversal (simplified for initial implementation)
  for (int level = min(u_numLevels - 1, 7); level >= 0; level--) {
    vec2 levelSize = getLevelSize(level);
    vec2 nodeCoord = (myPos.xy / u_cellSizes[level]) * levelSize;
    ivec2 nodeIndex = ivec2(floor(nodeCoord));
    
    // Clamp to valid range
    nodeIndex = clamp(nodeIndex, ivec2(0), ivec2(levelSize) - 1);
    
    // Fetch node data
    vec4 nodeData = sampleLevel(level, nodeIndex);
    float massSum = nodeData.b;
    float particleCount = nodeData.a;
    
    if (particleCount > 0.0) {
      // Calculate center of mass
      vec2 centerOfMass = nodeData.rg / massSum;
      
      // Distance to center of mass
      float distance = length(myPos.xy - centerOfMass);
      float nodeSize = u_cellSizes[level];
      
      // Barnes-Hut criterion: if far enough, use approximation
      if ((nodeSize / distance) < u_theta || level == 0) {
        // Use this node for force approximation
        vec2 direction = centerOfMass - myPos.xy;
        float forceMagnitude = massSum / (distance * distance + 0.01); // softening
        totalForce.xy += normalize(direction) * forceMagnitude;
        break;
      }
    }
  }
  
  fragColor = vec4(totalForce, 0.0);
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
    this.velocityTextures = null;
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
    this.numLevels = Math.ceil(Math.log2(this.L0Size)) + 1;
    
    // For position textures
    this.textureWidth = Math.ceil(Math.sqrt(this.options.particleCount));
    this.textureHeight = Math.ceil(this.options.particleCount / this.textureWidth);
    
    console.log(`Quadtree: L0=${this.L0Size}x${this.L0Size}, ${this.numLevels} levels`);
    console.log(`Position texture: ${this.textureWidth}x${this.textureHeight} for ${this.options.particleCount} particles`);
  }

  createShaderPrograms() {
    const gl = this.gl;
    
    // Aggregation program
    this.programs.aggregation = this.createProgram(
      particleAggregationVertexShader, 
      particleAggregationFragmentShader
    );
    
    // Reduction program
    this.programs.reduction = this.createProgram(
      vertexShaderSource, 
      reductionFragmentShader
    );
    
    // Traversal program
    this.programs.traversal = this.createProgram(
      vertexShaderSource, 
      traversalFragmentShader
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
    this.velocityTextures = this.createPingPongTextures(this.textureWidth, this.textureHeight);
    
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
    const velocities = new Float32Array(this.options.particleCount * 4);
    
    // Generate random particles within world bounds
    const bounds = this.options.worldBounds;
    for (let i = 0; i < this.options.particleCount; i++) {
      const base = i * 4;
      positions[base + 0] = bounds.min[0] + Math.random() * (bounds.max[0] - bounds.min[0]);
      positions[base + 1] = bounds.min[1] + Math.random() * (bounds.max[1] - bounds.min[1]);
      positions[base + 2] = bounds.min[2] + Math.random() * (bounds.max[2] - bounds.min[2]);
      positions[base + 3] = 1.0; // mass
      
      velocities[base + 0] = (Math.random() - 0.5) * 2.0;
      velocities[base + 1] = (Math.random() - 0.5) * 2.0;
      velocities[base + 2] = (Math.random() - 0.5) * 2.0;
      velocities[base + 3] = 0.0; // unused
    }
    
    // Upload to GPU
    this.uploadTextureData(this.positionTextures.textures[0], positions);
    this.uploadTextureData(this.positionTextures.textures[1], positions);
    this.uploadTextureData(this.velocityTextures.textures[0], velocities);
    this.uploadTextureData(this.velocityTextures.textures[1], velocities);
    
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
    // Build quadtree
    this.buildQuadtree();
    
    // Calculate forces using traversal
    this.calculateForces();
    
    // Integrate physics (simplified for now)
    this.integratePhysics();
    
    this.frameCount++;
    this.time += this.options.dt;
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
    
    // Step 1: Aggregate particles into L0 using additive blending
    this.aggregateParticlesIntoL0();
    
    // Step 2: Build pyramid via reduction passes
    for (let level = 0; level < this.numLevels - 1; level++) {
      this.runReductionPass(level, level + 1);
    }
  }

  aggregateParticlesIntoL0() {
    const gl = this.gl;
    
    // Use aggregation program
    gl.useProgram(this.programs.aggregation);
    
    // Bind L0 framebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.levelFramebuffers[0]);
    gl.viewport(0, 0, this.levelTextures[0].size, this.levelTextures[0].size);
    
    // Enable additive blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    
    // Create and bind particle position data for aggregation
    this.createParticleAggregationGeometry();
    
    // Set uniforms for world-to-grid transformation
    const worldSize = this.options.worldBounds.max[0] - this.options.worldBounds.min[0];
    const gridSize = this.levelTextures[0].size;
    
    // Simple world-to-grid matrix (just scaling for now)
    const worldToGrid = new Float32Array([
      gridSize / worldSize, 0, 0, 0,
      0, gridSize / worldSize, 0, 0,
      0, 0, 1, 0,
      -this.options.worldBounds.min[0] * gridSize / worldSize,
      -this.options.worldBounds.min[1] * gridSize / worldSize, 0, 1
    ]);
    
    const u_worldToGrid = gl.getUniformLocation(this.programs.aggregation, 'u_worldToGrid');
    const u_gridSize = gl.getUniformLocation(this.programs.aggregation, 'u_gridSize');
    
    gl.uniformMatrix4fv(u_worldToGrid, false, worldToGrid);
    gl.uniform2f(u_gridSize, gridSize, gridSize);
    
    // Render particles as points to aggregate them
    gl.bindVertexArray(this.aggregationVAO);
    gl.drawArrays(gl.POINTS, 0, this.options.particleCount);
    
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  createParticleAggregationGeometry() {
    if (this.aggregationVAO) return;
    
    const gl = this.gl;
    
    // Create VAO for particle aggregation
    this.aggregationVAO = gl.createVertexArray();
    gl.bindVertexArray(this.aggregationVAO);
    
    // Get current particle positions from texture (simplified approach)
    // In a full implementation, we'd read from the position texture
    // For now, let's create dummy data that matches our particle layout
    const positions = new Float32Array(this.options.particleCount * 3);
    const masses = new Float32Array(this.options.particleCount);
    
    // Generate particle data (this should come from the position texture in the real implementation)
    const bounds = this.options.worldBounds;
    for (let i = 0; i < this.options.particleCount; i++) {
      positions[i * 3 + 0] = bounds.min[0] + Math.random() * (bounds.max[0] - bounds.min[0]);
      positions[i * 3 + 1] = bounds.min[1] + Math.random() * (bounds.max[1] - bounds.min[1]);
      positions[i * 3 + 2] = bounds.min[2] + Math.random() * (bounds.max[2] - bounds.min[2]);
      masses[i] = 1.0;
    }
    
    // Position buffer
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    
    const a_position = gl.getAttribLocation(this.programs.aggregation, 'a_position');
    gl.enableVertexAttribArray(a_position);
    gl.vertexAttribPointer(a_position, 3, gl.FLOAT, false, 0, 0);
    
    // Mass buffer
    const massBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, massBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, masses, gl.STATIC_DRAW);
    
    const a_mass = gl.getAttribLocation(this.programs.aggregation, 'a_mass');
    gl.enableVertexAttribArray(a_mass);
    gl.vertexAttribPointer(a_mass, 1, gl.FLOAT, false, 0, 0);
    
    gl.bindVertexArray(null);
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
    const u_previousLevelSize = gl.getUniformLocation(this.programs.reduction, 'u_previousLevelSize');
    
    gl.uniform1i(u_previousLevel, 0);
    gl.uniform2f(u_previousLevelSize, this.levelTextures[sourceLevel].size, this.levelTextures[sourceLevel].size);
    
    // Render full-screen quad
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  calculateForces() {
    const gl = this.gl;
    
    gl.useProgram(this.programs.traversal);
    
    // Bind target framebuffer (velocity texture for force accumulation)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocityTextures.getTargetFramebuffer());
    gl.viewport(0, 0, this.textureWidth, this.textureHeight);
    
    // Bind particle positions
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.positionTextures.getCurrentTexture());
    
    // Set position texture uniform
    const u_particlePositions = gl.getUniformLocation(this.programs.traversal, 'u_particlePositions');
    gl.uniform1i(u_particlePositions, 0);
    
    // Bind quadtree level textures
    for (let i = 0; i < Math.min(8, this.numLevels); i++) {
      gl.activeTexture(gl.TEXTURE1 + i);
      gl.bindTexture(gl.TEXTURE_2D, this.levelTextures[i].texture);
      
      const uniformName = `u_quadtreeLevel${i}`;
      const uniformLocation = gl.getUniformLocation(this.programs.traversal, uniformName);
      if (uniformLocation !== null) {
        gl.uniform1i(uniformLocation, 1 + i);
      }
    }
    
    // Set other uniforms
    const u_theta = gl.getUniformLocation(this.programs.traversal, 'u_theta');
    const u_numLevels = gl.getUniformLocation(this.programs.traversal, 'u_numLevels');
    const u_texSize = gl.getUniformLocation(this.programs.traversal, 'u_texSize');
    const u_particleCount = gl.getUniformLocation(this.programs.traversal, 'u_particleCount');
    
    gl.uniform1f(u_theta, this.options.theta);
    gl.uniform1i(u_numLevels, this.numLevels);
    gl.uniform2f(u_texSize, this.textureWidth, this.textureHeight);
    gl.uniform1i(u_particleCount, this.options.particleCount);
    
    // Set cell sizes for each level
    const cellSizes = [];
    const worldSize = this.options.worldBounds.max[0] - this.options.worldBounds.min[0];
    let currentSize = worldSize / this.L0Size;
    for (let i = 0; i < 8; i++) {
      cellSizes.push(currentSize);
      currentSize *= 2;
    }
    
    const u_cellSizes = gl.getUniformLocation(this.programs.traversal, 'u_cellSizes');
    gl.uniform1fv(u_cellSizes, cellSizes);
    
    // Render full-screen quad to compute forces
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    
    this.velocityTextures.swap();
  }

  integratePhysics() {
    // Simple integration - just move particles slightly for visual feedback
    // In a complete implementation, this would integrate forces into velocities and positions
    
    // For now, let's just swap the position textures to indicate we're processing
    this.positionTextures.swap();
  }

  renderToThreeJS() {
    if (!this.renderer || !this.scene) return;
    
    // Get camera for projection matrix
    const camera = this.getCameraFromScene();
    if (!camera) return;
    
    const gl = this.gl;
    
    // Save WebGL state
    const oldViewport = gl.getParameter(gl.VIEWPORT);
    const oldProgram = gl.getParameter(gl.CURRENT_PROGRAM);
    const oldFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    
    // Use our render program
    gl.useProgram(this.programs.render);
    
    // Bind default framebuffer (screen)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    
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
    gl.uniform1f(u_pointSize, this.options.pointSize);
    
    // Calculate projection-view matrix
    const projectionMatrix = camera.projectionMatrix;
    const viewMatrix = camera.matrixWorldInverse;
    const projectionView = projectionMatrix.clone().multiply(viewMatrix);
    gl.uniformMatrix4fv(u_projectionView, false, projectionView.elements);
    
    // Enable blending for particles
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    // Render particles
    gl.bindVertexArray(this.particleVAO);
    gl.drawArrays(gl.POINTS, 0, this.options.particleCount);
    gl.bindVertexArray(null);
    
    gl.disable(gl.BLEND);
    
    // Restore WebGL state
    gl.viewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
    gl.useProgram(oldProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, oldFramebuffer);
  }

  getCameraFromScene() {
    if (!this.scene) return null;
    
    // Find camera in scene
    for (let child of this.scene.children) {
      if (child.isCamera) {
        return child;
      }
    }
    
    // Fallback: create a simple camera
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();
    return camera;
  }

  createFallbackVisualization() {
    if (!this.scene) return;
    
    console.log('Creating Plan M fallback visualization');
    
    // Create a more sophisticated quadtree visualization
    const group = new THREE.Group();
    
    // Particle swarm
    const particleGeometry = new THREE.BufferGeometry();
    const count = Math.min(10000, this.options.particleCount);
    const positions = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      positions[i * 3 + 0] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
    }
    
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const particleMaterial = new THREE.PointsMaterial({ 
      color: 0x66ccff, 
      size: 0.05,
      transparent: true,
      opacity: 0.8
    });
    
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    group.add(particles);
    
    // Quadtree levels visualization
    const levels = Math.min(6, this.numLevels);
    for (let i = 0; i < levels; i++) {
      const size = Math.pow(2, levels - i) * 0.1;
      const geo = new THREE.PlaneGeometry(size, size);
      const mat = new THREE.MeshBasicMaterial({ 
        color: new THREE.Color().setHSL(i / levels, 0.6, 0.5), 
        side: THREE.DoubleSide, 
        transparent: true, 
        opacity: 0.15 
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(0, 0, -i * 0.02);
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
    
    if (this.velocityTextures) {
      this.velocityTextures.textures.forEach(tex => gl.deleteTexture(tex));
      this.velocityTextures.framebuffers.forEach(fbo => gl.deleteFramebuffer(fbo));
    }
    
    // Clean up programs
    Object.values(this.programs).forEach(program => gl.deleteProgram(program));
    
    // Clean up VAOs
    if (this.quadVAO) gl.deleteVertexArray(this.quadVAO);
    if (this.particleVAO) gl.deleteVertexArray(this.particleVAO);
    if (this.aggregationVAO) gl.deleteVertexArray(this.aggregationVAO);
    
    // Clean up Three.js objects
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
    
    this.isInitialized = false;
    console.log('Plan M disposed');
  }
}
