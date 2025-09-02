import * as THREE from 'three';

/**
 * Plan C - Fragment Shader Spatial Grid with Additive Blending
 * 
 * Reduces neighbor checks from O(N×56) to O(N×8-12) using spatial grid 
 * construction with fragment shaders and additive blending.
 */
export default class PlanC {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this._objects = [];
    
    // Get WebGL2 context
    this.gl = renderer.getContext();
    
    // Configuration
    this.options = {
      particleCount: 100000,
      cellSize: 0.8,
      worldBounds: { min: [-8, -8, -8], max: [8, 8, 8] },
      dt: 0.016,
      pointSize: 4.0
    };
    
    // Grid dimensions
    const worldSize = [
      this.options.worldBounds.max[0] - this.options.worldBounds.min[0],
      this.options.worldBounds.max[1] - this.options.worldBounds.min[1],
      this.options.worldBounds.max[2] - this.options.worldBounds.min[2]
    ];
    
    this.gridDimensions = [
      Math.ceil(worldSize[0] / this.options.cellSize),
      Math.ceil(worldSize[1] / this.options.cellSize),
      Math.ceil(worldSize[2] / this.options.cellSize)
    ];
    
    this.totalCells = this.gridDimensions[0] * this.gridDimensions[1] * this.gridDimensions[2];
    this.gridTextureSize = Math.ceil(Math.sqrt(this.totalCells));
    
    // State
    this.isInitialized = false;
    this.frameCount = 0;
    this.time = 0;
    this.running = false;
    
    // GPU resources
    this.textures = {};
    this.framebuffers = {};
    this.programs = {};
    this.buffers = {};
    this.vaos = {};
  }

  async start() {
    this.running = true;
    
    try {
      await this.initialize();
      console.log('Plan C: Spatial Grid initialized successfully');
    } catch (error) {
      console.error('Plan C initialization failed:', error);
      this.createFallbackVisualization();
    }
  }

  async initialize() {
    if (!this.checkWebGL2Support()) {
      throw new Error('WebGL2 not supported');
    }
    
    this.createTextures();
    this.createShaderPrograms();
    this.createBuffers();
    this.initializeParticles();
    
    this.isInitialized = true;
    console.log(`Plan C initialized with ${this.options.particleCount} particles`);
    console.log(`Grid: ${this.gridDimensions[0]}×${this.gridDimensions[1]}×${this.gridDimensions[2]} = ${this.totalCells} cells`);
  }

  checkWebGL2Support() {
    const gl = this.gl;
    
    // Check for required extensions
    const floatTextures = gl.getExtension('EXT_color_buffer_float');
    const floatBlend = gl.getExtension('EXT_float_blend');
    
    if (!floatTextures || !floatBlend) {
      console.warn('Plan C requires EXT_color_buffer_float and EXT_float_blend');
      return false;
    }
    
    return true;
  }

  createTextures() {
    const gl = this.gl;
    
    // Particle data textures (ping-pong)
    const particleTexSize = Math.ceil(Math.sqrt(this.options.particleCount));
    
    this.textures.positions = this.createPingPongTextures(particleTexSize, particleTexSize);
    this.textures.velocities = this.createPingPongTextures(particleTexSize, particleTexSize);
    
    // Grid textures
    this.textures.gridCounts = this.createTexture(this.gridTextureSize, this.gridTextureSize, gl.RGBA32F);
    this.textures.gridData = this.createTexture(this.gridTextureSize, this.gridTextureSize, gl.RGBA32F);
    
    // Create framebuffers
    this.framebuffers.gridCounts = this.createFramebuffer(this.textures.gridCounts);
    this.framebuffers.gridData = this.createFramebuffer(this.textures.gridData);
    
    this.particleTexSize = particleTexSize;
  }

  createTexture(width, height, format = null) {
    const gl = this.gl;
    const texture = gl.createTexture();
    
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
    const textureFormat = format || gl.RGBA32F;
    gl.texImage2D(gl.TEXTURE_2D, 0, textureFormat, width, height, 0, gl.RGBA, gl.FLOAT, null);
    
    return texture;
  }

  createPingPongTextures(width, height) {
    const gl = this.gl;
    
    const textures = [this.createTexture(width, height), this.createTexture(width, height)];
    const framebuffers = textures.map(tex => this.createFramebuffer(tex));
    
    return {
      textures,
      framebuffers,
      current: 0,
      getCurrentTexture() { return this.textures[this.current]; },
      getTargetTexture() { return this.textures[1 - this.current]; },
      getTargetFramebuffer() { return this.framebuffers[1 - this.current]; },
      swap() { this.current = 1 - this.current; }
    };
  }

  createFramebuffer(texture) {
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer not complete: ${status}`);
    }
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fbo;
  }

  createShaderPrograms() {
    // Particle initialization shader
    this.programs.init = this.createProgram(
      this.getPassVertexShader(),
      this.getInitFragmentShader()
    );
    
    // Grid assignment shader
    this.programs.gridAssign = this.createProgram(
      this.getGridAssignVertexShader(),
      this.getGridAssignFragmentShader()
    );
    
    // Force calculation shader
    this.programs.force = this.createProgram(
      this.getPassVertexShader(),
      this.getForceFragmentShader()
    );
    
    // Position integration shader
    this.programs.integrate = this.createProgram(
      this.getPassVertexShader(),
      this.getIntegrateFragmentShader()
    );
    
    // Particle rendering shader
    this.programs.render = this.createProgram(
      this.getRenderVertexShader(),
      this.getRenderFragmentShader()
    );
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

  createBuffers() {
    const gl = this.gl;
    
    // Full-screen quad for texture passes
    const quadVertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1
    ]);
    
    this.buffers.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.quad);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
    
    // Particle indices for rendering
    const particleIndices = new Float32Array(this.options.particleCount);
    for (let i = 0; i < this.options.particleCount; i++) {
      particleIndices[i] = i;
    }
    
    this.buffers.particles = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.particles);
    gl.bufferData(gl.ARRAY_BUFFER, particleIndices, gl.STATIC_DRAW);
    
    // Create VAOs
    this.createVertexArrayObjects();
  }

  createVertexArrayObjects() {
    const gl = this.gl;
    
    // Quad VAO
    this.vaos.quad = gl.createVertexArray();
    gl.bindVertexArray(this.vaos.quad);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.quad);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    
    // Particle rendering VAO
    this.vaos.particles = gl.createVertexArray();
    gl.bindVertexArray(this.vaos.particles);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.particles);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
    
    gl.bindVertexArray(null);
  }

  initializeParticles() {
    const gl = this.gl;
    
    // Generate initial particle data
    const positions = this.generateParticlePositions();
    const velocities = this.generateParticleVelocities();
    
    // Upload to both ping-pong textures
    this.uploadTextureData(this.textures.positions.textures[0], positions);
    this.uploadTextureData(this.textures.positions.textures[1], positions);
    this.uploadTextureData(this.textures.velocities.textures[0], velocities);
    this.uploadTextureData(this.textures.velocities.textures[1], velocities);
  }

  generateParticlePositions() {
    const data = new Float32Array(this.particleTexSize * this.particleTexSize * 4);
    const bounds = this.options.worldBounds;
    
    for (let i = 0; i < this.options.particleCount; i++) {
      const base = i * 4;
      
      // Create multiple clusters for more interesting visual distribution
      const clusterCount = 5;
      const clusterIndex = Math.floor(Math.random() * clusterCount);
      const clusterRadius = 1.5;
      
      // Cluster centers
      const clusterCenters = [
        [0, 0, 0],
        [3, 2, 1],
        [-2, 3, -1],
        [1, -2, 3],
        [-3, -1, -2]
      ];
      
      const center = clusterCenters[clusterIndex];
      
      // Random position within cluster
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      const r = Math.random() * clusterRadius;
      
      data[base + 0] = center[0] + r * Math.sin(phi) * Math.cos(theta);
      data[base + 1] = center[1] + r * Math.sin(phi) * Math.sin(theta);
      data[base + 2] = center[2] + r * Math.cos(phi);
      data[base + 3] = 1.0; // mass
    }
    
    return data;
  }

  generateParticleVelocities() {
    const data = new Float32Array(this.particleTexSize * this.particleTexSize * 4);
    
    for (let i = 0; i < this.options.particleCount; i++) {
      const base = i * 4;
      
      // Add some initial swirling motion
      const swirl = 0.2;
      data[base + 0] = (Math.random() - 0.5) * 0.5 + swirl * Math.sin(i * 0.01);
      data[base + 1] = (Math.random() - 0.5) * 0.5 + swirl * Math.cos(i * 0.01);
      data[base + 2] = (Math.random() - 0.5) * 0.5;
      data[base + 3] = 0.0; // unused
    }
    
    return data;
  }

  uploadTextureData(texture, data) {
    const gl = this.gl;
    
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.particleTexSize, this.particleTexSize, gl.RGBA, gl.FLOAT, data);
  }

  update() {
    if (!this.running || !this.isInitialized) return;
    
    this.step();
    this.render();
  }

  step() {
    this.clearGrid();
    this.assignParticlesToGrid();
    this.calculateForces();
    this.integrateParticles();
    
    this.time += this.options.dt;
    this.frameCount++;
  }

  clearGrid() {
    const gl = this.gl;
    
    // Clear grid count texture
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.gridCounts);
    gl.viewport(0, 0, this.gridTextureSize, this.gridTextureSize);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  assignParticlesToGrid() {
    const gl = this.gl;
    
    // Use additive blending to count particles per grid cell
    gl.useProgram(this.programs.gridAssign);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.gridCounts);
    gl.viewport(0, 0, this.gridTextureSize, this.gridTextureSize);
    
    // Enable additive blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    
    // Bind particle position texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.positions.getCurrentTexture());
    gl.uniform1i(gl.getUniformLocation(this.programs.gridAssign, 'u_positions'), 0);
    
    // Set uniforms
    this.setGridUniforms(this.programs.gridAssign);
    
    // Render particles as points
    gl.bindVertexArray(this.vaos.particles);
    gl.drawArrays(gl.POINTS, 0, this.options.particleCount);
    
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  calculateForces() {
    const gl = this.gl;
    
    // Calculate forces using spatial grid
    gl.useProgram(this.programs.force);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.textures.velocities.getTargetFramebuffer());
    gl.viewport(0, 0, this.particleTexSize, this.particleTexSize);
    
    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.positions.getCurrentTexture());
    gl.uniform1i(gl.getUniformLocation(this.programs.force, 'u_positions'), 0);
    
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.velocities.getCurrentTexture());
    gl.uniform1i(gl.getUniformLocation(this.programs.force, 'u_velocities'), 1);
    
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.gridCounts);
    gl.uniform1i(gl.getUniformLocation(this.programs.force, 'u_gridCounts'), 2);
    
    // Set uniforms
    this.setGridUniforms(this.programs.force);
    gl.uniform1f(gl.getUniformLocation(this.programs.force, 'u_dt'), this.options.dt);
    gl.uniform1f(gl.getUniformLocation(this.programs.force, 'u_time'), this.time);
    gl.uniform1i(gl.getUniformLocation(this.programs.force, 'u_frameCount'), this.frameCount);
    
    // Render full-screen quad
    gl.bindVertexArray(this.vaos.quad);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    
    this.textures.velocities.swap();
  }

  integrateParticles() {
    const gl = this.gl;
    
    // Integrate positions based on velocities
    gl.useProgram(this.programs.integrate);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.textures.positions.getTargetFramebuffer());
    gl.viewport(0, 0, this.particleTexSize, this.particleTexSize);
    
    // Bind current position and velocity textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.positions.getCurrentTexture());
    gl.uniform1i(gl.getUniformLocation(this.programs.integrate, 'u_positions'), 0);
    
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.velocities.getCurrentTexture());
    gl.uniform1i(gl.getUniformLocation(this.programs.integrate, 'u_velocities'), 1);
    
    // Set uniforms
    gl.uniform1f(gl.getUniformLocation(this.programs.integrate, 'u_dt'), this.options.dt);
    gl.uniform3f(gl.getUniformLocation(this.programs.integrate, 'u_worldMin'), 
                 this.options.worldBounds.min[0], this.options.worldBounds.min[1], this.options.worldBounds.min[2]);
    gl.uniform3f(gl.getUniformLocation(this.programs.integrate, 'u_worldMax'), 
                 this.options.worldBounds.max[0], this.options.worldBounds.max[1], this.options.worldBounds.max[2]);
    gl.uniform2f(gl.getUniformLocation(this.programs.integrate, 'u_particleTexSize'), 
                 this.particleTexSize, this.particleTexSize);
    gl.uniform1i(gl.getUniformLocation(this.programs.integrate, 'u_particleCount'), this.options.particleCount);
    
    // Render full-screen quad
    gl.bindVertexArray(this.vaos.quad);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    
    this.textures.positions.swap();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  render() {
    if (!this.isInitialized) return;
    
    const gl = this.gl;
    
    // Get camera from Three.js scene - try multiple approaches
    let camera = null;
    
    // Try to find camera in scene
    this.scene.traverse((obj) => {
      if (obj.isCamera && !camera) {
        camera = obj;
      }
    });
    
    // Fallback: try to get camera from renderer
    if (!camera && this.renderer.xr) {
      camera = this.renderer.xr.getCamera();
    }
    
    // If still no camera, create a default view
    if (!camera) {
      const projectionView = new THREE.Matrix4();
      projectionView.makePerspective(-1, 1, 1, -1, 0.1, 1000);
      
      const viewMatrix = new THREE.Matrix4();
      viewMatrix.lookAt(new THREE.Vector3(0, 0, 5), new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0));
      
      projectionView.multiply(viewMatrix);
      this.renderParticles(projectionView);
      return;
    }
    
    // Use Three.js camera
    camera.updateMatrixWorld();
    const projectionMatrix = camera.projectionMatrix;
    const viewMatrix = camera.matrixWorldInverse;
    const projectionView = projectionMatrix.clone().multiply(viewMatrix);
    
    this.renderParticles(projectionView);
  }

  renderParticles(projectionView) {
    const gl = this.gl;
    
    // Save current WebGL state
    const oldViewport = gl.getParameter(gl.VIEWPORT);
    const oldProgram = gl.getParameter(gl.CURRENT_PROGRAM);
    const oldBlend = gl.getParameter(gl.BLEND);
    
    // Render particles
    gl.useProgram(this.programs.render);
    
    // Bind position texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.positions.getCurrentTexture());
    gl.uniform1i(gl.getUniformLocation(this.programs.render, 'u_positions'), 0);
    
    // Bind velocity texture
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.textures.velocities.getCurrentTexture());
    gl.uniform1i(gl.getUniformLocation(this.programs.render, 'u_velocities'), 1);
    
    // Set uniforms
    gl.uniform2f(gl.getUniformLocation(this.programs.render, 'u_texSize'), this.particleTexSize, this.particleTexSize);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.programs.render, 'u_projectionView'), false, projectionView.elements);
    gl.uniform1f(gl.getUniformLocation(this.programs.render, 'u_pointSize'), this.options.pointSize);
    gl.uniform1i(gl.getUniformLocation(this.programs.render, 'u_particleCount'), this.options.particleCount);
    gl.uniform1f(gl.getUniformLocation(this.programs.render, 'u_time'), this.time);
    
    // Enable blending for particles
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    // Render particles
    gl.bindVertexArray(this.vaos.particles);
    gl.drawArrays(gl.POINTS, 0, this.options.particleCount);
    
    // Restore WebGL state
    if (!oldBlend) gl.disable(gl.BLEND);
    gl.useProgram(oldProgram);
    gl.viewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
    gl.bindVertexArray(null);
  }

  setGridUniforms(program) {
    const gl = this.gl;
    
    gl.uniform1f(gl.getUniformLocation(program, 'u_cellSize'), this.options.cellSize);
    gl.uniform3f(gl.getUniformLocation(program, 'u_gridDimensions'), 
                 this.gridDimensions[0], this.gridDimensions[1], this.gridDimensions[2]);
    gl.uniform3f(gl.getUniformLocation(program, 'u_worldMin'), 
                 this.options.worldBounds.min[0], this.options.worldBounds.min[1], this.options.worldBounds.min[2]);
    gl.uniform2f(gl.getUniformLocation(program, 'u_gridTextureSize'), this.gridTextureSize, this.gridTextureSize);
    gl.uniform2f(gl.getUniformLocation(program, 'u_particleTexSize'), this.particleTexSize, this.particleTexSize);
    gl.uniform1i(gl.getUniformLocation(program, 'u_particleCount'), this.options.particleCount);
  }

  stop() {
    this.running = false;
    this.cleanup();
  }

  cleanup() {
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
    
    if (!this.gl) return;
    
    const gl = this.gl;
    
    // Clean up GPU resources
    Object.values(this.textures).forEach(resource => {
      if (resource.textures) {
        resource.textures.forEach(tex => gl.deleteTexture(tex));
        resource.framebuffers.forEach(fbo => gl.deleteFramebuffer(fbo));
      } else {
        gl.deleteTexture(resource);
      }
    });
    
    Object.values(this.framebuffers).forEach(fbo => gl.deleteFramebuffer(fbo));
    Object.values(this.programs).forEach(program => gl.deleteProgram(program));
    Object.values(this.buffers).forEach(buffer => gl.deleteBuffer(buffer));
    Object.values(this.vaos).forEach(vao => gl.deleteVertexArray(vao));
  }

  createFallbackVisualization() {
    // Fallback grid visualization when WebGL2 features aren't available
    const group = new THREE.Group();
    const cellSize = this.options.cellSize;
    const dim = Math.floor(this.gridDimensions[0] / 2);
    
    const geo = new THREE.PlaneGeometry(cellSize, cellSize);
    for (let x = -dim; x <= dim; x++) {
      for (let y = -dim; y <= dim; y++) {
        const mat = new THREE.MeshBasicMaterial({ 
          color: 0x223344, 
          side: THREE.DoubleSide, 
          transparent: true, 
          opacity: 0.15 
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x * cellSize, y * cellSize, 0);
        mesh.rotation.x = -Math.PI / 2;
        group.add(mesh);
      }
    }
    
    // Add some animated particles as fallback
    this.createFallbackParticles(group);
    
    this.scene.add(group);
    this._objects.push(group);
  }

  createFallbackParticles(group) {
    const particleCount = Math.min(1000, this.options.particleCount);
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      positions[i * 3 + 0] = (Math.random() - 0.5) * 10;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 10;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const material = new THREE.PointsMaterial({
      color: 0x66ccff,
      size: 0.05,
      transparent: true,
      opacity: 0.8
    });
    
    const particles = new THREE.Points(geometry, material);
    group.add(particles);
    
    // Store for animation
    this.fallbackParticles = particles;
  }

  // Shader source methods (will be implemented next)
  getPassVertexShader() {
    return `#version 300 es
      precision highp float;
      
      in vec2 a_position;
      
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
      }`;
  }

  getInitFragmentShader() {
    return `#version 300 es
      precision highp float;
      out vec4 fragColor;
      
      void main() {
        fragColor = vec4(0.0);
      }`;
  }

  getGridAssignVertexShader() {
    return `#version 300 es
      precision highp float;
      
      in float a_index;
      
      uniform sampler2D u_positions;
      uniform vec2 u_particleTexSize;
      uniform float u_cellSize;
      uniform vec3 u_gridDimensions;
      uniform vec3 u_worldMin;
      uniform vec2 u_gridTextureSize;
      
      void main() {
        // Get particle position
        int x = int(a_index) % int(u_particleTexSize.x);
        int y = int(a_index) / int(u_particleTexSize.x);
        vec2 uv = (vec2(x, y) + 0.5) / u_particleTexSize;
        vec3 pos = texture(u_positions, uv).xyz;
        
        // Calculate grid cell
        vec3 gridPos = (pos - u_worldMin) / u_cellSize;
        ivec3 cellCoord = ivec3(clamp(gridPos, vec3(0.0), u_gridDimensions - 1.0));
        
        // Convert 3D cell to 2D texture coordinate
        int cellId = cellCoord.z * int(u_gridDimensions.x * u_gridDimensions.y) + 
                     cellCoord.y * int(u_gridDimensions.x) + cellCoord.x;
        
        int gridX = cellId % int(u_gridTextureSize.x);
        int gridY = cellId / int(u_gridTextureSize.x);
        
        vec2 gridUV = (vec2(gridX, gridY) + 0.5) / u_gridTextureSize;
        gl_Position = vec4(gridUV * 2.0 - 1.0, 0.0, 1.0);
        gl_PointSize = 1.0;
      }`;
  }

  getGridAssignFragmentShader() {
    return `#version 300 es
      precision highp float;
      out vec4 fragColor;
      
      void main() {
        // Contribute 1 to the cell count (will be additively blended)
        fragColor = vec4(1.0, 0.0, 0.0, 0.0);
      }`;
  }

  getForceFragmentShader() {
    return `#version 300 es
      precision highp float;
      
      uniform sampler2D u_positions;
      uniform sampler2D u_velocities;
      uniform sampler2D u_gridCounts;
      uniform float u_cellSize;
      uniform vec3 u_gridDimensions;
      uniform vec3 u_worldMin;
      uniform vec2 u_gridTextureSize;
      uniform vec2 u_particleTexSize;
      uniform float u_dt;
      uniform float u_time;
      uniform int u_frameCount;
      uniform int u_particleCount;
      
      out vec4 fragColor;
      
      vec3 calculateForce(vec3 pos1, vec3 pos2, float mass1, float mass2) {
        vec3 dir = pos2 - pos1;
        float dist2 = dot(dir, dir) + 0.01; // softening
        float dist = sqrt(dist2);
        
        // Attractive force at medium range, repulsive at close range
        float force;
        if (dist < 0.5) {
          force = -mass1 * mass2 / (dist2 + 0.1); // repulsive
        } else {
          force = mass1 * mass2 / (dist2 + 0.1) * 0.5; // attractive
        }
        
        return force * normalize(dir);
      }
      
      void main() {
        ivec2 coord = ivec2(gl_FragCoord.xy);
        int particleIndex = coord.y * int(u_particleTexSize.x) + coord.x;
        
        if (particleIndex >= u_particleCount) {
          fragColor = vec4(0.0);
          return;
        }
        
        vec2 uv = (vec2(coord) + 0.5) / u_particleTexSize;
        vec3 myPos = texture(u_positions, uv).xyz;
        vec3 myVel = texture(u_velocities, uv).xyz;
        
        // Calculate which grid cell this particle is in
        vec3 gridPos = (myPos - u_worldMin) / u_cellSize;
        ivec3 myCell = ivec3(clamp(gridPos, vec3(0.0), u_gridDimensions - 1.0));
        
        vec3 totalForce = vec3(0.0);
        
        // Add global attractor force (center of simulation)
        vec3 centerForce = -myPos * 0.05;
        totalForce += centerForce;
        
        // Add swirling force
        vec3 swirl = vec3(-myPos.y, myPos.x, 0.0) * 0.1;
        totalForce += swirl;
        
        // Check neighboring cells (3x3x3 = 27 cells)
        for (int dz = -1; dz <= 1; dz++) {
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              ivec3 neighborCell = myCell + ivec3(dx, dy, dz);
              
              // Skip out-of-bounds cells
              if (any(lessThan(neighborCell, ivec3(0))) || 
                  any(greaterThanEqual(neighborCell, ivec3(u_gridDimensions)))) {
                continue;
              }
              
              // Get cell ID and look up particle count
              int cellId = neighborCell.z * int(u_gridDimensions.x * u_gridDimensions.y) + 
                          neighborCell.y * int(u_gridDimensions.x) + neighborCell.x;
              
              int gridX = cellId % int(u_gridTextureSize.x);
              int gridY = cellId / int(u_gridTextureSize.x);
              vec2 gridUV = (vec2(gridX, gridY) + 0.5) / u_gridTextureSize;
              
              float particleCount = texture(u_gridCounts, gridUV).r;
              
              // Apply forces based on particle density in neighboring cells
              if (particleCount > 0.0) {
                vec3 cellCenter = u_worldMin + (vec3(neighborCell) + 0.5) * u_cellSize;
                vec3 force = calculateForce(myPos, cellCenter, 1.0, particleCount);
                totalForce += force * 0.2;
              }
            }
          }
        }
        
        // Apply forces and update velocity
        vec3 acceleration = totalForce;
        vec3 newVel = myVel + acceleration * u_dt;
        
        // Apply damping
        newVel *= 0.99;
        
        // Limit velocity
        float maxVel = 3.0;
        if (length(newVel) > maxVel) {
          newVel = normalize(newVel) * maxVel;
        }
        
        fragColor = vec4(newVel, 0.0);
      }`;
  }

  getRenderVertexShader() {
    return `#version 300 es
      precision highp float;
      
      in float a_index;
      
      uniform sampler2D u_positions;
      uniform sampler2D u_velocities;
      uniform vec2 u_texSize;
      uniform mat4 u_projectionView;
      uniform float u_pointSize;
      uniform int u_particleCount;
      uniform float u_time;
      
      out vec3 v_color;
      out float v_speed;
      
      void main() {
        int particleIndex = int(a_index);
        
        // Skip if beyond particle count
        if (particleIndex >= u_particleCount) {
          gl_Position = vec4(0.0, 0.0, -1.0, 1.0); // Clip this vertex
          return;
        }
        
        int x = particleIndex % int(u_texSize.x);
        int y = particleIndex / int(u_texSize.x);
        vec2 uv = (vec2(x, y) + 0.5) / u_texSize;
        
        vec3 worldPos = texture(u_positions, uv).xyz;
        vec3 velocity = texture(u_velocities, uv).xyz;
        
        gl_Position = u_projectionView * vec4(worldPos, 1.0);
        gl_PointSize = u_pointSize;
        
        // Color based on position and velocity
        float speed = length(velocity);
        v_speed = speed;
        
        // Create rainbow colors based on position and time
        float hue = (worldPos.x + worldPos.y + worldPos.z) * 0.1 + u_time * 0.5;
        vec3 hsv = vec3(mod(hue, 6.28) / 6.28, 0.8, 0.9);
        
        // Convert HSV to RGB
        vec3 c = vec3(hsv.x * 6.0, hsv.y, hsv.z);
        vec3 rgb = hsv.z * mix(vec3(1.0), clamp(abs(mod(c.xxx + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0), hsv.y);
        
        v_color = rgb;
      }`;
  }

  getRenderFragmentShader() {
    return `#version 300 es
      precision mediump float;
      
      in vec3 v_color;
      in float v_speed;
      out vec4 fragColor;
      
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);
        float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
        
        // Brighten based on speed
        float brightness = 0.7 + 0.3 * clamp(v_speed * 2.0, 0.0, 1.0);
        vec3 color = v_color * brightness;
        
        fragColor = vec4(color, alpha * 0.8);
      }`;
  }

  getIntegrateFragmentShader() {
    return `#version 300 es
      precision highp float;
      
      uniform sampler2D u_positions;
      uniform sampler2D u_velocities;
      uniform vec2 u_particleTexSize;
      uniform float u_dt;
      uniform vec3 u_worldMin;
      uniform vec3 u_worldMax;
      uniform int u_particleCount;
      
      out vec4 fragColor;
      
      void main() {
        ivec2 coord = ivec2(gl_FragCoord.xy);
        int particleIndex = coord.y * int(u_particleTexSize.x) + coord.x;
        
        if (particleIndex >= u_particleCount) {
          fragColor = vec4(0.0);
          return;
        }
        
        vec2 uv = (vec2(coord) + 0.5) / u_particleTexSize;
        vec3 pos = texture(u_positions, uv).xyz;
        vec3 vel = texture(u_velocities, uv).xyz;
        float mass = texture(u_positions, uv).w;
        
        // Integrate position
        vec3 newPos = pos + vel * u_dt;
        
        // Apply boundary conditions (wrap around)
        newPos.x = mod(newPos.x - u_worldMin.x, u_worldMax.x - u_worldMin.x) + u_worldMin.x;
        newPos.y = mod(newPos.y - u_worldMin.y, u_worldMax.y - u_worldMin.y) + u_worldMin.y;
        newPos.z = mod(newPos.z - u_worldMin.z, u_worldMax.z - u_worldMin.z) + u_worldMin.z;
        
        fragColor = vec4(newPos, mass);
      }`;
  }
}
