/**
 * GPU Texture and Framebuffer utilities for Plan A
 * Handles capability detection, texture creation, and FBO management
 */

export class GPUTexture {
  constructor(gl) {
    this.gl = gl;
    this.capabilities = this.detectCapabilities();
  }

  /**
   * Detect WebGL capabilities for texture formats and extensions
   */
  detectCapabilities() {
    const gl = this.gl;
    
    const capabilities = {
      webgl2: !!gl.getParameter,
      float_texture: !!gl.getExtension('EXT_color_buffer_float'),
      float_blend: !!gl.getExtension('EXT_float_blend'),
      max_texture_size: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      timer_query: gl.getExtension('EXT_disjoint_timer_query_webgl2')
    };

    // Decision matrix for texture format
    if (capabilities.float_texture && capabilities.float_blend) {
      capabilities.textureFormat = gl.RGBA32F;
      capabilities.textureType = gl.FLOAT;
      capabilities.precision = 'high';
    } else {
      capabilities.textureFormat = gl.RGBA;
      capabilities.textureType = gl.UNSIGNED_BYTE;
      capabilities.precision = 'medium';
    }

    return capabilities;
  }

  /**
   * Validate particle count against texture size limits
   */
  validateParticleCount(particleCount) {
    const requiredSize = Math.ceil(Math.sqrt(particleCount));
    if (requiredSize > this.capabilities.max_texture_size) {
      throw new Error(`Particle count ${particleCount} requires texture size ${requiredSize}, but max is ${this.capabilities.max_texture_size}`);
    }
    return requiredSize;
  }

  /**
   * Calculate texture dimensions for given particle count
   */
  getTextureDimensions(particleCount) {
    const size = Math.ceil(Math.sqrt(particleCount));
    return { width: size, height: size };
  }

  /**
   * Create a texture with specified format and dimensions
   */
  createTexture(width, height, format = null, type = null) {
    const gl = this.gl;
    const texture = gl.createTexture();
    
    const actualFormat = format || this.capabilities.textureFormat;
    const actualType = type || this.capabilities.textureType;

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, actualFormat,
      width, height, 0, 
      gl.RGBA, actualType, null
    );

    // Set texture parameters
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  /**
   * Create a framebuffer with attached texture
   */
  createFramebuffer(texture) {
    const gl = this.gl;
    const framebuffer = gl.createFramebuffer();
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, texture, 0
    );

    // Check framebuffer completeness
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      gl.deleteFramebuffer(framebuffer);
      throw new Error(`Framebuffer incomplete: ${this.getFramebufferStatusString(status)}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return framebuffer;
  }

  /**
   * Create ping-pong texture pair with framebuffers
   */
  createPingPongTextures(width, height) {
    const texture1 = this.createTexture(width, height);
    const texture2 = this.createTexture(width, height);
    
    const fbo1 = this.createFramebuffer(texture1);
    const fbo2 = this.createFramebuffer(texture2);

    return {
      textures: [texture1, texture2],
      framebuffers: [fbo1, fbo2],
      current: 0,
      
      // Helper methods
      getCurrentTexture() { return this.textures[this.current]; },
      getCurrentFramebuffer() { return this.framebuffers[this.current]; },
      getTargetTexture() { return this.textures[1 - this.current]; },
      getTargetFramebuffer() { return this.framebuffers[1 - this.current]; },
      swap() { this.current = 1 - this.current; }
    };
  }

  /**
   * Upload initial particle data to texture
   */
  uploadParticleData(texture, width, height, data) {
    const gl = this.gl;
    
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texSubImage2D(
      gl.TEXTURE_2D, 0, 0, 0, width, height,
      gl.RGBA, this.capabilities.textureType, data
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /**
   * Read texture data back to CPU (for testing/debugging)
   */
  readTextureData(texture, width, height) {
    const gl = this.gl;
    const framebuffer = this.createFramebuffer(texture);
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    
    const format = this.capabilities.precision === 'high' ? Float32Array : Uint8Array;
    const data = new format(width * height * 4);
    
    gl.readPixels(0, 0, width, height, gl.RGBA, this.capabilities.textureType, data);
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(framebuffer);
    
    return data;
  }

  /**
   * Convert framebuffer status to readable string
   */
  getFramebufferStatusString(status) {
    const gl = this.gl;
    switch (status) {
      case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT: return 'INCOMPLETE_ATTACHMENT';
      case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: return 'MISSING_ATTACHMENT';
      case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS: return 'INCOMPLETE_DIMENSIONS';
      case gl.FRAMEBUFFER_UNSUPPORTED: return 'UNSUPPORTED';
      default: return `UNKNOWN(${status})`;
    }
  }

  /**
   * Estimate memory usage for given configuration
   */
  estimateMemoryUsage(particleCount) {
    const bytesPerTexel = this.capabilities.precision === 'high' ? 16 : 4; // RGBA32F vs RGBA8
    const textureSize = Math.ceil(Math.sqrt(particleCount));
    const texelsPerTexture = textureSize * textureSize;
    const textureCount = 4; // positions, velocities × 2 (ping-pong)
    
    return {
      textureSizePx: textureSize,
      totalTexels: texelsPerTexture * textureCount,
      memoryMB: (texelsPerTexture * textureCount * bytesPerTexel) / (1024 * 1024),
      precision: this.capabilities.precision
    };
  }

  /**
   * Clean up GPU resources
   */
  dispose() {
    // Individual textures and framebuffers should be disposed by caller
    // This is just a placeholder for any global cleanup
  }
}

/**
 * Index to UV coordinate conversion
 */
export function indexToUV(index, width, height) {
  const x = index % width;
  const y = Math.floor(index / width);
  return [(x + 0.5) / width, (y + 0.5) / height];
}

/**
 * UV to index conversion
 */
export function uvToIndex(u, v, width, height) {
  const x = Math.floor(u * width);
  const y = Math.floor(v * height);
  return y * width + x;
}

/**
 * Generate deterministic particle positions
 */
export function generateParticlePositions(count, seed = 12345, worldBounds = { min: [-10, -10, -10], max: [10, 10, 10] }) {
  // Calculate texture dimensions to determine actual array size needed
  const textureSize = Math.ceil(Math.sqrt(count));
  const totalTexels = textureSize * textureSize;
  const positions = new Float32Array(totalTexels * 4); // RGBA - pad to texture size
  
  // Simple LCG for deterministic random numbers
  let rng = seed;
  function random() {
    rng = (rng * 1664525 + 1013904223) % 4294967296;
    return rng / 4294967296;
  }

  for (let i = 0; i < count; i++) {
    const base = i * 4;
    
    // Random position within world bounds
    positions[base + 0] = worldBounds.min[0] + random() * (worldBounds.max[0] - worldBounds.min[0]); // x
    positions[base + 1] = worldBounds.min[1] + random() * (worldBounds.max[1] - worldBounds.min[1]); // y
    positions[base + 2] = worldBounds.min[2] + random() * (worldBounds.max[2] - worldBounds.min[2]); // z
    positions[base + 3] = 1.0; // mass
  }

  // Fill remaining texels with zero data (inactive particles)
  for (let i = count; i < totalTexels; i++) {
    const base = i * 4;
    positions[base + 0] = 0.0;
    positions[base + 1] = 0.0;
    positions[base + 2] = 0.0;
    positions[base + 3] = 0.0;
  }

  return positions;
}

/**
 * Generate deterministic particle velocities
 */
export function generateParticleVelocities(count, seed = 54321) {
  // Calculate texture dimensions to determine actual array size needed
  const textureSize = Math.ceil(Math.sqrt(count));
  const totalTexels = textureSize * textureSize;
  const velocities = new Float32Array(totalTexels * 4); // RGBA - pad to texture size
  
  let rng = seed;
  function random() {
    rng = (rng * 1664525 + 1013904223) % 4294967296;
    return rng / 4294967296;
  }

  for (let i = 0; i < count; i++) {
    const base = i * 4;
    
    // Small initial velocities
    velocities[base + 0] = (random() - 0.5) * 2.0; // vx
    velocities[base + 1] = (random() - 0.5) * 2.0; // vy
    velocities[base + 2] = (random() - 0.5) * 2.0; // vz
    velocities[base + 3] = 0.0; // padding
  }

  // Fill remaining texels with zero data (inactive particles)
  for (let i = count; i < totalTexels; i++) {
    const base = i * 4;
    velocities[base + 0] = 0.0;
    velocities[base + 1] = 0.0;
    velocities[base + 2] = 0.0;
    velocities[base + 3] = 0.0;
  }

  return velocities;
}
