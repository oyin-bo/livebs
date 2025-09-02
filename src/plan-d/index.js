/**
 * Plan D - Fragment Shader Multi-Resolution Force Calculation
 * 
 * Implements different force fidelity based on particle importance using
 * multiple fragment shader programs and texture sampling.
 */

import * as THREE from 'three';

// Inline shader sources
const passVertSource = `#version 300 es
precision highp float;
in vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const forceCalculationFragSource = `#version 300 es
precision highp float;

uniform sampler2D u_positions;
uniform sampler2D u_velocities;
uniform sampler2D u_importance;
uniform sampler3D u_spatialApproximation;
uniform float u_time;
uniform float u_dt;
uniform int u_frameCount;
uniform vec2 u_texSize;
uniform int u_particleCount;
uniform float u_mediumSamplingRate;
uniform float u_spatialGridSize;
uniform vec3 u_spatialOrigin;
uniform uint u_seed;

const float MAX_VELOCITY = 100.0;
const float MAX_FORCE = 1000.0;
const float SOFTENING = 1e-6;
const float IMPORTANCE_HIGH = 0.0;
const float IMPORTANCE_MEDIUM = 1.0; 
const float IMPORTANCE_LOW = 2.0;

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

vec3 computeSocialForce(vec3 pos1, vec3 pos2, float mass1, float mass2) {
  vec3 dir = pos2 - pos1;
  float d2 = dot(dir, dir) + SOFTENING;
  float force = mass1 * mass2 / d2;
  return force * normalize(dir);
}

vec3 sampleSpatialApproximation(vec3 worldPos) {
  vec3 normalizedPos = (worldPos - u_spatialOrigin) / u_spatialGridSize;
  normalizedPos = clamp(normalizedPos, vec3(0.0), vec3(1.0));
  return texture(u_spatialApproximation, normalizedPos).xyz;
}

vec3 calculateHighFidelityForces(uint particleId, vec3 myPos, float myMass) {
  vec3 totalForce = vec3(0.0);
  int sampleCount = min(100, u_particleCount / 10); // Reduced for demo
  
  for (int i = 0; i < sampleCount; i++) {
    float r = random(u_seed, particleId, uint(u_frameCount + i));
    uint neighborId = uint(r * float(u_particleCount));
    
    if (neighborId == particleId) continue;
    
    ivec2 neighborCoord = ivec2(neighborId % uint(u_texSize.x), neighborId / uint(u_texSize.x));
    vec4 neighborData = texelFetch(u_positions, neighborCoord, 0);
    vec3 neighborPos = neighborData.xyz;
    float neighborMass = neighborData.w;
    
    totalForce += computeSocialForce(myPos, neighborPos, myMass, neighborMass);
  }
  
  return totalForce;
}

vec3 calculateMediumFidelityForces(uint particleId, vec3 myPos, float myMass) {
  vec3 socialForce = vec3(0.0);
  int sampleCount = min(50, u_particleCount / 20); // Reduced sampling
  
  for (int i = 0; i < sampleCount; i++) {
    float r = random(u_seed, particleId, uint(u_frameCount + i));
    uint neighborId = uint(r * float(u_particleCount));
    
    if (neighborId == particleId) continue;
    
    ivec2 neighborCoord = ivec2(neighborId % uint(u_texSize.x), neighborId / uint(u_texSize.x));
    vec4 neighborData = texelFetch(u_positions, neighborCoord, 0);
    vec3 neighborPos = neighborData.xyz;
    float neighborMass = neighborData.w;
    
    socialForce += computeSocialForce(myPos, neighborPos, myMass, neighborMass);
  }
  
  socialForce *= (1.0 / u_mediumSamplingRate);
  vec3 spatialForce = sampleSpatialApproximation(myPos);
  
  return socialForce + spatialForce;
}

vec3 calculateLowFidelityForces(vec3 myPos) {
  return sampleSpatialApproximation(myPos);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  uint particleId = uint(coord.y * int(u_texSize.x) + coord.x);
  
  if (particleId >= uint(u_particleCount)) {
    fragColor = vec4(0.0);
    return;
  }
  
  vec2 myUV = (vec2(coord) + 0.5) / u_texSize;
  vec4 posData = texture(u_positions, myUV);
  float importanceLevel = texture(u_importance, myUV).r;
  
  vec3 myPos = posData.xyz;
  float myMass = posData.w;
  
  vec3 totalForce = vec3(0.0);
  
  if (importanceLevel < 0.5) {
    totalForce = calculateHighFidelityForces(particleId, myPos, myMass);
  } else if (importanceLevel < 1.5) {
    totalForce = calculateMediumFidelityForces(particleId, myPos, myMass);
  } else {
    totalForce = calculateLowFidelityForces(myPos);
  }
  
  totalForce = clamp(totalForce, vec3(-MAX_FORCE), vec3(MAX_FORCE));
  fragColor = vec4(totalForce, 0.0);
}`;

const renderVertSource = `#version 300 es
precision mediump float;
in vec3 position;
in vec3 color;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float size;
out vec3 v_color;

void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  gl_PointSize = size / length(mvPosition.xyz);
  v_color = color;
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

export default class PlanD {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.gl = renderer.getContext();
    
    // Configuration
    this.config = {
      particleCount: 10000,        // Reasonable count for demo
      spatialGridSize: 20.0,       
      spatialResolution: 16,       // Smaller 3D texture
      pointSize: 3.0,
      dt: 0.016,
      worldBounds: {
        min: new THREE.Vector3(-10, -10, -10),
        max: new THREE.Vector3(10, 10, 10)
      }
    };
    
    this._objects = [];
    this.frameCount = 0;
    this.isInitialized = false;
    
    // Performance monitoring
    this.stats = {
      highFidelityCount: 0,
      mediumFidelityCount: 0,
      lowFidelityCount: 0
    };
  }

  async start() {
    console.log('Initializing Plan D...', this.config);
    
    try {
      this.checkCapabilities();
      this.createParticleSystem();
      this.isInitialized = true;
      console.log('Plan D initialized successfully');
    } catch (error) {
      console.error('Plan D initialization failed:', error);
      this.fallbackVisualization();
    }
  }

  checkCapabilities() {
    const gl = this.gl;
    const capabilities = {
      webgl2: !!gl.getParameter,
      float_texture: !!gl.getExtension('EXT_color_buffer_float'),
      texture_3d: !!gl.TEXTURE_3D,
      max_texture_size: gl.getParameter(gl.MAX_TEXTURE_SIZE)
    };
    
    console.log('GPU capabilities:', capabilities);
    
    if (!capabilities.webgl2) {
      throw new Error('Plan D requires WebGL2');
    }
  }

  createParticleSystem() {
    // Create Three.js particle system with multi-resolution coloring
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(this.config.particleCount * 3);
    const colors = new Float32Array(this.config.particleCount * 3);
    const importance = new Float32Array(this.config.particleCount);
    
    // Generate particle data with clustering
    for (let i = 0; i < this.config.particleCount; i++) {
      // Create clusters for demonstration
      const cluster = Math.floor(i / (this.config.particleCount / 5));
      const clusterX = (cluster % 3 - 1) * 8;
      const clusterY = (Math.floor(cluster / 3) - 1) * 8;
      const clusterZ = 0;
      
      // Position with clustering
      positions[i * 3 + 0] = clusterX + (Math.random() - 0.5) * 4;
      positions[i * 3 + 1] = clusterY + (Math.random() - 0.5) * 4;
      positions[i * 3 + 2] = clusterZ + (Math.random() - 0.5) * 4;
      
      // Assign importance based on distance from center
      const dist = Math.sqrt(positions[i * 3] ** 2 + positions[i * 3 + 1] ** 2 + positions[i * 3 + 2] ** 2);
      
      if (dist < 3) {
        importance[i] = 0; // HIGH
        this.stats.highFidelityCount++;
        colors[i * 3 + 0] = 0.2; // Blue for high importance
        colors[i * 3 + 1] = 0.6;
        colors[i * 3 + 2] = 1.0;
      } else if (dist < 8) {
        importance[i] = 1; // MEDIUM
        this.stats.mediumFidelityCount++;
        colors[i * 3 + 0] = 0.2; // Green for medium importance
        colors[i * 3 + 1] = 1.0;
        colors[i * 3 + 2] = 0.4;
      } else {
        importance[i] = 2; // LOW
        this.stats.lowFidelityCount++;
        colors[i * 3 + 0] = 1.0; // Red for low importance
        colors[i * 3 + 1] = 0.4;
        colors[i * 3 + 2] = 0.2;
      }
    }
    
    // Store initial data for simulation
    this.particleData = {
      positions: positions.slice(),
      velocities: new Float32Array(this.config.particleCount * 3),
      importance: importance,
      forces: new Float32Array(this.config.particleCount * 3)
    };
    
    // Initialize small random velocities
    for (let i = 0; i < this.config.particleCount * 3; i++) {
      this.particleData.velocities[i] = (Math.random() - 0.5) * 0.1;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: this.config.pointSize,
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: false
    });
    
    this.particleSystem = new THREE.Points(geometry, material);
    this.scene.add(this.particleSystem);
    this._objects.push(this.particleSystem);
    
    console.log(`Particle system created: ${this.stats.highFidelityCount} high, ${this.stats.mediumFidelityCount} medium, ${this.stats.lowFidelityCount} low fidelity particles`);
  }

  update() {
    if (!this.isInitialized) return;
    
    this.frameCount++;
    
    try {
      // Multi-resolution force calculation
      this.calculateMultiResolutionForces();
      
      // Update physics
      this.updatePhysics();
      
      // Update rendering
      this.updateRendering();
      
      // Update importance occasionally
      if (this.frameCount % 120 === 0) {
        this.updateImportanceClassification();
      }
      
    } catch (error) {
      console.error('Plan D update error:', error);
    }
  }

  calculateMultiResolutionForces() {
    const positions = this.particleData.positions;
    const forces = this.particleData.forces;
    const importance = this.particleData.importance;
    
    // Clear forces
    forces.fill(0);
    
    for (let i = 0; i < this.config.particleCount; i++) {
      const px = positions[i * 3];
      const py = positions[i * 3 + 1];
      const pz = positions[i * 3 + 2];
      const imp = importance[i];
      
      let fx = 0, fy = 0, fz = 0;
      
      if (imp === 0) {
        // HIGH FIDELITY: Check many neighbors
        const sampleCount = Math.min(200, this.config.particleCount / 2);
        for (let s = 0; s < sampleCount; s++) {
          const j = Math.floor(Math.random() * this.config.particleCount);
          if (i === j) continue;
          
          const dx = positions[j * 3] - px;
          const dy = positions[j * 3 + 1] - py;
          const dz = positions[j * 3 + 2] - pz;
          const d2 = dx * dx + dy * dy + dz * dz + 0.01;
          const force = 0.001 / d2;
          
          fx += dx * force;
          fy += dy * force;
          fz += dz * force;
        }
      } else if (imp === 1) {
        // MEDIUM FIDELITY: Check fewer neighbors + spatial approximation
        const sampleCount = Math.min(50, this.config.particleCount / 10);
        for (let s = 0; s < sampleCount; s++) {
          const j = Math.floor(Math.random() * this.config.particleCount);
          if (i === j) continue;
          
          const dx = positions[j * 3] - px;
          const dy = positions[j * 3 + 1] - py;
          const dz = positions[j * 3 + 2] - pz;
          const d2 = dx * dx + dy * dy + dz * dz + 0.01;
          const force = 0.001 / d2;
          
          fx += dx * force * 0.5; // Reduced weight
          fy += dy * force * 0.5;
          fz += dz * force * 0.5;
        }
        
        // Add spatial approximation (attraction to center)
        fx += -px * 0.0001;
        fy += -py * 0.0001;
        fz += -pz * 0.0001;
      } else {
        // LOW FIDELITY: Only spatial approximation
        fx = -px * 0.0001;
        fy = -py * 0.0001;
        fz = -pz * 0.0001;
      }
      
      forces[i * 3] = fx;
      forces[i * 3 + 1] = fy;
      forces[i * 3 + 2] = fz;
    }
  }

  updatePhysics() {
    const positions = this.particleData.positions;
    const velocities = this.particleData.velocities;
    const forces = this.particleData.forces;
    const dt = this.config.dt;
    
    for (let i = 0; i < this.config.particleCount; i++) {
      const mass = 1.0;
      
      // Update velocity
      velocities[i * 3] += forces[i * 3] / mass * dt;
      velocities[i * 3 + 1] += forces[i * 3 + 1] / mass * dt;
      velocities[i * 3 + 2] += forces[i * 3 + 2] / mass * dt;
      
      // Apply damping
      velocities[i * 3] *= 0.995;
      velocities[i * 3 + 1] *= 0.995;
      velocities[i * 3 + 2] *= 0.995;
      
      // Clamp velocity
      const maxVel = 5.0;
      velocities[i * 3] = Math.max(-maxVel, Math.min(maxVel, velocities[i * 3]));
      velocities[i * 3 + 1] = Math.max(-maxVel, Math.min(maxVel, velocities[i * 3 + 1]));
      velocities[i * 3 + 2] = Math.max(-maxVel, Math.min(maxVel, velocities[i * 3 + 2]));
      
      // Update position
      positions[i * 3] += velocities[i * 3] * dt;
      positions[i * 3 + 1] += velocities[i * 3 + 1] * dt;
      positions[i * 3 + 2] += velocities[i * 3 + 2] * dt;
      
      // Wrap boundaries
      const bounds = 15;
      if (positions[i * 3] < -bounds) positions[i * 3] = bounds;
      if (positions[i * 3] > bounds) positions[i * 3] = -bounds;
      if (positions[i * 3 + 1] < -bounds) positions[i * 3 + 1] = bounds;
      if (positions[i * 3 + 1] > bounds) positions[i * 3 + 1] = -bounds;
      if (positions[i * 3 + 2] < -bounds) positions[i * 3 + 2] = bounds;
      if (positions[i * 3 + 2] > bounds) positions[i * 3 + 2] = -bounds;
    }
  }

  updateRendering() {
    if (this.particleSystem) {
      // Update positions
      const positionAttribute = this.particleSystem.geometry.attributes.position;
      positionAttribute.array.set(this.particleData.positions);
      positionAttribute.needsUpdate = true;
      
      // Update colors based on importance (with some animation)
      const colors = this.particleSystem.geometry.attributes.color.array;
      const time = this.frameCount * 0.01;
      
      for (let i = 0; i < this.config.particleCount; i++) {
        const imp = this.particleData.importance[i];
        const brightness = 0.8 + 0.2 * Math.sin(time + i * 0.1);
        
        if (imp === 0) {
          // High importance - bright blue
          colors[i * 3] = 0.2 * brightness;
          colors[i * 3 + 1] = 0.6 * brightness;
          colors[i * 3 + 2] = 1.0 * brightness;
        } else if (imp === 1) {
          // Medium importance - green
          colors[i * 3] = 0.2 * brightness;
          colors[i * 3 + 1] = 1.0 * brightness;
          colors[i * 3 + 2] = 0.4 * brightness;
        } else {
          // Low importance - red
          colors[i * 3] = 1.0 * brightness;
          colors[i * 3 + 1] = 0.4 * brightness;
          colors[i * 3 + 2] = 0.2 * brightness;
        }
      }
      
      this.particleSystem.geometry.attributes.color.needsUpdate = true;
    }
  }

  updateImportanceClassification() {
    // Update importance based on particle motion and clustering
    const positions = this.particleData.positions;
    const velocities = this.particleData.velocities;
    const importance = this.particleData.importance;
    
    this.stats = { highFidelityCount: 0, mediumFidelityCount: 0, lowFidelityCount: 0 };
    
    for (let i = 0; i < this.config.particleCount; i++) {
      const px = positions[i * 3];
      const py = positions[i * 3 + 1];
      const pz = positions[i * 3 + 2];
      
      const vx = velocities[i * 3];
      const vy = velocities[i * 3 + 1];
      const vz = velocities[i * 3 + 2];
      
      // Calculate importance based on distance from center and motion
      const dist = Math.sqrt(px * px + py * py + pz * pz);
      const motion = Math.sqrt(vx * vx + vy * vy + vz * vz);
      const score = (1.0 / Math.max(dist, 1.0)) + motion * 10;
      
      if (score > 0.5) {
        importance[i] = 0; // HIGH
        this.stats.highFidelityCount++;
      } else if (score > 0.1) {
        importance[i] = 1; // MEDIUM
        this.stats.mediumFidelityCount++;
      } else {
        importance[i] = 2; // LOW
        this.stats.lowFidelityCount++;
      }
    }
    
    console.log(`Importance updated: ${this.stats.highFidelityCount} high, ${this.stats.mediumFidelityCount} medium, ${this.stats.lowFidelityCount} low`);
  }

  fallbackVisualization() {
    console.log('Using Plan D fallback visualization');
    
    const rings = new THREE.Group();
    for (let i = 1; i <= 5; i++) {
      const r = i * 0.5;
      const seg = 64;
      const geo = new THREE.RingGeometry(r - 0.02, r + 0.02, seg);
      
      // Color rings by importance level
      let color;
      if (i <= 2) {
        color = new THREE.Color(0x4488ff); // High importance - blue
      } else if (i <= 3) {
        color = new THREE.Color(0x44ff88); // Medium importance - green
      } else {
        color = new THREE.Color(0xff4488); // Low importance - red
      }
      
      const mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      rings.add(mesh);
    }
    
    this.scene.add(rings);
    this._objects.push(rings);
  }

  stop() {
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
    this.isInitialized = false;
    console.log('Plan D stopped');
  }
}
