#version 300 es
precision highp float;

// Plan D Multi-Resolution Force Calculation Fragment Shader
// Implements different fidelity levels based on particle importance

// Input textures
uniform sampler2D u_positions;           // Current positions (RGBA32F) xyz=pos, w=mass
uniform sampler2D u_velocities;          // Current velocities (RGBA32F) xyz=vel, w=damping
uniform sampler2D u_importance;          // Particle importance (R32F) 0.0=HIGH, 1.0=MEDIUM, 2.0=LOW
uniform sampler2D u_neighborData;        // Neighbor lists (packed format)
uniform sampler2D u_neighborCounts;      // Number of neighbors per particle
uniform sampler3D u_spatialApproximation; // 3D force field approximation

// Simulation parameters
uniform float u_time;                    // Global time
uniform float u_dt;                      // Delta time
uniform int u_frameCount;                // Frame counter for RNG
uniform vec2 u_texSize;                  // Texture dimensions
uniform int u_particleCount;             // Active particle count
uniform vec3 u_worldMin;                 // World bounds
uniform vec3 u_worldMax;                 // World bounds
uniform vec3 u_cameraPos;                // Camera position for importance
uniform vec3 u_userFocus;                // User focus point
uniform uint u_seed;                     // RNG seed

// Quality parameters
uniform float u_mediumSamplingRate;      // Sampling rate for medium fidelity (0.5)
uniform float u_spatialGridSize;         // Size of spatial approximation grid
uniform vec3 u_spatialOrigin;            // Origin of spatial grid

// Constants
const float MAX_VELOCITY = 100.0;
const float MAX_FORCE = 1000.0;
const int MAX_NEIGHBORS = 256;
const float SOFTENING = 1e-6;
const float DAMPING = 0.8;

// Importance levels
const float IMPORTANCE_HIGH = 0.0;
const float IMPORTANCE_MEDIUM = 1.0; 
const float IMPORTANCE_LOW = 2.0;

// Output
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

// Force calculation between two particles
vec3 computeSocialForce(vec3 pos1, vec3 pos2, float mass1, float mass2, float weight) {
  vec3 dir = pos2 - pos1;
  float d2 = dot(dir, dir) + SOFTENING;
  float d = sqrt(d2);
  
  // Social network force with weight
  float force = mass1 * mass2 * weight / d2;
  
  return force * normalize(dir);
}

// Sample spatial approximation from 3D texture
vec3 sampleSpatialApproximation(vec3 worldPos) {
  // Normalize world position to texture coordinates [0,1]
  vec3 normalizedPos = (worldPos - u_spatialOrigin) / u_spatialGridSize;
  normalizedPos = clamp(normalizedPos, vec3(0.0), vec3(1.0));
  
  // Sample from 3D texture
  return texture(u_spatialApproximation, normalizedPos).xyz;
}

// HIGH FIDELITY: Process all social network neighbors
vec3 calculateHighFidelityForces(uint particleId, vec3 myPos, float myMass) {
  vec3 totalForce = vec3(0.0);
  
  // Get neighbor count
  ivec2 countCoord = ivec2(particleId % uint(u_texSize.x), particleId / uint(u_texSize.x));
  float neighborCount = texelFetch(u_neighborCounts, countCoord, 0).r;
  uint maxNeighbors = min(uint(neighborCount), uint(MAX_NEIGHBORS));
  
  // Process all neighbors
  for (uint i = 0u; i < maxNeighbors; i++) {
    // Read neighbor data (simplified - would need actual neighbor list texture format)
    ivec2 neighborCoord = ivec2((particleId * uint(MAX_NEIGHBORS) + i) % uint(u_texSize.x * 4u),
                               (particleId * uint(MAX_NEIGHBORS) + i) / uint(u_texSize.x * 4u));
    
    // For now, use stochastic sampling similar to Plan A but with all neighbors
    float r = random(u_seed, particleId, uint(u_frameCount + int(i)));
    uint neighborId = uint(r * float(u_particleCount));
    
    if (neighborId == particleId) continue;
    
    ivec2 neighborPosCoord = ivec2(neighborId % uint(u_texSize.x), neighborId / uint(u_texSize.x));
    vec4 neighborData = texelFetch(u_positions, neighborPosCoord, 0);
    vec3 neighborPos = neighborData.xyz;
    float neighborMass = neighborData.w;
    
    totalForce += computeSocialForce(myPos, neighborPos, myMass, neighborMass, 1.0);
  }
  
  return totalForce;
}

// MEDIUM FIDELITY: Sampled neighbors + spatial approximation
vec3 calculateMediumFidelityForces(uint particleId, vec3 myPos, float myMass) {
  vec3 socialForce = vec3(0.0);
  
  // Sample subset of neighbors
  ivec2 countCoord = ivec2(particleId % uint(u_texSize.x), particleId / uint(u_texSize.x));
  float neighborCount = texelFetch(u_neighborCounts, countCoord, 0).r;
  uint maxNeighbors = min(uint(neighborCount), uint(MAX_NEIGHBORS));
  uint samplesToCheck = max(1u, uint(float(maxNeighbors) * u_mediumSamplingRate));
  
  for (uint i = 0u; i < samplesToCheck; i++) {
    float r = random(u_seed, particleId, uint(u_frameCount + int(i)));
    uint neighborId = uint(r * float(u_particleCount));
    
    if (neighborId == particleId) continue;
    
    ivec2 neighborPosCoord = ivec2(neighborId % uint(u_texSize.x), neighborId / uint(u_texSize.x));
    vec4 neighborData = texelFetch(u_positions, neighborPosCoord, 0);
    vec3 neighborPos = neighborData.xyz;
    float neighborMass = neighborData.w;
    
    socialForce += computeSocialForce(myPos, neighborPos, myMass, neighborMass, 1.0);
  }
  
  // Compensate for sampling
  socialForce *= (1.0 / u_mediumSamplingRate);
  
  // Add spatial approximation
  vec3 spatialForce = sampleSpatialApproximation(myPos);
  
  return socialForce + spatialForce;
}

// LOW FIDELITY: Only spatial approximation
vec3 calculateLowFidelityForces(vec3 myPos) {
  return sampleSpatialApproximation(myPos);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  uint particleId = uint(coord.y * int(u_texSize.x) + coord.x);
  
  // Skip particles beyond count
  if (particleId >= uint(u_particleCount)) {
    fragColor = vec4(0.0);
    return;
  }
  
  // Read current state
  vec2 myUV = (vec2(coord) + 0.5) / u_texSize;
  vec4 posData = texture(u_positions, myUV);
  vec4 velData = texture(u_velocities, myUV);
  float importanceLevel = texture(u_importance, myUV).r;
  
  vec3 myPos = posData.xyz;
  float myMass = posData.w;
  vec3 myVel = velData.xyz;
  
  // Calculate forces based on importance level
  vec3 totalForce = vec3(0.0);
  
  if (importanceLevel < 0.5) {
    // HIGH IMPORTANCE: Full neighbor processing
    totalForce = calculateHighFidelityForces(particleId, myPos, myMass);
  } else if (importanceLevel < 1.5) {
    // MEDIUM IMPORTANCE: Sampled neighbors + spatial approximation
    totalForce = calculateMediumFidelityForces(particleId, myPos, myMass);
  } else {
    // LOW IMPORTANCE: Spatial approximation only
    totalForce = calculateLowFidelityForces(myPos);
  }
  
  // Apply stability limits
  totalForce = clamp(totalForce, vec3(-MAX_FORCE), vec3(MAX_FORCE));
  
  // Integration (Semi-implicit Euler)
  vec3 acceleration = totalForce / myMass;
  vec3 newVel = myVel + acceleration * u_dt;
  newVel = clamp(newVel, vec3(-MAX_VELOCITY), vec3(MAX_VELOCITY));
  vec3 newPos = myPos + newVel * u_dt;
  
  // Apply world bounds (wrap mode)
  vec3 worldSize = u_worldMax - u_worldMin;
  newPos = mod(newPos - u_worldMin, worldSize) + u_worldMin;
  
  // Output new position
  fragColor = vec4(newPos, myMass);
}