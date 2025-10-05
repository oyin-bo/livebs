#version 300 es
precision highp float;

// Update fragment shader - performs particle physics simulation
// Each fragment processes one particle, outputs new position

// Input uniforms (must be bound by host)
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

// Constants
const float MAX_VELOCITY = 100.0;
const float MAX_FORCE = 1000.0;
const int MAX_SAMPLES = 256; // Fixed limit for shader loop compatibility
const float SOFTENING = 1e-6;
const float DAMPING = 0.8;

// Outputs
out vec4 fragColor;                // New position (xyz) + mass (w)

// RNG implementation - deterministic hash for sampling
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

// Index/UV conversion utilities
vec2 indexToUV(int index) {
  int x = index % int(u_texSize.x);
  int y = index / int(u_texSize.x);
  return (vec2(x, y) + 0.5) / u_texSize;
}

// Force calculation between two particles
vec3 computePairForce(vec3 pos1, vec3 pos2, float mass1, float mass2) {
  vec3 dir = pos2 - pos1;
  float d2 = dot(dir, dir) + SOFTENING;
  float d = sqrt(d2);
  
  // Gravitational-like force (attractive)
  float force = mass1 * mass2 / d2;
  
  return force * normalize(dir);
}

// World bounds handling - wrap mode
vec3 applyWrapBounds(vec3 position, vec3 minBounds, vec3 maxBounds) {
  vec3 size = maxBounds - minBounds;
  position = mod(position - minBounds, size) + minBounds;
  return position;
}

// World bounds handling - clamp mode with bounce
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
  // Current particle index
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int myIndex = coord.y * int(u_texSize.x) + coord.x;
  
  // Skip particles beyond count
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
  
  // Accumulate forces from neighbors using stochastic sampling
  vec3 totalForce = vec3(0.0);
  int sampleCount = max(1, min(MAX_SAMPLES, int(float(u_particleCount) * u_samplingFraction)));
  
  for (int i = 0; i < MAX_SAMPLES; i++) {
    if (i >= sampleCount) break; // Early exit for bounded loops
    
    float r = random(u_seed, uint(myIndex), uint(u_frameCount + i));
    int neighborIndex = int(r * float(u_particleCount));
    
    if (neighborIndex == myIndex) continue;
    
    vec2 neighborUV = indexToUV(neighborIndex);
    vec4 neighborPosData = texture(u_positions, neighborUV);
    vec3 neighborPos = neighborPosData.xyz;
    float neighborMass = neighborPosData.w;
    
    totalForce += computePairForce(myPos, neighborPos, myMass, neighborMass);
  }
  
  // Scale force to account for sampling fraction
  totalForce *= (1.0 / u_samplingFraction);
  
  // Apply stability limits
  totalForce = clamp(totalForce, vec3(-MAX_FORCE), vec3(MAX_FORCE));
  myVel = clamp(myVel, vec3(-MAX_VELOCITY), vec3(MAX_VELOCITY));
  
  // Integration
  vec3 newPos;
  vec3 newVel = myVel;
  
  if (u_integrationMethod == 0) { // Euler
    vec3 acceleration = totalForce / myMass;
    newVel += acceleration * u_dt;
    newPos = myPos + newVel * u_dt;
  } else { // Semi-implicit Euler
    vec3 acceleration = totalForce / myMass;
    newVel += acceleration * u_dt;
    newPos = myPos + newVel * u_dt;
  }
  
  // Apply world bounds
  if (u_wrapMode == 0) { // Wrap
    newPos = applyWrapBounds(newPos, u_worldMin, u_worldMax);
  } else { // Clamp with bounce
    newPos = applyClampBounds(newPos, newVel, u_worldMin, u_worldMax);
  }
  
  // Final stability check
  newVel = clamp(newVel, vec3(-MAX_VELOCITY), vec3(MAX_VELOCITY));
  
  // Output new position (velocity gets written to separate target in actual implementation)
  fragColor = vec4(newPos, myMass);
}
