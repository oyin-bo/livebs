#version 300 es
precision highp float;

// Velocity update fragment shader
// Updates velocities based on calculated forces

uniform sampler2D u_velocities;     // Current velocities
uniform sampler2D u_forces;         // Calculated forces from force-calculation.frag
uniform float u_dt;                 // Delta time
uniform vec2 u_texSize;            // Texture dimensions
uniform int u_particleCount;

out vec4 fragColor;

const float MAX_VELOCITY = 100.0;
const float DAMPING = 0.99;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  uint particleId = uint(coord.y * int(u_texSize.x) + coord.x);
  
  if (particleId >= uint(u_particleCount)) {
    fragColor = vec4(0.0);
    return;
  }
  
  vec2 myUV = (vec2(coord) + 0.5) / u_texSize;
  vec3 currentVel = texture(u_velocities, myUV).xyz;
  vec3 force = texture(u_forces, myUV).xyz;
  float mass = texture(u_velocities, myUV).w; // Mass stored in w component
  
  // Apply force to update velocity
  vec3 acceleration = force / max(mass, 0.001); // Avoid division by zero
  vec3 newVel = currentVel + acceleration * u_dt;
  
  // Apply damping
  newVel *= DAMPING;
  
  // Clamp velocity
  newVel = clamp(newVel, vec3(-MAX_VELOCITY), vec3(MAX_VELOCITY));
  
  fragColor = vec4(newVel, mass);
}