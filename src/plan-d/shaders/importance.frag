#version 300 es
precision highp float;

// Importance classification fragment shader
// Determines particle importance based on distance to camera, motion, screen size

uniform sampler2D u_positions;          // Current positions
uniform sampler2D u_velocities;         // Current velocities
uniform vec3 u_cameraPos;               // Camera position
uniform vec3 u_userFocus;               // User focus point
uniform mat4 u_viewProjectionMatrix;    // View-projection matrix for screen projection
uniform vec2 u_screenSize;              // Screen dimensions
uniform vec2 u_texSize;                  // Texture dimensions
uniform int u_particleCount;

out vec4 fragColor;

// Importance levels
const float IMPORTANCE_HIGH = 0.0;
const float IMPORTANCE_MEDIUM = 1.0; 
const float IMPORTANCE_LOW = 2.0;

// Classification thresholds
const float HIGH_THRESHOLD = 0.1;
const float MEDIUM_THRESHOLD = 0.01;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  uint particleId = uint(coord.y * int(u_texSize.x) + coord.x);
  
  if (particleId >= uint(u_particleCount)) {
    fragColor = vec4(IMPORTANCE_LOW, 0.0, 0.0, 1.0);
    return;
  }
  
  vec2 myUV = (vec2(coord) + 0.5) / u_texSize;
  vec3 position = texture(u_positions, myUV).xyz;
  vec3 velocity = texture(u_velocities, myUV).xyz;
  
  // Calculate screen-space position
  vec4 clipSpace = u_viewProjectionMatrix * vec4(position, 1.0);
  vec3 ndc = clipSpace.xyz / clipSpace.w;
  vec2 screenPos = (ndc.xy * 0.5 + 0.5) * u_screenSize;
  
  // Calculate screen size (approximate based on distance)
  float distanceToCamera = length(position - u_cameraPos);
  float screenSize = 1.0 / max(distanceToCamera, 1.0);
  
  // Calculate distance to user focus point
  float focusDistance = length(position - u_userFocus);
  float focusFactor = 1.0 / max(focusDistance, 1.0);
  
  // Calculate motion factor
  float motion = length(velocity);
  
  // Combine factors to get importance score
  float importanceScore = screenSize * focusFactor * (1.0 + motion);
  
  // Classify importance
  float importance;
  if (importanceScore > HIGH_THRESHOLD) {
    importance = IMPORTANCE_HIGH;
  } else if (importanceScore > MEDIUM_THRESHOLD) {
    importance = IMPORTANCE_MEDIUM;
  } else {
    importance = IMPORTANCE_LOW;
  }
  
  fragColor = vec4(importance, 0.0, 0.0, 1.0);
}