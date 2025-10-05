#version 300 es

// Point rendering vertex shader - reads particle positions from texture
// Each vertex represents one particle, rendered as a point

in float a_index;               // Particle index

uniform sampler2D u_positions;  // Position texture
uniform vec2 u_texSize;         // Texture dimensions
uniform mat4 u_projectionView;  // Combined projection-view matrix
uniform float u_pointSize;      // Point size in pixels

out vec3 v_color;               // Color based on velocity/force

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
}
