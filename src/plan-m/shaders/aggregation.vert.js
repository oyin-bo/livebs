export default `#version 300 es
precision highp float;

// Use gl_VertexID instead of attribute for robust indexing

uniform sampler2D u_positions;   // RGBA: xyz + mass
uniform vec2 u_texSize;          // positions texture size
uniform vec2 u_worldMin;         // XY world min
uniform vec2 u_worldMax;         // XY world max
uniform float u_gridSize;        // L0 size (square)

out vec4 v_particleData;

ivec2 indexToCoord(int index, vec2 texSize) {
  int w = int(texSize.x);
  int ix = index % w;
  int iy = index / w;
  return ivec2(ix, iy);
}

void main() {
  int index = gl_VertexID;
  ivec2 coord = indexToCoord(index, u_texSize);
  vec4 pos = texelFetch(u_positions, coord, 0);
  float mass = pos.a;

  if (mass <= 0.0) {
    // Cull zero-mass entries
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 0.0;
    v_particleData = vec4(0.0);
    return;
  }

  // Map to grid texel
  vec2 norm = (pos.xy - u_worldMin) / (u_worldMax - u_worldMin);
  norm = clamp(norm, vec2(0.0), vec2(1.0 - (1.0 / u_gridSize)));
  vec2 gridCoord = floor(norm * u_gridSize);
  vec2 texelCenter = (gridCoord + 0.5) / u_gridSize;
  vec2 clip = texelCenter * 2.0 - 1.0;

  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = 1.0;

  // Weighted sum and counters
  v_particleData = vec4(pos.xy * mass, mass, 1.0);
}`;
