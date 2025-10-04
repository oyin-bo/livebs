export default `#version 300 es
precision highp float;

uniform sampler2D u_positions;
uniform vec2 u_texSize;
uniform mat4 u_projectionView;
uniform float u_pointSize;

out vec3 v_color;

ivec2 indexToCoord(int index, vec2 texSize) {
  int w = int(texSize.x);
  int ix = index % w;
  int iy = index / w;
  return ivec2(ix, iy);
}

void main() {
  int index = gl_VertexID;
  ivec2 coord = indexToCoord(index, u_texSize);
  vec4 posData = texelFetch(u_positions, coord, 0);
  vec3 worldPos = posData.xyz;
  
  // Cull unused texels (mass == 0)
  if (posData.w <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // Behind camera
    gl_PointSize = 0.0;
    v_color = vec3(0.0);
    return;
  }
  
  // Transform world position using camera's projection-view matrix
  gl_Position = u_projectionView * vec4(worldPos, 1.0);
  gl_PointSize = u_pointSize;
  
  // Color based on position for visualization
  v_color = normalize(worldPos) * 0.5 + 0.5;
}`;
