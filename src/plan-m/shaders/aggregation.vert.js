export default `#version 300 es
precision highp float;

// Octree 3D aggregation with Z-slice stacking

uniform sampler2D u_positions;   // RGBA: xyz + mass
uniform vec2 u_texSize;          // positions texture size
uniform vec3 u_worldMin;         // XYZ world min
uniform vec3 u_worldMax;         // XYZ world max
uniform float u_gridSize;        // octree grid size (e.g., 64)
uniform float u_slicesPerRow;    // slices per row (e.g., 8 for 8x8 grid)

out vec4 v_particleData;

ivec2 indexToCoord(int index, vec2 texSize) {
  int w = int(texSize.x);
  int ix = index % w;
  int iy = index / w;
  return ivec2(ix, iy);
}

// Convert 3D voxel coordinate to 2D texture coordinate
vec2 voxelToTexel(vec3 voxelCoord, float gridSize, float slicesPerRow) {
  float vx = voxelCoord.x;
  float vy = voxelCoord.y;
  float vz = voxelCoord.z;
  
  // Which Z-slice?
  float sliceIndex = vz;
  float sliceRow = floor(sliceIndex / slicesPerRow);
  float sliceCol = mod(sliceIndex, slicesPerRow);
  
  // Texel position
  float texelX = sliceCol * gridSize + vx;
  float texelY = sliceRow * gridSize + vy;
  
  return vec2(texelX, texelY);
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

  // Map particle XYZ to 3D voxel grid
  vec3 norm = (pos.xyz - u_worldMin) / (u_worldMax - u_worldMin);
  norm = clamp(norm, vec3(0.0), vec3(1.0 - (1.0 / u_gridSize)));
  vec3 voxelCoord = floor(norm * u_gridSize);
  
  // Convert 3D voxel to 2D texture coordinate
  float textureSize = u_gridSize * u_slicesPerRow;
  vec2 texelPos = voxelToTexel(voxelCoord, u_gridSize, u_slicesPerRow);
  vec2 texelCenter = (texelPos + 0.5) / textureSize;
  vec2 clip = texelCenter * 2.0 - 1.0;

  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = 1.0;

  // Weighted sum: store 3D weighted position and mass
  v_particleData = vec4(pos.xyz * mass, mass);
}`;
