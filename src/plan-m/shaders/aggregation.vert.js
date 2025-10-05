export default `#version 300 es
precision highp float;

// Attribute to match VAO binding (used to satisfy WebGL validation)
in float a_particleIndex;

// Use attribute for indexing instead of gl_VertexID to avoid GL error 0x502

uniform sampler2D u_positions;   // RGBA: xyz + mass
uniform vec2 u_texSize;          // positions texture size
uniform vec3 u_worldMin;         // XYZ world min
uniform vec3 u_worldMax;         // XYZ world max
uniform float u_gridSize;        // L0 size (cubic: gridSize³ voxels)
uniform int u_textureWidth;      // Width of output texture for 3D→2D mapping

out vec4 v_particleData;

ivec2 indexToCoord(int index, vec2 texSize) {
  int w = int(texSize.x);
  int ix = index % w;
  int iy = index / w;
  return ivec2(ix, iy);
}

// Map 3D voxel coordinates to 2D texture coordinates
// Layout: stack Z-slices in rows (each slice is gridSize×gridSize)
ivec2 voxelToTexCoord(ivec3 voxel, int gridSize, int texWidth) {
  int sliceSize = gridSize * gridSize;
  int slicesPerRow = texWidth / gridSize;
  int sliceIndex = voxel.z;
  int sliceRow = sliceIndex / slicesPerRow;
  int sliceCol = sliceIndex % slicesPerRow;
  
  int baseX = sliceCol * gridSize + voxel.x;
  int baseY = sliceRow * gridSize + voxel.y;
  return ivec2(baseX, baseY);
}

void main() {
  int index = int(a_particleIndex);
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

  // Map to 3D voxel grid
  vec3 norm = (pos.xyz - u_worldMin) / (u_worldMax - u_worldMin);
  norm = clamp(norm, vec3(0.0), vec3(1.0 - (1.0 / u_gridSize)));
  ivec3 voxelCoord = ivec3(floor(norm * u_gridSize));
  
  // Convert 3D voxel to 2D texture coordinate
  ivec2 texCoord = voxelToTexCoord(voxelCoord, int(u_gridSize), u_textureWidth);
  vec2 texelCenter = (vec2(texCoord) + 0.5) / float(u_textureWidth);
  vec2 clip = texelCenter * 2.0 - 1.0;

  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = 1.0;

  // Weighted sum including Z coordinate now!
  v_particleData = vec4(pos.xyz * mass, mass);
}`;
