export default `#version 300 es
precision highp float;

uniform sampler2D u_previousLevel;
uniform int u_prevGridSize;      // Grid size of previous level (cubic)
uniform int u_currGridSize;      // Grid size of current level (cubic)
uniform int u_textureWidth;      // Width of texture for coordinate mapping

out vec4 fragColor;

// Map 3D voxel coordinates to 2D texture coordinates
ivec2 voxelToTexCoord(ivec3 voxel, int gridSize, int texWidth) {
  int slicesPerRow = texWidth / gridSize;
  int sliceIndex = voxel.z;
  int sliceRow = sliceIndex / slicesPerRow;
  int sliceCol = sliceIndex % slicesPerRow;
  
  int baseX = sliceCol * gridSize + voxel.x;
  int baseY = sliceRow * gridSize + voxel.y;
  return ivec2(baseX, baseY);
}

// Map 2D texture coordinate back to 3D voxel
ivec3 texCoordToVoxel(ivec2 texCoord, int gridSize, int texWidth) {
  int slicesPerRow = texWidth / gridSize;
  int sliceCol = texCoord.x / gridSize;
  int sliceRow = texCoord.y / gridSize;
  int sliceIndex = sliceRow * slicesPerRow + sliceCol;
  
  int voxelX = texCoord.x % gridSize;
  int voxelY = texCoord.y % gridSize;
  
  return ivec3(voxelX, voxelY, sliceIndex);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  
  // Convert output 2D coord to 3D voxel
  ivec3 voxel = texCoordToVoxel(coord, u_currGridSize, u_textureWidth);
  
  // Read 2×2×2 block from previous level (8 children)
  ivec3 childBase = voxel * 2;
  vec4 aggregate = vec4(0.0);
  
  for (int dz = 0; dz < 2; dz++) {
    for (int dy = 0; dy < 2; dy++) {
      for (int dx = 0; dx < 2; dx++) {
        ivec3 childVoxel = childBase + ivec3(dx, dy, dz);
        ivec2 childTexCoord = voxelToTexCoord(childVoxel, u_prevGridSize, u_textureWidth);
        vec4 childData = texelFetch(u_previousLevel, childTexCoord, 0);
        aggregate += childData;
      }
    }
  }
  
  fragColor = aggregate;
}`;
