export default `#version 300 es
precision highp float;

uniform sampler2D u_previousLevel;
uniform float u_gridSize;       // current level grid size (e.g., 32 for L1)
uniform float u_slicesPerRow;   // slices per row for current level (e.g., 4 for L1)

out vec4 fragColor;

// Convert 3D voxel coordinate to 2D texture coordinate
ivec2 voxelToTexel(ivec3 voxelCoord, float gridSize, float slicesPerRow) {
  int vx = voxelCoord.x;
  int vy = voxelCoord.y;
  int vz = voxelCoord.z;
  
  // Which Z-slice?
  int sliceIndex = vz;
  int sliceRow = sliceIndex / int(slicesPerRow);
  int sliceCol = sliceIndex - sliceRow * int(slicesPerRow);  // mod without float conversion
  
  // Texel position
  int texelX = sliceCol * int(gridSize) + vx;
  int texelY = sliceRow * int(gridSize) + vy;
  
  return ivec2(texelX, texelY);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  
  // Reverse the voxelToTexel mapping to get parent voxel coordinate
  int gridSizeInt = int(u_gridSize);
  int slicesPerRowInt = int(u_slicesPerRow);
  
  int sliceCol = coord.x / gridSizeInt;
  int sliceRow = coord.y / gridSizeInt;
  int sliceIndex = sliceRow * slicesPerRowInt + sliceCol;
  
  int vx = coord.x - sliceCol * gridSizeInt;
  int vy = coord.y - sliceRow * gridSizeInt;
  int vz = sliceIndex;
  
  ivec3 parentVoxel = ivec3(vx, vy, vz);
  
  // Parent grid is at half resolution in previous level
  float childGridSize = u_gridSize * 2.0;
  float childSlicesPerRow = u_slicesPerRow * 2.0;
  
  // 8 children: 2x2x2 cube
  ivec3 childBase = parentVoxel * 2;
  vec4 child000 = texelFetch(u_previousLevel, voxelToTexel(childBase + ivec3(0,0,0), childGridSize, childSlicesPerRow), 0);
  vec4 child001 = texelFetch(u_previousLevel, voxelToTexel(childBase + ivec3(0,0,1), childGridSize, childSlicesPerRow), 0);
  vec4 child010 = texelFetch(u_previousLevel, voxelToTexel(childBase + ivec3(0,1,0), childGridSize, childSlicesPerRow), 0);
  vec4 child011 = texelFetch(u_previousLevel, voxelToTexel(childBase + ivec3(0,1,1), childGridSize, childSlicesPerRow), 0);
  vec4 child100 = texelFetch(u_previousLevel, voxelToTexel(childBase + ivec3(1,0,0), childGridSize, childSlicesPerRow), 0);
  vec4 child101 = texelFetch(u_previousLevel, voxelToTexel(childBase + ivec3(1,0,1), childGridSize, childSlicesPerRow), 0);
  vec4 child110 = texelFetch(u_previousLevel, voxelToTexel(childBase + ivec3(1,1,0), childGridSize, childSlicesPerRow), 0);
  vec4 child111 = texelFetch(u_previousLevel, voxelToTexel(childBase + ivec3(1,1,1), childGridSize, childSlicesPerRow), 0);
  
  // Aggregate: sum all 8 children
  vec4 aggregate = child000 + child001 + child010 + child011 + child100 + child101 + child110 + child111;
  
  fragColor = aggregate;
}`;
