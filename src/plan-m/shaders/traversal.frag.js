export default `#version 300 es
precision highp float;

// 3D isotropic octree traversal with Barnes-Hut

uniform sampler2D u_particlePositions;
uniform sampler2D u_quadtreeLevel0;
uniform sampler2D u_quadtreeLevel1;
uniform sampler2D u_quadtreeLevel2;
uniform sampler2D u_quadtreeLevel3;
uniform sampler2D u_quadtreeLevel4;
uniform sampler2D u_quadtreeLevel5;
uniform sampler2D u_quadtreeLevel6;
uniform sampler2D u_quadtreeLevel7;
uniform float u_theta;
uniform int u_numLevels;
uniform float u_cellSizes[8];
uniform float u_gridSizes[8];         // voxel grid sizes per level
uniform float u_slicesPerRow[8];      // slices per row per level
uniform vec2 u_texSize;
uniform int u_particleCount;
uniform vec3 u_worldMin;
uniform vec3 u_worldMax;
uniform float u_softening;
uniform float u_G;

out vec4 fragColor;

vec4 sampleLevel(int level, ivec2 coord) {
  if (level == 0) { return texelFetch(u_quadtreeLevel0, coord, 0); }
  else if (level == 1) { return texelFetch(u_quadtreeLevel1, coord, 0); }
  else if (level == 2) { return texelFetch(u_quadtreeLevel2, coord, 0); }
  else if (level == 3) { return texelFetch(u_quadtreeLevel3, coord, 0); }
  else if (level == 4) { return texelFetch(u_quadtreeLevel4, coord, 0); }
  else if (level == 5) { return texelFetch(u_quadtreeLevel5, coord, 0); }
  else if (level == 6) { return texelFetch(u_quadtreeLevel6, coord, 0); }
  else if (level == 7) { return texelFetch(u_quadtreeLevel7, coord, 0); }
  else { return vec4(0.0); }
}

// Convert 3D voxel coordinate to 2D texture coordinate
ivec2 voxelToTexel(ivec3 voxelCoord, float gridSize, float slicesPerRow) {
  int vx = voxelCoord.x;
  int vy = voxelCoord.y;
  int vz = voxelCoord.z;
  
  // Which Z-slice?
  int sliceIndex = vz;
  int sliceRow = sliceIndex / int(slicesPerRow);
  int sliceCol = sliceIndex - sliceRow * int(slicesPerRow);
  
  // Texel position
  int texelX = sliceCol * int(gridSize) + vx;
  int texelY = sliceRow * int(gridSize) + vy;
  
  return ivec2(texelX, texelY);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int myIndex = coord.y * int(u_texSize.x) + coord.x;
  if (myIndex >= u_particleCount) {
    fragColor = vec4(0.0);
    return;
  }

  vec2 myUV = (vec2(coord) + 0.5) / u_texSize;
  vec3 myPos = texture(u_particlePositions, myUV).xyz;
  vec3 totalForce = vec3(0.0);

  vec3 worldExtent = u_worldMax - u_worldMin;
  float eps = max(u_softening, 1e-6);

  // Traverse octree levels from coarsest to finest
  for (int level = min(u_numLevels - 1, 7); level >= 0; level--) {
    float gridSize = u_gridSizes[level];
    float slicesPerRow = u_slicesPerRow[level];
    float cellSize = u_cellSizes[level];

    // Special case: root level (1×1×1 voxel)
    if (gridSize == 1.0) {
      vec4 root = sampleLevel(level, ivec2(0, 0));
      float massSum = root.a;
      if (massSum > 0.0) {
        vec3 com = root.rgb / max(massSum, 1e-6);
        vec3 delta = com - myPos;
        float d = length(delta);
        float s = cellSize;
        if ((s / max(d, eps)) < u_theta) {
          // Approximate: use COM
          float inv = 1.0 / (d * d + eps * eps);
          totalForce += normalize(delta) * massSum * inv;
        }
      }
      continue;
    }

    // Find my voxel coordinate at this level
    vec3 norm = (myPos - u_worldMin) / worldExtent;
    norm = clamp(norm, vec3(0.0), vec3(1.0 - (1.0 / gridSize)));
    ivec3 myVoxel = ivec3(floor(norm * gridSize));

    // Sample full 3D neighborhood (27 voxels - 1 center = 26 voxels)
    // Use all neighbors to avoid anisotropic sampling bias
    const int R = 1;
    for (int dz = -R; dz <= R; dz++) {
      for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
          if (dx == 0 && dy == 0 && dz == 0) { continue; } // Skip center
          
          ivec3 neighborVoxel = myVoxel + ivec3(dx, dy, dz);
          
          // Bounds check
          if (neighborVoxel.x < 0 || neighborVoxel.y < 0 || neighborVoxel.z < 0 ||
              neighborVoxel.x >= int(gridSize) || neighborVoxel.y >= int(gridSize) || neighborVoxel.z >= int(gridSize)) {
            continue;
          }
          
          ivec2 texCoord = voxelToTexel(neighborVoxel, gridSize, slicesPerRow);
          vec4 nodeData = sampleLevel(level, texCoord);
          float m = nodeData.a;
          if (m <= 0.0) { continue; }
          
          // Sub-voxel COM for smoother force field
          vec3 com = nodeData.rgb / max(m, 1e-6);
          vec3 delta = com - myPos;
          float d = length(delta);
          float s = cellSize;
          
          // Distance-based theta criterion (isotropic)
          if ((s / max(d, eps)) < u_theta) {
            // Approximate: use COM with isotropic softening
            float denom = d * d + eps * eps;
            float inv = 1.0 / denom;
            // Use delta directly (already has direction) instead of normalize + multiply
            totalForce += (delta / max(d, eps)) * m * inv;
          }
        }
      }
    }
  }

  // Near-field: sample L0 local neighborhood (3×3×3 - 1 = 26 voxels)
  // with extended radius for smoother transitions
  {
    float gridSize = u_gridSizes[0];
    float slicesPerRow = u_slicesPerRow[0];
    vec3 norm = (myPos - u_worldMin) / worldExtent;
    norm = clamp(norm, vec3(0.0), vec3(1.0 - (1.0 / gridSize)));
    ivec3 myL0Voxel = ivec3(floor(norm * gridSize));
    
    // Sample 5×5×5 neighborhood at L0 for smoother near-field
    const int R0 = 2;
    for (int dz = -R0; dz <= R0; dz++) {
      for (int dy = -R0; dy <= R0; dy++) {
        for (int dx = -R0; dx <= R0; dx++) {
          if (dx == 0 && dy == 0 && dz == 0) { continue; }
          
          // Skip far corners to maintain isotropy
          int manhattan = abs(dx) + abs(dy) + abs(dz);
          if (manhattan > 4) { continue; }
          
          ivec3 neighborVoxel = myL0Voxel + ivec3(dx, dy, dz);
          if (neighborVoxel.x < 0 || neighborVoxel.y < 0 || neighborVoxel.z < 0 ||
              neighborVoxel.x >= int(gridSize) || neighborVoxel.y >= int(gridSize) || neighborVoxel.z >= int(gridSize)) {
            continue;
          }
          
          ivec2 texCoord = voxelToTexel(neighborVoxel, gridSize, slicesPerRow);
          vec4 nodeData = sampleLevel(0, texCoord);
          float m = nodeData.a;
          if (m <= 0.0) { continue; }
          
          // Sub-voxel COM for high-resolution force field
          vec3 com = nodeData.rgb / max(m, 1e-6);
          vec3 delta = com - myPos;
          float d = length(delta);
          float denom = d * d + eps * eps;
          float inv = 1.0 / denom;
          // Isotropic force calculation
          totalForce += (delta / max(d, eps)) * m * inv;
        }
      }
    }
  }

  fragColor = vec4(totalForce * u_G, 0.0);
}`;
