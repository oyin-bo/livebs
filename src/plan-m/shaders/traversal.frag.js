export default `#version 300 es
precision highp float;

uniform sampler2D u_particlePositions;
uniform sampler2D u_quadtreeLevel0;
uniform sampler2D u_quadtreeLevel1;
uniform sampler2D u_quadtreeLevel2;
uniform sampler2D u_quadtreeLevel3;
uniform sampler2D u_quadtreeLevel4;
uniform sampler2D u_quadtreeLevel5;
uniform sampler2D u_quadtreeLevel6;
uniform float u_theta;
uniform int u_numLevels;
uniform float u_cellSizes[8];
uniform vec2 u_texSize;
uniform int u_particleCount;
uniform vec3 u_worldMin;
uniform vec3 u_worldMax;
uniform float u_softening;
uniform float u_G;
uniform int u_textureWidth;      // For 3D→2D mapping

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

vec4 sampleLevel(int level, ivec3 voxel, int gridSize) {
  ivec2 texCoord = voxelToTexCoord(voxel, gridSize, u_textureWidth);
  if (level == 0) { return texelFetch(u_quadtreeLevel0, texCoord, 0); }
  else if (level == 1) { return texelFetch(u_quadtreeLevel1, texCoord, 0); }
  else if (level == 2) { return texelFetch(u_quadtreeLevel2, texCoord, 0); }
  else if (level == 3) { return texelFetch(u_quadtreeLevel3, texCoord, 0); }
  else if (level == 4) { return texelFetch(u_quadtreeLevel4, texCoord, 0); }
  else if (level == 5) { return texelFetch(u_quadtreeLevel5, texCoord, 0); }
  else if (level == 6) { return texelFetch(u_quadtreeLevel6, texCoord, 0); }
  else { return vec4(0.0); }
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

  // 3D Barnes-Hut octree traversal
  for (int level = min(u_numLevels - 1, 7); level >= 0; level--) {
    int gridSize = int(pow(2.0, float(6 - level))); // L0=64, L1=32, ..., L6=1
    float cellSize = u_cellSizes[level];

    // Find which voxel contains this particle
    vec3 norm = (myPos - u_worldMin) / worldExtent;
    norm = clamp(norm, vec3(0.0), vec3(1.0 - (1.0 / float(gridSize))));
    ivec3 myVoxel = ivec3(floor(norm * float(gridSize)));

    // Adaptive neighborhood size for octree (reduced for performance)
    // L6(1³)=1 cell, L5(2³)=1, L4(4³)=1, L3(8³)=1, L2(16³)=1, L1(32³)=1, L0(64³)=1
    int R = 1;
    
    for (int dz = -R; dz <= R; dz++) {
      for (int dy = -R; dy <= R; dy++) {
        for (int dx = -R; dx <= R; dx++) {
          ivec3 voxel = myVoxel + ivec3(dx, dy, dz);
          
          // Bounds check
          if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
              voxel.x >= gridSize || voxel.y >= gridSize || voxel.z >= gridSize) { continue; }
          
          // Skip own voxel
          if (dx == 0 && dy == 0 && dz == 0) { continue; }
          
          vec4 nodeData = sampleLevel(level, voxel, gridSize);
          float m = nodeData.a;
          if (m <= 0.0) { continue; }
          
          // Extract 3D center of mass
          vec3 com = nodeData.rgb / max(m, 1e-6);
          vec3 dir = com - myPos;
          float d = length(dir);
          if (d < eps) { continue; }
          
          float s = cellSize;
          
          // Barnes-Hut criterion: if s/d < theta, accept approximation
          if ((s / max(d, eps)) < u_theta) {
            float inv = 1.0 / (d * d + eps*eps);
            totalForce += normalize(dir) * m * inv;
          }
        }
      }
    }
  }

  fragColor = vec4(totalForce * u_G, 0.0);
}`;
