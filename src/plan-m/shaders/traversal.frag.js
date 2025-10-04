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
uniform sampler2D u_quadtreeLevel7;
uniform float u_theta;
uniform int u_numLevels;
uniform float u_cellSizes[8];
uniform vec2 u_texSize;
uniform int u_particleCount;
uniform vec2 u_worldMin;
uniform vec2 u_worldMax;
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

ivec2 getLevelSize(int level) {
  if (level == 0) { return textureSize(u_quadtreeLevel0, 0); }
  else if (level == 1) { return textureSize(u_quadtreeLevel1, 0); }
  else if (level == 2) { return textureSize(u_quadtreeLevel2, 0); }
  else if (level == 3) { return textureSize(u_quadtreeLevel3, 0); }
  else if (level == 4) { return textureSize(u_quadtreeLevel4, 0); }
  else if (level == 5) { return textureSize(u_quadtreeLevel5, 0); }
  else if (level == 6) { return textureSize(u_quadtreeLevel6, 0); }
  else if (level == 7) { return textureSize(u_quadtreeLevel7, 0); }
  else { return ivec2(1); }
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

  vec2 worldExtent = u_worldMax - u_worldMin;
  float eps = max(u_softening, 1e-6);

  for (int level = min(u_numLevels - 1, 7); level >= 0; level--) {
    ivec2 levelSize = getLevelSize(level);
    float cellSize = u_cellSizes[level];

    // Special case: root level, sample the only node
    if (levelSize.x == 1 && levelSize.y == 1) {
      vec4 root = sampleLevel(level, ivec2(0));
      float massSum = root.b;
      if (massSum > 0.0) {
        vec2 com = root.rg / max(massSum, 1e-6);
        float d = length(com - myPos.xy);
        float s = cellSize;
        if ((s / max(d, eps)) < u_theta) {
          vec2 dir = com - myPos.xy;
          float inv = 1.0 / (d * d + eps);
          totalForce.xy += normalize(dir) * massSum * inv;
        }
      }
      continue;
    }

    // Find my node coordinate at this level
    vec2 norm = (myPos.xy - u_worldMin) / worldExtent;
    ivec2 myNode = ivec2(floor(clamp(norm, vec2(0.0), vec2(1.0 - (1.0 / vec2(levelSize)))) * vec2(levelSize)));

    // Sample a 1-ring neighborhood to gather far contributions
    const int R = 1;
    for (int dy = -R; dy <= R; dy++) {
      for (int dx = -R; dx <= R; dx++) {
        if (dx == 0 && dy == 0) { continue; }
        if (max(abs(dx), abs(dy)) != R) { continue; }
        ivec2 n = myNode + ivec2(dx, dy);
        if (n.x < 0 || n.y < 0 || n.x >= levelSize.x || n.y >= levelSize.y) { continue; }
        vec4 nodeData = sampleLevel(level, n);
        float m = nodeData.b;
        if (m <= 0.0) { continue; }
        vec2 com = nodeData.rg / max(m, 1e-6);
        float d = length(com - myPos.xy);
        float s = cellSize;
        if ((s / max(d, eps)) < u_theta) {
          vec2 dir = com - myPos.xy;
          float inv = 1.0 / (d * d + eps*eps);
          totalForce.xy += normalize(dir) * m * inv;
        }
      }
    }
  }

  // Local near-field from L0 small neighborhood (3x3 excluding center) for anisotropy
  {
    ivec2 L0Size = textureSize(u_quadtreeLevel0, 0);
    vec2 norm = (myPos.xy - u_worldMin) / worldExtent;
    ivec2 myL0 = ivec2(floor(clamp(norm, vec2(0.0), vec2(1.0 - (1.0 / vec2(L0Size)))) * vec2(L0Size)));
    const int R0 = 1; // 3x3 neighborhood
    for (int dy = -R0; dy <= R0; dy++) {
      for (int dx = -R0; dx <= R0; dx++) {
        if (dx == 0 && dy == 0) { continue; }
        ivec2 n = myL0 + ivec2(dx, dy);
        if (n.x < 0 || n.y < 0 || n.x >= L0Size.x || n.y >= L0Size.y) { continue; }
        vec4 nodeData = texelFetch(u_quadtreeLevel0, n, 0);
        float m = nodeData.b;
        if (m <= 0.0) { continue; }
        vec2 com = nodeData.rg / max(m, 1e-6);
        vec2 dir = com - myPos.xy;
        float d = length(dir);
        float inv = 1.0 / (d * d + eps*eps);
        totalForce.xy += normalize(dir) * m * inv;
      }
    }
  }

  fragColor = vec4(totalForce * u_G, 0.0);
}`;
