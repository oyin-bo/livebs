export default `#version 300 es
precision highp float;

uniform sampler2D u_positions;
uniform sampler2D u_velocity;
uniform vec2 u_texSize;
uniform int u_particleCount;
uniform float u_dt;

out vec4 fragColor;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int idx = coord.y * int(u_texSize.x) + coord.x;
  vec4 pos = texelFetch(u_positions, coord, 0);
  if (idx >= u_particleCount) {
    fragColor = pos;
    return;
  }
  vec3 vel = texelFetch(u_velocity, coord, 0).xyz;
  vec3 newPos = pos.xyz + vel * u_dt;
  fragColor = vec4(newPos, pos.w);
}`;
