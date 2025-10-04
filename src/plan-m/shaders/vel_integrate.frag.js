export default `#version 300 es
precision highp float;

uniform sampler2D u_velocity;
uniform sampler2D u_force;
uniform vec2 u_texSize;
uniform int u_particleCount;
uniform float u_dt;
uniform float u_damping;
uniform float u_maxSpeed;
uniform float u_maxAccel;

out vec4 fragColor;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  int idx = coord.y * int(u_texSize.x) + coord.x;
  vec4 vel = texelFetch(u_velocity, coord, 0);
  if (idx >= u_particleCount) {
    fragColor = vel;
    return;
  }
  vec3 force = texelFetch(u_force, coord, 0).xyz;
  float fmag = length(force);
  if (fmag > u_maxAccel) {
    force = force * (u_maxAccel / max(fmag, 1e-6));
  }
  vec3 newVel = vel.xyz + force * u_dt;
  newVel *= (1.0 - u_damping);
  float vmag = length(newVel);
  if (vmag > u_maxSpeed) {
    newVel = newVel * (u_maxSpeed / vmag);
  }
  fragColor = vec4(newVel, 0.0);
}`;
