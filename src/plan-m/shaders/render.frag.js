export default `#version 300 es
precision mediump float;

in vec3 v_color;
out vec4 fragColor;

void main() {
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);
  float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
  vec3 color = v_color * (0.8 + 0.2 * (1.0 - dist));
  fragColor = vec4(color, alpha);
}`;
