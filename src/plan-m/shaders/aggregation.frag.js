export default `#version 300 es
precision highp float;

in vec4 v_particleData;
out vec4 fragColor;

void main() {
  fragColor = v_particleData;
}`;
