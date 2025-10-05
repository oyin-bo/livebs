export default `#version 300 es
precision highp float;

// Accumulate weighted 3D positions and mass
// Input: vec4(pos.xyz * mass, mass) from vertex shader
in vec4 v_particleData;
out vec4 fragColor;

void main() {
  fragColor = v_particleData;
}`;
