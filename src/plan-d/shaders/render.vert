#version 300 es
precision mediump float;

// Particle rendering vertex shader
attribute vec3 a_position;
attribute vec3 a_color;

uniform mat4 u_modelViewMatrix;
uniform mat4 u_projectionMatrix;
uniform float u_pointSize;

varying vec3 v_color;

void main() {
  vec4 mvPosition = u_modelViewMatrix * vec4(a_position, 1.0);
  gl_Position = u_projectionMatrix * mvPosition;
  
  // Size based on distance
  float distance = length(mvPosition.xyz);
  gl_PointSize = u_pointSize / distance;
  
  v_color = a_color;
}