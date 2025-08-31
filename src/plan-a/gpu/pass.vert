#version 300 es
precision highp float;

// Full-screen quad vertex shader for GPGPU passes
// Used by update pass to cover entire texture

in vec2 a_position;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
