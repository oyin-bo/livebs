#version 300 es
precision mediump float;

// Particle rendering fragment shader with importance-based coloring
varying vec3 v_color;

out vec4 fragColor;

void main() {
  // Create circular points
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);
  
  // Soft circular falloff
  float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
  
  // Color with brightness variation
  vec3 color = v_color * (0.8 + 0.2 * (1.0 - dist));
  
  fragColor = vec4(color, alpha);
}