#version 300 es
precision mediump float;

// Point rendering fragment shader
// Simple colored points with some visual appeal

in vec3 v_color;

out vec4 fragColor;

void main() {
  // Calculate distance from center of point
  vec2 coord = gl_PointCoord - vec2(0.5);
  float dist = length(coord);
  
  // Create a circular point with soft edges
  float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
  
  // Apply color with some brightness variation
  vec3 color = v_color * (0.8 + 0.2 * (1.0 - dist));
  
  fragColor = vec4(color, alpha);
}
