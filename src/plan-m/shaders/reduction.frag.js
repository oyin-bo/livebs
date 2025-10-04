export default `#version 300 es
precision highp float;

uniform sampler2D u_previousLevel;

out vec4 fragColor;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  
  // Read 2x2 block from previous level
  ivec2 childBase = coord * 2;
  vec4 child00 = texelFetch(u_previousLevel, childBase + ivec2(0,0), 0);
  vec4 child01 = texelFetch(u_previousLevel, childBase + ivec2(0,1), 0);
  vec4 child10 = texelFetch(u_previousLevel, childBase + ivec2(1,0), 0);
  vec4 child11 = texelFetch(u_previousLevel, childBase + ivec2(1,1), 0);
  
  // Aggregate: sum all components
  vec4 aggregate = child00 + child01 + child10 + child11;
  
  fragColor = aggregate;
}`;
