#version 300 es
precision highp float;
precision highp int;

in float a_index; // per-instance particle index
uniform sampler2D u_positions;
uniform vec2 u_texSize; // width,height
uniform mat4 u_projectionView;

out vec3 v_color;

vec2 indexToUV(int index) {
    int W = int(u_texSize.x);
    int x = index % W;
    int y = index / W;
    return (vec2(float(x) + 0.5, float(y) + 0.5) ) / u_texSize;
}

void main() {
    int idx = int(a_index + 0.5);
    vec2 uv = indexToUV(idx);
    vec3 pos = texture(u_positions, uv).xyz;

    gl_Position = u_projectionView * vec4(pos, 1.0);
    gl_PointSize = 2.0; // tweakable or per-instance attribute
    v_color = vec3(1.0, 1.0, 1.0);
}
