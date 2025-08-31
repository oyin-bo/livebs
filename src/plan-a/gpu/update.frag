#version 300 es
precision highp float;
precision highp int;

// Update fragment shader header and helper functions for Plan A
// This shader expects to be run to produce a 1:1 texture of size texWidth x texHeight
// Each fragment corresponds to one particle. The implementation below is complete
// in terms of uniforms and utilities; the force model loop is a clear, copyable
// starting point that samples a stochastic subset of particles per-frame.

uniform sampler2D u_positions;    // RGBA32F texel: xyz = position, w = mass
uniform sampler2D u_velocities;   // RGBA32F texel: xyz = velocity, w = damping (optional)
uniform float u_dt;
uniform float u_samplingFraction; // 0.0 .. 1.0
uniform vec2 u_texSize;          // width,height of particle texture
uniform int u_particleCount;
uniform float u_time;
uniform uint u_seed;

out vec4 outColor; // new position (xyz) + mass

// Convert ivec2 coord -> flat index
int coordToIndex(ivec2 coord) {
    return coord.y * int(u_texSize.x) + coord.x;
}

vec2 indexToUV(int index) {
    int W = int(u_texSize.x);
    int H = int(u_texSize.y);
    int x = index % W;
    int y = index / W;
    // center of texel
    return (vec2(float(x) + 0.5, float(y) + 0.5) ) / u_texSize;
}

// Simple LFSR-ish PRNG returning float in [0,1).
// Not cryptographically strong; deterministic per-fragment when seeded.
uint lfsr32(uint x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

float randFloat(inout uint state) {
    state = lfsr32(state);
    return float(state & 0x00ffffffu) / float(0x01000000u);
}

// Pairwise force (repulsive soft-sphere example)
vec3 pairForce(vec3 aPos, vec3 bPos) {
    vec3 dir = bPos - aPos;
    float d2 = dot(dir, dir) + 1e-6;
    float inv = 1.0 / sqrt(d2);
    float strength = 0.01; // tweakable
    // soft repulsion
    return -strength * dir * inv / d2;
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy) - ivec2(1,1) + ivec2(0); // gl_FragCoord is 1-based in some systems; safe to use -0.5 in UV
    // Better: compute uv directly
    vec2 uv = (gl_FragCoord.xy - vec2(0.5)) / u_texSize;

    vec4 selfT = texture(u_positions, uv);
    vec3 pos = selfT.xyz;
    float mass = selfT.w;

    // fetch velocity if available
    vec3 vel = texture(u_velocities, uv).xyz;

    // RNG seeded by particle index + frame seed
    int myIndex = coordToIndex(ivec2(int(gl_FragCoord.x)-1, int(gl_FragCoord.y)-1));
    uint state = uint(myIndex) + u_seed + uint(int(mod(floor(u_time*60.0), 1024.0)));

    // Determine number of samples
    int maxSamples = int(clamp(float(u_particleCount) * u_samplingFraction, 1.0, 512.0));

    vec3 totalForce = vec3(0.0);
    for (int s = 0; s < 512; ++s) {
        if (s >= maxSamples) break;
        float r = randFloat(state);
        int j = int(floor(r * float(u_particleCount)));
        vec2 uvj = indexToUV(j);
        vec3 q = texture(u_positions, uvj).xyz;
        totalForce += pairForce(pos, q);
    }

    // integrate (semi-implicit Euler)
    vec3 acc = totalForce / max(mass, 1e-6);
    vel += acc * u_dt;
    float maxSpeed = 10.0;
    if (length(vel) > maxSpeed) vel = normalize(vel) * maxSpeed;
    vec3 newPos = pos + vel * u_dt;

    // world bounds clamp (simple)
    // TODO: expose uniforms for bounds
    for (int i=0;i<3;++i) {
        if (newPos[i] < -10.0) newPos[i] = -10.0;
        if (newPos[i] > 10.0) newPos[i] = 10.0;
    }

    outColor = vec4(newPos, mass);
}
