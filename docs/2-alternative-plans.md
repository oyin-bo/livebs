# Practical Implementation Plans for Large-Scale Social Graph Physics

*Concrete technical approaches with specific building blocks and data flows*

## Executive Summary

This document presents four distinct technical approaches to solving the core bandwidth problem in large-scale social graph physics simulation: **reducing the 56M+ neighbor reads per frame** that occur when simulating 1M+ particles with typical social graph connectivity.

### Plan Overview

**Plan A: Texture-Based Brute Force with Stochastic Sampling**
- *Approach*: Keep simple texture-based storage, reduce reads by only checking 25% of neighbors per frame
- *Complexity*: Low - extends existing WebGL2 fragment shader patterns
- *Performance*: 4× bandwidth reduction, targets 1M particles
- *Risk*: Low technical risk, some visual jitter from sampling

**Plan B: Compute Shader with Shared Memory Optimization** 
- *Approach*: Use WebGL2 compute shaders with workgroup memory to batch neighbor reads efficiently
- *Complexity*: Medium - requires compute shader expertise and CSR data structures
- *Performance*: Optimal memory access patterns, targets 1.5M+ particles
- *Risk*: Medium - compute shader browser support and debugging complexity

**Plan C: Tile-Based Spatial Culling with Explicit Spatial Grid**
- *Approach*: Spatial grid reduces neighbor checks from O(N×56) to O(N×8-12) by only checking nearby cells
- *Complexity*: Medium - requires spatial data structure management and GPU sorting
- *Performance*: 5-7× computational reduction, targets 2M+ particles  
- *Risk*: Medium - complex to tune cell size and handle dynamic particle distributions

**Plan D: Multi-Resolution Force Calculation**
- *Approach*: Different force fidelity based on particle importance (screen size, motion, user focus)
- *Complexity*: High - requires importance classification, multiple force kernels, and quality management
- *Performance*: Adaptive scaling, targets 3M+ particles with quality graceful degradation
- *Risk*: High - complex to validate quality and avoid visual artifacts

**Plan M: "The Menace" — GPU-side Dynamic Quadtree**
- *Approach*: Build a GPU-resident quadtree as a stack of textures with iterative reduction, then traverse it per-particle (Barnes–Hut style)
- *Complexity*: Very High — requires compute/image atomics, careful numeric design and debugging
- *Performance*: O(N log N) asymptotics; targets 2M+ particles on capable hardware
- *Risk*: High — heavy device/extension requirements and high implementation complexity

### Strategic Positioning

- **Plan A** is the **safe fallback** - proven techniques with incremental optimization
- **Plan B** is the **performance sweet spot** - modern GPU capabilities without excessive complexity  
- **Plan C** is the **scaling champion** - best computational complexity for uniform distributions
- **Plan D** is the **quality optimizer** - maintains visual fidelity while scaling to massive sizes
 - **Plan M** is the **research/scaling path** - maximal asymptotic scaling if device features present; keep fallbacks for broad compatibility

---

## Plan A: Texture-Based Brute Force with Stochastic Sampling
*Real solution to the bandwidth problem: don't read all neighbors every frame*

### Technical Implementation

#### Core Data Structures
```glsl
// Particle state (RGBA32F textures)
uniform sampler2D u_positions;    // xyz = position, w = mass
uniform sampler2D u_velocities;   // xyz = velocity, w = damping
uniform sampler2D u_temp_forces;  // xyz = accumulated force, w = unused

// Neighbor lists (RGBA8 texture, packed)
uniform usampler2D u_neighbors;   // r,g,b = 24-bit neighbor index, a = 8-bit weight
uniform sampler2D u_neighbor_counts; // r = actual neighbor count for this particle
```

#### Stochastic Sampling Engine
```glsl
// Fragment shader - one fragment per particle
void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint particleId = coord.y * textureWidth + coord.x;
    
    // Read my state
    vec4 myPos = texelFetch(u_positions, coord, 0);
    vec3 totalForce = vec3(0.0);
    
    // Get my neighbor count
    float neighborCount = texelFetch(u_neighbor_counts, coord, 0).r;
    uint maxNeighbors = uint(neighborCount);
    
    // Stochastic sampling: only check 25% of neighbors this frame
    float samplingRate = 0.25;
    uint samplesToCheck = max(1u, uint(float(maxNeighbors) * samplingRate));
    
    // Generate frame-stable random offset
    uint frameOffset = hash(particleId + u_frameNumber) % maxNeighbors;
    
    for (uint i = 0u; i < samplesToCheck; i++) {
        uint neighborIndex = (frameOffset + i) % maxNeighbors;
        
        // Read neighbor data
        ivec2 neighborCoord = ivec2(neighborIndex % NEIGHBOR_TEX_WIDTH, 
                                   particleId * MAX_NEIGHBORS_PER_ROW + neighborIndex / NEIGHBOR_TEX_WIDTH);
        uvec4 neighborData = texelFetch(u_neighbors, neighborCoord, 0);
        
        // Unpack neighbor ID and weight
        uint neighborId = (neighborData.r << 16u) | (neighborData.g << 8u) | neighborData.b;
        float weight = float(neighborData.a) / 255.0;
        
        // Read neighbor position
        ivec2 neighborPosCoord = ivec2(neighborId % textureWidth, neighborId / textureWidth);
        vec3 neighborPos = texelFetch(u_positions, neighborPosCoord, 0).xyz;
        
        // Calculate force (compensate for sampling)
        vec3 force = calculateSocialForce(myPos.xyz, neighborPos, weight) / samplingRate;
        totalForce += force;
    }
    
    gl_FragColor = vec4(totalForce, 0.0);
}
```

#### Memory Layout Specifics
- **Particle textures**: 1024x1024 RGBA32F = 1M particles, 64MB per texture
- **Neighbor texture**: Pack 3 neighbors per texel in RGBA format
- **CSR-style layout**: Each particle gets fixed rows in neighbor texture

#### Integration System
```glsl
// Second pass: integrate forces
void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    
    vec4 currentPos = texelFetch(u_positions, coord, 0);
    vec4 currentVel = texelFetch(u_velocities, coord, 0);
    vec3 force = texelFetch(u_temp_forces, coord, 0).xyz;
    
    // Verlet integration
    vec3 acceleration = force / currentPos.w; // mass in w component
    vec3 newVel = currentVel.xyz + acceleration * u_deltaTime;
    vec3 newPos = currentPos.xyz + newVel * u_deltaTime;
    
    // Write back
    gl_FragColor = vec4(newPos, currentPos.w);
}
```

### Practical Building Blocks

#### 1. Neighbor List Uploader
```javascript
class NeighborListManager {
    constructor(gl, maxParticles, maxNeighborsPerParticle) {
        this.neighborsPerTexel = 3; // Pack 3 neighbors in RGBA8
        this.texWidth = Math.ceil(Math.sqrt(maxParticles * maxNeighborsPerParticle / this.neighborsPerTexel));
        this.texHeight = this.texWidth;
        
        this.neighborTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.neighborTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8UI, this.texWidth, this.texHeight, 0, 
                     gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, null);
    }
    
    uploadNeighborList(particleId, neighbors) {
        // neighbors = [{id: 12345, weight: 0.8}, ...]
        const data = new Uint8Array(neighbors.length * 4);
        
        for (let i = 0; i < neighbors.length; i++) {
            const neighborId = neighbors[i].id;
            const weight = Math.floor(neighbors[i].weight * 255);
            
            // Pack 24-bit neighbor ID into RGB
            data[i * 4 + 0] = (neighborId >> 16) & 0xFF;
            data[i * 4 + 1] = (neighborId >> 8) & 0xFF;
            data[i * 4 + 2] = neighborId & 0xFF;
            data[i * 4 + 3] = weight;
        }
        
        // Calculate texture coordinates for this particle's neighbor block
        const startRow = Math.floor(particleId * this.maxNeighborsPerParticle / this.texWidth);
        const startCol = (particleId * this.maxNeighborsPerParticle) % this.texWidth;
        
        gl.texSubImage2D(gl.TEXTURE_2D, 0, startCol, startRow, 
                        neighbors.length, 1, gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, data);
    }
}
```

#### 2. Performance Monitor
```javascript
class PerformanceMonitor {
    constructor() {
        this.frameTimes = new Float32Array(60);
        this.frameIndex = 0;
        this.samplingRate = 0.25;
    }
    
    update(deltaTime) {
        this.frameTimes[this.frameIndex] = deltaTime;
        this.frameIndex = (this.frameIndex + 1) % 60;
        
        // Adjust sampling rate based on performance
        const avgFrameTime = this.frameTimes.reduce((a, b) => a + b) / 60;
        
        if (avgFrameTime > 16.67) { // Over 60fps
            this.samplingRate = Math.max(0.1, this.samplingRate - 0.01);
        } else if (avgFrameTime < 13.33) { // Under 75fps
            this.samplingRate = Math.min(1.0, this.samplingRate + 0.01);
        }
    }
}
```

### Specific Problems & Solutions

**Problem**: Neighbor texture becomes enormous (1M particles × 56 neighbors = 56M entries)
**Solution**: Use CSR layout with row pointers, allocate only actual edges

**Problem**: Random access patterns kill GPU cache
**Solution**: Sort neighbors by spatial proximity during upload, use coherent access patterns

**Problem**: Stochastic sampling creates visual jitter
**Solution**: Use temporal blue noise patterns, ensure each neighbor gets sampled within N frames

---

## Plan B: Compute Shader with Shared Memory Optimization
*WebGL2 compute shaders with workgroup-optimized neighbor processing*

### Technical Implementation

#### Compute Shader Architecture
```glsl
#version 310 es

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Storage buffers (much cleaner than textures)
layout(std430, binding = 0) restrict readonly buffer ParticlePositions {
    vec4 positions[]; // xyz = position, w = mass
};

layout(std430, binding = 1) restrict readonly buffer ParticleVelocities {
    vec4 velocities[]; // xyz = velocity, w = damping
};

layout(std430, binding = 2) restrict writeonly buffer ParticleForces {
    vec4 forces[]; // xyz = force, w = unused
};

layout(std430, binding = 3) restrict readonly buffer NeighborList {
    uint neighbors[]; // Packed: [count][id0][id1]...[idN]
};

layout(std430, binding = 4) restrict readonly buffer RowPointers {
    uint rowPtrs[]; // CSR-style row pointers
};

// Shared memory for cooperative loading
shared vec4 sharedPositions[64];
shared float sharedMasses[64];

void main() {
    uint particleId = gl_GlobalInvocationID.x;
    if (particleId >= positions.length()) return;
    
    vec4 myPos = positions[particleId];
    vec3 totalForce = vec3(0.0);
    
    // Get my neighbor range
    uint rowStart = rowPtrs[particleId];
    uint rowEnd = rowPtrs[particleId + 1];
    uint neighborCount = rowEnd - rowStart;
    
    // Process neighbors in chunks that fit in shared memory
    for (uint chunkStart = rowStart; chunkStart < rowEnd; chunkStart += 64u) {
        uint chunkSize = min(64u, rowEnd - chunkStart);
        
        // Cooperative loading into shared memory
        uint localId = gl_LocalInvocationID.x;
        if (localId < chunkSize) {
            uint neighborId = neighbors[chunkStart + localId];
            sharedPositions[localId] = positions[neighborId];
            sharedMasses[localId] = positions[neighborId].w;
        }
        
        barrier();
        memoryBarrierShared();
        
        // Now each thread processes the shared data
        for (uint i = 0u; i < chunkSize; i++) {
            vec3 neighborPos = sharedPositions[i].xyz;
            float neighborMass = sharedMasses[i];
            
            vec3 force = calculateForce(myPos.xyz, neighborPos, myPos.w, neighborMass);
            totalForce += force;
        }
        
        barrier();
    }
    
    forces[particleId] = vec4(totalForce, 0.0);
}
```

#### CSR Buffer Management
```javascript
class CSRNeighborStorage {
    constructor(gl, maxParticles, totalEdges) {
        this.gl = gl;
        
        // Create storage buffers
        this.neighborsBuffer = gl.createBuffer();
        this.rowPtrsBuffer = gl.createBuffer();
        
        // Allocate space
        gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, this.neighborsBuffer);
        gl.bufferData(gl.SHADER_STORAGE_BUFFER, totalEdges * 4, gl.DYNAMIC_DRAW);
        
        gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, this.rowPtrsBuffer);
        gl.bufferData(gl.SHADER_STORAGE_BUFFER, (maxParticles + 1) * 4, gl.DYNAMIC_DRAW);
    }
    
    uploadGraph(adjacencyLists) {
        // adjacencyLists[particleId] = [neighborId1, neighborId2, ...]
        
        const neighbors = [];
        const rowPtrs = [0];
        
        for (let i = 0; i < adjacencyLists.length; i++) {
            const list = adjacencyLists[i] || [];
            
            // Add neighbors for this particle
            for (const neighborId of list) {
                neighbors.push(neighborId);
            }
            
            // Record where next particle's neighbors start
            rowPtrs.push(neighbors.length);
        }
        
        // Upload to GPU
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.neighborsBuffer);
        this.gl.bufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, new Uint32Array(neighbors));
        
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.rowPtrsBuffer);
        this.gl.bufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, new Uint32Array(rowPtrs));
    }
}
```

#### Workgroup Size Optimization
```javascript
class ComputeOptimizer {
    constructor(gl) {
        this.gl = gl;
        this.optimalWorkgroupSize = this.findOptimalWorkgroupSize();
    }
    
    findOptimalWorkgroupSize() {
        const sizes = [32, 64, 128, 256];
        const testParticles = 10000;
        let bestSize = 64;
        let bestTime = Infinity;
        
        for (const size of sizes) {
            const time = this.benchmarkWorkgroupSize(size, testParticles);
            if (time < bestTime) {
                bestTime = time;
                bestSize = size;
            }
        }
        
        return bestSize;
    }
    
    benchmarkWorkgroupSize(workgroupSize, particleCount) {
        // Create test compute shader with this workgroup size
        // Run benchmark and measure time
        // Return average execution time
    }
}
```

### Specific Data Flow

#### Frame Processing Pipeline
1. **Upload Phase** (CPU): Update neighbor lists if graph changed
2. **Force Calculation** (GPU Compute): One dispatch call for all particles
3. **Integration** (GPU Compute): Update positions/velocities 
4. **Rendering** (GPU Graphics): Draw particles with instanced rendering

#### Memory Access Pattern
```
Workgroup 0: Particles 0-63
- Each thread loads its own position
- Cooperatively load neighbors 0-63 into shared memory
- Each thread calculates forces from shared neighbors
- Repeat for neighbors 64-127, 128-191, etc.

Workgroup 1: Particles 64-127
- Same pattern, different particle range
```

---

## Plan C: Tile-Based Spatial Culling with Explicit Spatial Grid
*Reduce neighbor checks by only processing nearby spatial regions*

### Technical Implementation

#### Spatial Grid Structure
```glsl
// Spatial grid parameters
uniform float u_cellSize;        // Size of each grid cell
uniform ivec3 u_gridDimensions;  // Number of cells in each dimension
uniform vec3 u_gridOrigin;       // World-space origin of the grid

// Grid data textures
uniform usampler2D u_gridCells;     // Each cell contains particle count + start offset
uniform usampler2D u_gridParticles; // Sorted list of particle IDs by cell
```

#### Particle-to-Grid Assignment
```glsl
// Compute shader: assign particles to grid cells
layout(local_size_x = 64) in;

layout(std430, binding = 0) restrict readonly buffer Positions {
    vec4 positions[];
};

layout(std430, binding = 1) restrict writeonly buffer ParticleCells {
    uint particleCells[]; // Cell ID for each particle
};

void main() {
    uint particleId = gl_GlobalInvocationID.x;
    if (particleId >= positions.length()) return;
    
    vec3 pos = positions[particleId].xyz;
    
    // Calculate grid cell
    ivec3 cellCoord = ivec3((pos - u_gridOrigin) / u_cellSize);
    
    // Clamp to grid bounds
    cellCoord = clamp(cellCoord, ivec3(0), u_gridDimensions - 1);
    
    // Convert 3D cell coordinate to 1D cell ID
    uint cellId = uint(cellCoord.z * u_gridDimensions.x * u_gridDimensions.y + 
                      cellCoord.y * u_gridDimensions.x + 
                      cellCoord.x);
    
    particleCells[particleId] = cellId;
}
```

#### Force Calculation with Spatial Culling
```glsl
// Compute shader: calculate forces using spatial grid
void main() {
    uint particleId = gl_GlobalInvocationID.x;
    
    vec3 myPos = positions[particleId].xyz;
    vec3 totalForce = vec3(0.0);
    
    // Get my grid cell
    ivec3 myCell = ivec3((myPos - u_gridOrigin) / u_cellSize);
    
    // Check neighboring cells (3x3x3 = 27 cells)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                ivec3 neighborCell = myCell + ivec3(dx, dy, dz);
                
                // Skip out-of-bounds cells
                if (any(lessThan(neighborCell, ivec3(0))) || 
                    any(greaterThanEqual(neighborCell, u_gridDimensions))) {
                    continue;
                }
                
                // Process particles in this cell
                totalForce += processCell(particleId, neighborCell);
            }
        }
    }
    
    forces[particleId] = vec4(totalForce, 0.0);
}

vec3 processCell(uint myParticleId, ivec3 cellCoord) {
    // Convert cell coordinate to cell ID
    uint cellId = uint(cellCoord.z * u_gridDimensions.x * u_gridDimensions.y + 
                      cellCoord.y * u_gridDimensions.x + 
                      cellCoord.x);
    
    // Get cell info (count + start index)
    uvec2 cellInfo = texelFetch(u_gridCells, 
                               ivec2(cellId % GRID_TEX_WIDTH, cellId / GRID_TEX_WIDTH), 0).xy;
    uint particleCount = cellInfo.x;
    uint startIndex = cellInfo.y;
    
    vec3 cellForce = vec3(0.0);
    
    // Process all particles in this cell
    for (uint i = 0u; i < particleCount; i++) {
        uint neighborId = texelFetch(u_gridParticles, 
                                   ivec2((startIndex + i) % PARTICLES_TEX_WIDTH, 
                                        (startIndex + i) / PARTICLES_TEX_WIDTH), 0).x;
        
        if (neighborId != myParticleId) {
            vec3 neighborPos = positions[neighborId].xyz;
            cellForce += calculateForce(positions[myParticleId].xyz, neighborPos);
        }
    }
    
    return cellForce;
}
```

#### Grid Construction Pipeline
```javascript
class SpatialGrid {
    constructor(gl, worldBounds, targetCellSize) {
        this.gl = gl;
        this.cellSize = targetCellSize;
        
        // Calculate grid dimensions
        const worldSize = worldBounds.max.subtract(worldBounds.min);
        this.gridDimensions = {
            x: Math.ceil(worldSize.x / targetCellSize),
            y: Math.ceil(worldSize.y / targetCellSize),
            z: Math.ceil(worldSize.z / targetCellSize)
        };
        
        this.totalCells = this.gridDimensions.x * this.gridDimensions.y * this.gridDimensions.z;
        
        // Create buffers
        this.setupBuffers();
        this.setupSortingPipeline();
    }
    
    setupBuffers() {
        // Buffer to store which cell each particle belongs to
        this.particleCellsBuffer = this.gl.createBuffer();
        
        // Buffer for sorted particle indices
        this.sortedParticlesBuffer = this.gl.createBuffer();
        
        // Buffer for cell start indices and counts
        this.cellDataBuffer = this.gl.createBuffer();
    }
    
    updateGrid(particlePositions) {
        // 1. Assign particles to cells
        this.runParticleToCellShader(particlePositions);
        
        // 2. Sort particles by cell ID
        this.sortParticlesByCell();
        
        // 3. Build cell lookup table
        this.buildCellLookupTable();
    }
    
    sortParticlesByCell() {
        // Use GPU radix sort or CPU fallback
        // Sort particle indices by their cell ID
        // This groups particles by spatial locality
    }
    
    buildCellLookupTable() {
        // Scan through sorted particle list
        // For each cell, record start index and count
        // Upload to cell data buffer
    }
}
```

### Practical Performance Analysis

#### Memory Requirements
- **Grid cells**: 32x32x32 = 32K cells × 8 bytes = 256KB
- **Particle cell IDs**: 1M particles × 4 bytes = 4MB
- **Sorted particles**: 1M particles × 4 bytes = 4MB
- **Total overhead**: ~8.3MB (much less than storing full neighbor lists)

#### Computational Complexity
- **Without grid**: O(N×K) where K = average neighbors = 56
- **With grid**: O(N×C) where C = average particles per 27-cell neighborhood
- **Typical improvement**: C ≈ 8-12 for reasonable densities (5-7× speedup)

---

## Plan D: Multi-Resolution Force Calculation
*Different force calculation fidelity based on distance and importance*

### Technical Implementation

#### Particle Classification System
```glsl
// Particle importance levels
#define IMPORTANCE_HIGH 0u     // Full neighbor processing
#define IMPORTANCE_MEDIUM 1u   // Sampled neighbors + approximate far-field
#define IMPORTANCE_LOW 2u      // Approximate forces only

layout(std430, binding = 5) restrict readonly buffer ParticleImportance {
    uint importance[]; // Importance level per particle
};
```

#### Multi-Fidelity Force Kernel
```glsl
void main() {
    uint particleId = gl_GlobalInvocationID.x;
    uint importanceLevel = importance[particleId];
    
    vec3 totalForce = vec3(0.0);
    
    switch (importanceLevel) {
        case IMPORTANCE_HIGH:
            totalForce = calculateHighFidelityForces(particleId);
            break;
        case IMPORTANCE_MEDIUM:
            totalForce = calculateMediumFidelityForces(particleId);
            break;
        case IMPORTANCE_LOW:
            totalForce = calculateLowFidelityForces(particleId);
            break;
    }
    
    forces[particleId] = vec4(totalForce, 0.0);
}

vec3 calculateHighFidelityForces(uint particleId) {
    // Process all social network neighbors
    uint rowStart = rowPtrs[particleId];
    uint rowEnd = rowPtrs[particleId + 1];
    
    vec3 force = vec3(0.0);
    for (uint i = rowStart; i < rowEnd; i++) {
        uint neighborId = neighbors[i];
        force += calculateSocialForce(particleId, neighborId);
    }
    
    return force;
}

vec3 calculateMediumFidelityForces(uint particleId) {
    // Sample 50% of social neighbors + use spatial approximation
    uint rowStart = rowPtrs[particleId];
    uint rowEnd = rowPtrs[particleId + 1];
    uint neighborCount = rowEnd - rowStart;
    
    vec3 socialForce = vec3(0.0);
    uint samplesToCheck = neighborCount / 2u;
    
    for (uint i = 0u; i < samplesToCheck; i++) {
        uint neighborIndex = (hash(particleId + u_frameNumber + i) % neighborCount) + rowStart;
        uint neighborId = neighbors[neighborIndex];
        socialForce += calculateSocialForce(particleId, neighborId) * 2.0; // Compensate for sampling
    }
    
    // Add spatial approximation
    vec3 spatialForce = calculateSpatialApproximation(particleId);
    
    return socialForce + spatialForce;
}

vec3 calculateLowFidelityForces(uint particleId) {
    // Only use precomputed spatial approximations
    return calculateSpatialApproximation(particleId);
}
```

#### Importance Classification System
```javascript
class ParticleImportanceClassifier {
    constructor(gl, maxParticles) {
        this.gl = gl;
        this.importanceBuffer = gl.createBuffer();
        
        gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, this.importanceBuffer);
        gl.bufferData(gl.SHADER_STORAGE_BUFFER, maxParticles * 4, gl.DYNAMIC_DRAW);
    }
    
    updateImportance(particles, camera, userFocus) {
        const importance = new Uint32Array(particles.length);
        
        for (let i = 0; i < particles.length; i++) {
            const particle = particles[i];
            
            // Calculate screen size
            const screenPos = projectToScreen(particle.position, camera);
            const screenSize = calculateScreenSize(particle, camera);
            
            // Calculate distance to user focus point
            const focusDistance = distance(particle.position, userFocus);
            
            // Calculate motion
            const motion = particle.velocity.magnitude();
            
            // Combine factors
            const score = screenSize * (1.0 / Math.max(focusDistance, 1.0)) * motion;
            
            if (score > 0.1) {
                importance[i] = 0; // HIGH
            } else if (score > 0.01) {
                importance[i] = 1; // MEDIUM
            } else {
                importance[i] = 2; // LOW
            }
        }
        
        // Upload to GPU
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.importanceBuffer);
        this.gl.bufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, importance);
    }
}
```

#### Spatial Approximation Precomputation
```javascript
class SpatialApproximationGrid {
    constructor(gl, worldBounds, resolution) {
        this.resolution = resolution;
        this.cellSize = worldBounds.size / resolution;
        
        // Create 3D texture for precomputed forces
        this.forceFieldTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_3D, this.forceFieldTexture);
        gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA16F, 
                     resolution, resolution, resolution, 0, 
                     gl.RGBA, gl.HALF_FLOAT, null);
    }
    
    precomputeForceField(particles) {
        // Compute average forces at grid points
        const forceData = new Float32Array(this.resolution ** 3 * 4);
        
        for (let z = 0; z < this.resolution; z++) {
            for (let y = 0; y < this.resolution; y++) {
                for (let x = 0; x < this.resolution; x++) {
                    const worldPos = this.gridToWorld(x, y, z);
                    const avgForce = this.computeAverageForceAtPosition(worldPos, particles);
                    
                    const index = (z * this.resolution * this.resolution + y * this.resolution + x) * 4;
                    forceData[index + 0] = avgForce.x;
                    forceData[index + 1] = avgForce.y;
                    forceData[index + 2] = avgForce.z;
                    forceData[index + 3] = 0.0;
                }
            }
        }
        
        // Upload to 3D texture
        this.gl.bindTexture(this.gl.TEXTURE_3D, this.forceFieldTexture);
        this.gl.texSubImage3D(this.gl.TEXTURE_3D, 0, 0, 0, 0,
                             this.resolution, this.resolution, this.resolution,
                             this.gl.RGBA, this.gl.HALF_FLOAT, forceData);
    }
}
```

### Performance Targets & Validation

#### Expected Performance Distribution
- **High fidelity**: 10-20% of particles (100K-200K)
- **Medium fidelity**: 30-40% of particles (300K-400K)  
- **Low fidelity**: 40-60% of particles (400K-600K)

#### Validation Criteria
1. **Visual continuity**: No visible popping when importance changes
2. **Force conservation**: Total system energy remains stable
3. **Performance scaling**: Maintains 60fps as particle count increases
4. **Quality metrics**: High-fidelity regions show proper clustering behavior

Each plan provides concrete, implementable solutions to the core bandwidth and memory problems identified in the technical analysis.

---

## Comprehensive Plan Comparison

### Technical Implementation Difficulty

| Aspect | Plan A (Stochastic) | Plan B (Compute) | Plan C (Spatial Grid) | Plan D (Multi-Res) |
|--------|-------------------|------------------|---------------------|-------------------|
| **WebGL2 Knowledge Required** | Basic fragment shaders | Compute shaders + SSBO | Compute + GPU sorting | Advanced compute + 3D textures |
| **Data Structure Complexity** | Simple textures | CSR buffers | Spatial grid + sorting | Multi-level importance + approximation |
| **Shader Complexity** | Low (single fragment shader) | Medium (compute workgroups) | High (multiple compute passes) | Very High (branching kernels) |
| **Debugging Difficulty** | Easy (visual validation) | Medium (buffer inspection) | Hard (spatial coherence) | Very Hard (quality validation) |
| **Time to First Working Demo** | 1-2 weeks | 3-4 weeks | 4-6 weeks | 6-8 weeks |

### Performance Characteristics

| Metric | Plan A | Plan B | Plan C | Plan D |
|--------|--------|--------|--------|--------|
| **Target Particle Count** | 1M | 1.5M | 2M | 3M+ |
| **Memory Bandwidth Reduction** | 4× (sampling) | 8× (shared memory) | 5-7× (spatial culling) | 2-20× (adaptive) |
| **GPU Memory Usage** | High (full textures) | Medium (CSR buffers) | Medium (grid overhead) | Low (importance-based) |
| **CPU Overhead** | Minimal | Low | Medium (grid rebuild) | High (importance calc) |
| **Frame Rate Stability** | Good (consistent load) | Excellent (predictable) | Variable (distribution dependent) | Adaptive (quality scaling) |

### Browser Compatibility & Future Support

| Consideration | Plan A | Plan B | Plan C | Plan D |
|---------------|--------|--------|--------|--------|
| **WebGL2 Support** | Universal ✓ | Universal ✓ | Universal ✓ | Universal ✓ |
| **Compute Shader Requirement** | No | Yes (widely supported) | Yes (widely supported) | Yes (widely supported) |
| **Extension Dependencies** | None | None | None | EXT_color_buffer_float (optional) |
| **Mobile Performance** | Good | Excellent | Poor (memory bandwidth) | Excellent (adaptive) |
| **WebGPU Migration Path** | Hard (texture-dependent) | Easy (similar paradigms) | Medium (spatial structures) | Easy (multi-kernel design) |
| **Long-term Viability** | Limited scaling | Excellent | Good | Excellent |

### Development Risk Assessment

#### Plan A: Texture-Based Stochastic Sampling
**Low Risk ⭐⭐⭐⭐⭐**
- *Pros*: Proven techniques, easy debugging, incremental optimization path
- *Cons*: Limited scaling ceiling, memory inefficiency, visual sampling artifacts
- *Failure Modes*: Texture size limits, sampling jitter, poor cache utilization
- *Mitigation*: Blue noise sampling patterns, temporal coherence, progressive enhancement

#### Plan B: Compute Shader with Shared Memory
**Medium Risk ⭐⭐⭐⭐**
- *Pros*: Modern GPU paradigms, excellent performance potential, clean architecture
- *Cons*: Compute shader complexity, CSR data structure management
- *Failure Modes*: Workgroup size tuning, memory coalescing issues, debugging difficulty
- *Mitigation*: Extensive profiling, fallback to Plan A, workgroup size auto-detection

#### Plan C: Spatial Grid Culling
**Medium-High Risk ⭐⭐⭐**
- *Pros*: Excellent computational complexity, handles clustering well, proven algorithms
- *Cons*: Complex spatial data management, performance depends on distribution uniformity
- *Failure Modes*: Grid cell size tuning, particle clustering hotspots, sorting overhead
- *Mitigation*: Adaptive cell sizing, hybrid spatial-graph approaches, load balancing

#### Plan D: Multi-Resolution Forces
**High Risk ⭐⭐**
- *Pros*: Ultimate scalability, quality-performance trade-offs, user-adaptive
- *Cons*: Complex importance classification, multiple code paths, quality validation
- *Failure Modes*: Importance calculation overhead, visual artifacts at transitions, tuning complexity
- *Mitigation*: Conservative default parameters, extensive quality metrics, user feedback integration

### Recommended Implementation Strategy

#### Phase 1: Foundation (Weeks 1-4)
**Start with Plan A** - Establish working simulation with texture-based storage and stochastic sampling
- Validate core physics and rendering pipeline
- Implement performance monitoring and quality metrics
- Target: 100K particles at 60fps with basic social graph forces

#### Phase 2: Performance Optimization (Weeks 5-8)  
**Implement Plan B** - Migrate to compute shader architecture while maintaining Plan A fallback
- Convert to CSR storage format and compute-based force calculation
- Implement shared memory optimization and workgroup tuning
- Target: 500K-1M particles with improved memory efficiency

#### Phase 3: Scaling Enhancement (Weeks 9-12)
**Add Plan C Elements** - Integrate spatial culling as optional optimization layer
- Implement spatial grid construction and maintenance
- Add adaptive cell sizing and load balancing
- Target: 1.5M+ particles with spatial optimizations

#### Phase 4: Quality Scaling (Weeks 13-16)
**Integrate Plan D Concepts** - Add multi-resolution capabilities for ultimate scaling
- Implement importance classification and adaptive quality
- Add precomputed spatial approximations
- Target: 2M+ particles with quality-performance trade-offs

### Technical Decision Matrix

**Choose Plan A if:**
- Development timeline is tight (< 4 weeks)
- Team has limited WebGL2 compute shader experience  
- Target is primarily desktop browsers with good GPU memory
- Particle count requirements are < 1M

**Choose Plan B if:**
- Performance is critical and development time allows
- Team comfortable with compute shaders and advanced WebGL2
- Target includes mobile devices (excellent compute efficiency)
- Need clean architecture for future WebGPU migration

**Choose Plan C if:**
- Particle distributions are spatially clustered (social graphs typically are)
- Maximum particle count is the primary requirement
- Acceptable to handle complex spatial data structure management
- Performance can vary based on particle distribution

**Choose Plan D if:**
- Ultimate scalability is required (3M+ particles)
- Quality-performance trade-offs are acceptable
- Development team has advanced graphics programming experience
- Long-term maintenance and tuning resources are available

### Future Technology Considerations

**WebGPU Migration Readiness:**
- Plan B: Excellent (compute paradigms directly transfer)
- Plan D: Good (multi-kernel design fits compute model)
- Plan C: Medium (spatial structures need redesign)
- Plan A: Poor (texture-dependent approach becomes obsolete)
 - Plan M: Medium (conceptually fits WebGPU but requires careful mapping of image atomics / texture storage)

**Machine Learning Integration Potential:**
- Plan D: Excellent (importance classification can use ML)
- Plan B: Good (can integrate learned optimizations)
- Plan C: Medium (spatial clustering can benefit from ML)
- Plan A: Poor (limited optimization surface)
 - Plan M: Low-Medium (ML could tune theta/tiling/quantization but direct benefit limited)

**Maintenance and Evolution:**
- Plan B: Excellent (clean separation of concerns)
- Plan A: Good (simple to understand and modify)
- Plan C: Fair (spatial logic complexity)
- Plan D: Poor (multiple interacting systems)
 - Plan M: Medium (high initial complexity, but clearly separated build/traverse stages enable targeted maintenance)

## Plan M: "The Menace" — GPU-side Dynamic Quadtree (Texture-based)
*A full GPU-resident quadtree built as textures with iterative reduction and GPU-only traversal (Barnes–Hut style).* 

### Summary
Plan M is the research proposition described in `3-the-menace-high-level.md`. It treats the quadtree as a stack of textures: each level stores node aggregates (mass, center-of-mass numerator, particle count, metadata). The pipeline has two main stages: (1) Build (particle→lowest-level texel aggregation, then iterative 2×2 reductions up the pyramid), and (2) Traverse (per-particle Barnes–Hut traversal using texel fetches across levels). When the full set of GPU features (compute shaders, imageLoad/imageStore, and atomic ops) is available, this offers a largely GPU-only O(N log N) solution with excellent locality for highly-parallel traversal. When features are absent, there are explicit fallbacks described below.

### Key Capabilities Required (ranked)
- Compute shader dispatch (GLSL ES 3.10 or WebGPU compute) — strongly preferred
- imageLoad/imageStore or equivalent random-access texture reads/writes
- atomic operations on integers (imageAtomicAdd / atomicAdd) — for counters
- floating-point atomic add (imageAtomicAdd on float) or alternative fixed-point integer atomics — strongly desirable for single-pass CoM sums
- EXT_color_buffer_float (render-to-float textures) for storing float aggregates
- Reasonable MAX_TEXTURE_SIZE and sufficient VRAM to hold the quadtree levels

If compute/image atomic primitives are not present the Menace devolves into multi-pass or CPU-assisted alternatives (see Fallbacks section).

### Data Layouts and Formats
- Lowest-level texture (L0): power-of-two resolution covering simulation domain. Map particle space → texel coordinates (x,y). Each texel represents a leaf node that can aggregate many particles.
- Per-level textures: level L has resolution size_L = size_{L-1} / 2 (integer division), each texel is a node.
- Node payload (one texel): choose one of the safe formats depending on extension availability:
    - Preferred: RGBA32F (vec4) where: R = sum_x (COM numerator x), G = sum_y (COM numerator y), B = massSum, A = particleCount (stored as float) — requires float atomics/blend or careful reduction.
    - Conservative (portable): use two textures: (1) RGBA32F for float sums (sum_x, sum_y, mass, 0) written via reduction passes; (2) R32UI (integer) for particle counts updated with integer atomics. Or store fixed-point integers in R32I for sums and use integer atomics (Q16.16) to accumulate numerators.
    - Minimal (older devices): store per-particle slot texture (particleIndex→slot) and run a deterministic ping-pong reduction (no atomics) — more memory and passes.

Indexing convention: use texelFetch with ivec2 coordinates. For addressing children during reduction, child texel coords at level L map to a 2×2 block at level L-1 (x*2 + {0,1}, y*2 + {0,1}).

### Build Stage (detailed)
1) Initial aggregation (particles → L0):
     - Ideal path (atomics available): each compute thread for a particle computes its L0 texel coords and performs:
             - imageAtomicAdd(countTexel, 1) on an integer count
             - imageAtomicAdd(massTexel, mass) on float if supported OR imageAtomicAdd(fixedPointSumTexel, massFixed) on integer fixed-point
             - imageAtomicAdd(sumXTexel, pos.x * mass) etc.
         This yields a correct single-pass aggregation.

     - Conservative path (no float atomics): use integer atomics with fixed-point packing (Q16.16 or Q8.24) to accumulate sum_x, sum_y and mass as integers; convert to float in reduction pass. This requires quantization but avoids floating atomic requirements.

     - No-atomics path: write each particle into a unique per-particle slot (particleId → slot texel), then run a parallel segmented reduction (ping-pong) to accumulate sums per L0 node. This uses more memory (O(N)) and multiple passes but is robust on legacy WebGL2.

2) Iterative reduction passes (L0 → L1 → L2 ... → root):
     - For each level k produce level k+1 by launching one compute dispatch sized to the level k+1 texture.
     - Each invocation reads the 2×2 child texels, sums mass and weighted sums, sums counts, and writes parent texel. This uses ordinary float ops and imageStore; no atomics are needed during reduction since each parent texel is written by a single thread.
     - Number of passes = number of levels ≈ log2(lowestResolution).

3) Finalization: root texel contains global mass and CoM (normalize sums by count/mass as appropriate). Optionally compact or normalize data into a compact read-only texture for traversal.

### Traversal Stage (detailed)
- Approach: per-particle compute kernel implements stackless Barnes–Hut traversal reading from level textures with texelFetch. There are multiple traversal styles: recursive emulation with an explicit stack in shared memory or a breadth-first acceptance loop scanning nodes by index. For WebGL/WebGPU, explicit stack in Shader Storage (or per-thread fixed-size array) works.

Traversal pseudocode (stackless, iterative):
```
// Per-particle
vec3 myPos = ...; float theta = u_theta;
float totalForce = vec3(0.0);
int stack[STACK_SIZE]; stackTop = 0; push(rootIndex);
while (stackTop > 0) {
    int node = pop();
    NodeData n = fetchNode(node, level); // use texelFetch across levels
    float dist = distance(myPos, n.com);
    if ((nodeSize / dist) < theta || isLeaf(node)) {
         // approximate
         totalForce += approxForce(myPos, n.com, n.mass);
    } else {
         // descend: push children (if any)
         push(child0); push(child1); push(child2); push(child3);
    }
}
```

Notes:
- Node addressing can be level-indexed: for each level keep a baseOffset and a 2D-to-1D mapping; texel coords are computed from node index.
- Fixed stack size needs to be tuned; alternatively use depth-limited traversal to bound work per particle.

### Shader considerations and portability
- Float atomic add is not widely available in WebGL2; design for two implementation modes:
    1. Full-atomic mode: uses imageAtomicAdd on float (or imageAtomicAdd on float via EXT), simplest and fastest.
    2. Fixed-point integer atomics: pack sums into 32-bit integer fixed-point and use imageAtomicAdd (portable). Convert to float in reduction passes. Choose Q format to balance range/precision.
- Avoid dynamic shader code-gen. Keep traversal logic static and parameterized by uniforms (theta, levelCount, baseOffsets).
- Debugging: expose intermediate level textures by rendering them to screen; color-code counts, mass, and com to validate.

### Memory and Bandwidth Estimate (approximate)
- For N ≈ 1,048,576 particles (L0 = 1024×1024 ≈ 1M texels):
    - RGBA32F per level = 16 bytes/texel. L0 = ~16MB. The sum of geometric series for levels ≈ 4/3 * L0 = ~21.3MB for the entire pyramid.
    - If you need separate integer count texture (R32UI) add ~4MB.
    - Fixed-point integer approach: two R32I textures instead of float sums reduces float usage but increases conversions.
- Bandwidth: Initial aggregation (single-pass) reads N particle positions and writes N leaf updates (atomics). Reduction passes read/write decreasing amounts: total read/write ≈ O(N).

### Edge cases and failure modes
- Limited MAX_TEXTURE_SIZE: If the domain requires a larger lowest-level texture than the device allows, switch to tiled builds (process spatial tiles) or fall back to spatial-grid Plan C.
- No float atomics: use fixed-point integer atomics or per-particle-slot reduction (higher cost).
- VRAM shortage: reduce L0 resolution (coarser quadtree), use RGBA16F where safe, or fall back to Plan B/A depending on device.

### Fallbacks (concrete choices)
1. Atomics present (best): use single-pass aggregation with atomic integer counters + float/fixed-point sum atomics → run iterative reductions → traverse.
2. No float atomics but integer atomics present: accumulate integer fixed-point sums with imageAtomicAdd on R32I, normalize in reduction passes.
3. No atomics: per-particle-slot + ping-pong reduction (deterministic) or GPU sort+segmented reduce — solid but memory/passes heavy. Implement this as a guarded code path chosen by the runtime probe.
4. If compute shaders unavailable: emulate reductions in fragment pipelines (multiple render passes) or revert to Plan A (stochastic) for compatibility.

### Integration & Runtime Policy
- Implement a `deviceProbe` that checks: compute shader dispatch support, imageLoad/imageStore, image atomic ops, EXT_color_buffer_float, MAX_TEXTURE_SIZE. Return a policy enum: `menace:full`, `menace:fixed-int`, `menace:reduction`, `menace:emulate-frag`, `menace:disabled`.
- At runtime choose Menace if `menace:full` or `menace:fixed-int`. Otherwise fall back to Plan B (compute CSR) or Plan C depending on device memory/bandwidth.

### Phased Implementation Plan (practical)
1. Prototype reduction-only pipeline on desktop with full atomics: implement particle→L0 atomic aggregation + iterative reductions + simple traversal for correctness validation.
2. Add integer fixed-point path and validate numeric error bounds vs float baseline.
3. Implement non-atomic per-particle-slot reduction (ping-pong) to support low-end devices and compare perf vs atomic path.
4. Add runtime probe and policy chooser; expose debug visualizations for each level.

### Pros / Cons (brief)
- Pros: Fully GPU-resident quadtree, strong asymptotic scaling (O(N log N)), compact traversal kernel, good cache locality once tree is built.
- Cons: High API/feature requirements, complex to implement and debug, sensitive to atomic/texture feature availability, and memory cost for dense L0 grids.

### Validation targets
- Correctness: aggregated mass and COM at each node must match CPU reference within quantization error.
- Performance: for devices with atomics expect better throughput than Plan B at large N; fallback paths should keep performance within 2× of Plan B.

---