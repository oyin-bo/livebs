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

**Plan C: Fragment Shader Spatial Grid with Additive Blending**
- *Approach*: Spatial grid reduces neighbor checks from O(N×56) to O(N×8-12) using fragment shader grid construction and additive blending
- *Complexity*: Medium - requires spatial data structure management using texture-based storage
- *Performance*: 5-7× computational reduction, targets 2M+ particles  
- *Risk*: Medium - complex to tune cell size and handle dynamic particle distributions

**Plan D: Fragment Shader Multi-Resolution Force Calculation**
- *Approach*: Different force fidelity based on particle importance using multiple fragment shader programs and texture sampling
- *Complexity*: Medium-High - requires importance classification and quality management using fragment shaders
- *Performance*: Adaptive scaling, targets 3M+ particles with quality graceful degradation
- *Risk*: Medium-High - complex to validate quality and avoid visual artifacts

**Plan M: "The Menace" — GPU-side Dynamic Quadtree**
- *Approach*: Build a GPU-resident quadtree as a stack of textures using fragment shader reductions, then traverse it per-particle (Barnes–Hut style)
- *Complexity*: High — requires multi-pass texture reductions and careful numeric design
- *Performance*: O(N log N) asymptotics; targets 2M+ particles with standard WebGL2
- *Risk*: Medium — complex multi-pass pipeline but uses only standard WebGL2 features

### Strategic Positioning

- **Plan A** is the **safe fallback** - proven techniques with incremental optimization
- **Plan C** is the **scaling champion** - best computational complexity for uniform distributions using fragment shaders
- **Plan D** is the **quality optimizer** - maintains visual fidelity while scaling to massive sizes using adaptive fragment shader techniques
- **Plan M** is the **research/scaling path** - maximal asymptotic scaling using standard WebGL2 fragment shaders with multi-pass reductions for broad compatibility

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

## Plan C: Fragment Shader Spatial Grid with Additive Blending
*Reduce neighbor checks by building spatial grids using fragment shaders and additive blending*

### Technical Implementation

#### Spatial Grid Structure
```glsl
// Spatial grid parameters
uniform float u_cellSize;        // Size of each grid cell
uniform ivec3 u_gridDimensions;  // Number of cells in each dimension
uniform vec3 u_gridOrigin;       // World-space origin of the grid

// Grid data textures
uniform sampler2D u_gridCounts;     // Each cell contains particle count
uniform sampler2D u_gridParticles;  // Sorted list of particle IDs by cell
uniform sampler2D u_cellOffsets;    // Start index for each cell's particles
```

#### Particle-to-Grid Assignment (Fragment Shader)
```glsl
// Vertex shader - render particles as points to grid assignment texture
attribute vec3 a_position;
attribute float a_particleId;

uniform mat4 u_worldToGrid;
uniform vec2 u_gridTextureSize;

varying vec4 v_gridData; // Cell assignment data

void main() {
    vec3 pos = a_position;
    
    // Calculate grid cell
    ivec3 cellCoord = ivec3((pos - u_gridOrigin) / u_cellSize);
    cellCoord = clamp(cellCoord, ivec3(0), u_gridDimensions - 1);
    
    // Convert 3D cell coordinate to 2D texture coordinate
    uint cellId = uint(cellCoord.z * u_gridDimensions.x * u_gridDimensions.y + 
                      cellCoord.y * u_gridDimensions.x + 
                      cellCoord.x);
    
    // Map cell ID to texture coordinates
    vec2 texCoord = vec2(cellId % uint(u_gridTextureSize.x), 
                        cellId / uint(u_gridTextureSize.x)) / u_gridTextureSize;
    
    gl_Position = vec4(texCoord * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = 1.0;
    
    v_gridData = vec4(a_particleId, 1.0, 0.0, 0.0); // particle ID + count
}
```

```glsl
// Fragment shader - accumulate particles into grid cells
varying vec4 v_gridData;

void main() {
    // Use additive blending to count particles per cell
    gl_FragColor = v_gridData; // Will be additively blended
}
```

#### Grid Construction Pipeline (Fragment Shader Based)
```javascript
class FragmentSpatialGrid {
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
        this.gridTextureSize = Math.ceil(Math.sqrt(this.totalCells));
        
        this.setupTextures();
        this.setupShaders();
    }
    
    setupTextures() {
        // Grid cell count texture
        this.gridCountsTexture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.gridCountsTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA32F, 
                          this.gridTextureSize, this.gridTextureSize, 0, 
                          this.gl.RGBA, this.gl.FLOAT, null);
        
        // Particle assignment texture
        this.particleAssignmentTexture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.particleAssignmentTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA32F, 
                          this.gridTextureSize, this.gridTextureSize, 0, 
                          this.gl.RGBA, this.gl.FLOAT, null);
    }
    
    updateGrid(particlePositions) {
        // 1. Clear grid textures
        this.clearGridTextures();
        
        // 2. Assign particles to cells using additive blending
        this.assignParticlesToCells(particlePositions);
        
        // 3. Sort particles by cell (CPU fallback for now)
        this.sortParticlesByCell(particlePositions);
        
        // 4. Build cell lookup texture
        this.buildCellLookupTexture();
    }
    
    assignParticlesToCells(positions) {
        this.gl.useProgram(this.gridAssignmentProgram);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.gridFramebuffer);
        
        // Enable additive blending for counting
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.ONE, this.gl.ONE);
        
        // Upload particle data and render as points
        this.uploadParticleData(positions);
        this.gl.drawArrays(this.gl.POINTS, 0, positions.length);
        
        this.gl.disable(this.gl.BLEND);
    }
}
```

#### Force Calculation with Spatial Culling (Fragment Shader)
```glsl
// Fragment shader - one fragment per particle for force calculation
uniform sampler2D u_particlePositions;
uniform sampler2D u_gridCounts;      // Cell particle counts
uniform sampler2D u_gridParticles;   // Sorted particle IDs by cell
uniform sampler2D u_cellOffsets;     // Start indices for each cell

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint particleId = uint(coord.y * textureWidth + coord.x);
    
    vec3 myPos = texelFetch(u_particlePositions, coord, 0).xyz;
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
    
    gl_FragColor = vec4(totalForce, 0.0);
}

vec3 processCell(uint myParticleId, ivec3 cellCoord) {
    // Convert cell coordinate to cell ID
    uint cellId = uint(cellCoord.z * u_gridDimensions.x * u_gridDimensions.y + 
                      cellCoord.y * u_gridDimensions.x + 
                      cellCoord.x);
    
    // Map cell ID to texture coordinates
    ivec2 cellTexCoord = ivec2(cellId % gridTextureWidth, cellId / gridTextureWidth);
    
    // Get cell info (count + start index)
    vec4 cellData = texelFetch(u_gridCounts, cellTexCoord, 0);
    uint particleCount = uint(cellData.r);
    uint startIndex = uint(cellData.g);
    
    vec3 cellForce = vec3(0.0);
    
    // Process all particles in this cell
    for (uint i = 0u; i < particleCount; i++) {
        ivec2 particleTexCoord = ivec2((startIndex + i) % particleTextureWidth, 
                                      (startIndex + i) / particleTextureWidth);
        uint neighborId = uint(texelFetch(u_gridParticles, particleTexCoord, 0).r);
        
        if (neighborId != myParticleId) {
            ivec2 neighborPosCoord = ivec2(neighborId % textureWidth, neighborId / textureWidth);
            vec3 neighborPos = texelFetch(u_particlePositions, neighborPosCoord, 0).xyz;
            cellForce += calculateSpatialForce(myPos, neighborPos);
        }
    }
    
    return cellForce;
}
```

#### Grid Construction Pipeline
```javascript
class FragmentSpatialGrid {
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
        this.gridTextureSize = Math.ceil(Math.sqrt(this.totalCells));
        
        this.setupTextures();
        this.setupFramebuffers();
    }
    
    setupTextures() {
        // Grid cell data texture (counts and offsets)
        this.gridDataTexture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.gridDataTexture);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA32F, 
                          this.gridTextureSize, this.gridTextureSize, 0, 
                          this.gl.RGBA, this.gl.FLOAT, null);
        
        // Sorted particle list texture
        this.sortedParticlesTexture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.sortedParticlesTexture);
        
        const maxParticles = 1000000; // Adjust as needed
        const particleTextureSize = Math.ceil(Math.sqrt(maxParticles));
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.R32F, 
                          particleTextureSize, particleTextureSize, 0, 
                          this.gl.RED, this.gl.FLOAT, null);
    }
    
    updateGrid(particlePositions) {
        // 1. Clear grid textures
        this.clearGridTextures();
        
        // 2. Count particles per cell using additive blending
        this.countParticlesPerCell(particlePositions);
        
        // 3. Sort particles by cell ID (CPU fallback or GPU sort)
        const sortedData = this.sortParticlesByCell(particlePositions);
        
        // 4. Upload sorted data and build lookup texture
        this.uploadSortedParticles(sortedData);
        this.buildCellLookupTexture(sortedData.cellOffsets);
    }
    
    countParticlesPerCell(positions) {
        this.gl.useProgram(this.cellCountProgram);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.gridCountFramebuffer);
        
        // Enable additive blending
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.ONE, this.gl.ONE);
        
        // Render particles as points to count them per cell
        this.uploadParticlePositions(positions);
        this.gl.drawArrays(this.gl.POINTS, 0, positions.length);
        
        this.gl.disable(this.gl.BLEND);
    }
    
    sortParticlesByCell(positions) {
        // CPU-based sorting for now (can be replaced with GPU sort later)
        const particleCells = [];
        
        // Assign each particle to a cell
        for (let i = 0; i < positions.length; i++) {
            const pos = positions[i];
            const cellCoord = {
                x: Math.max(0, Math.min(this.gridDimensions.x - 1, 
                    Math.floor((pos.x - this.gridOrigin.x) / this.cellSize))),
                y: Math.max(0, Math.min(this.gridDimensions.y - 1, 
                    Math.floor((pos.y - this.gridOrigin.y) / this.cellSize))),
                z: Math.max(0, Math.min(this.gridDimensions.z - 1, 
                    Math.floor((pos.z - this.gridOrigin.z) / this.cellSize)))
            };
            
            const cellId = cellCoord.z * this.gridDimensions.x * this.gridDimensions.y + 
                          cellCoord.y * this.gridDimensions.x + cellCoord.x;
                          
            particleCells.push({particleId: i, cellId: cellId});
        }
        
        // Sort by cell ID
        particleCells.sort((a, b) => a.cellId - b.cellId);
        
        // Build cell offset table
        const cellOffsets = new Array(this.totalCells).fill(0);
        let currentCell = -1;
        let currentOffset = 0;
        
        for (let i = 0; i < particleCells.length; i++) {
            const cellId = particleCells[i].cellId;
            if (cellId !== currentCell) {
                currentCell = cellId;
                cellOffsets[cellId] = i;
            }
        }
        
        return {
            sortedParticles: particleCells.map(p => p.particleId),
            cellOffsets: cellOffsets
        };
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

## Plan D: Fragment Shader Multi-Resolution Force Calculation
*Different force fidelity based on particle importance using fragment shader programs and texture sampling*

### Technical Implementation

#### Particle Classification System
```glsl
// Particle importance levels stored in texture
uniform sampler2D u_particleImportance; // R channel contains importance level

// Importance level constants
#define IMPORTANCE_HIGH 0.0     // Full neighbor processing
#define IMPORTANCE_MEDIUM 1.0   // Sampled neighbors + approximate far-field
#define IMPORTANCE_LOW 2.0      // Approximate forces only
```

#### Multi-Fidelity Force Kernel (Fragment Shader)
```glsl
// Fragment shader - one fragment per particle
uniform sampler2D u_particlePositions;
uniform sampler2D u_particleImportance;
uniform sampler2D u_neighborData;        // Neighbor lists
uniform sampler3D u_spatialApproximation; // 3D force field approximation

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    uint particleId = uint(coord.y * textureWidth + coord.x);
    
    vec3 myPos = texelFetch(u_particlePositions, coord, 0).xyz;
    float importanceLevel = texelFetch(u_particleImportance, coord, 0).r;
    
    vec3 totalForce = vec3(0.0);
    
    if (importanceLevel < 0.5) {
        // HIGH IMPORTANCE: Full neighbor processing
        totalForce = calculateHighFidelityForces(particleId, myPos);
    } else if (importanceLevel < 1.5) {
        // MEDIUM IMPORTANCE: Sampled neighbors + spatial approximation
        totalForce = calculateMediumFidelityForces(particleId, myPos);
    } else {
        // LOW IMPORTANCE: Spatial approximation only
        totalForce = calculateLowFidelityForces(myPos);
    }
    
    gl_FragColor = vec4(totalForce, 0.0);
}

vec3 calculateHighFidelityForces(uint particleId, vec3 myPos) {
    // Process all social network neighbors
    vec3 force = vec3(0.0);
    
    // Read neighbor count
    ivec2 countCoord = ivec2(particleId % textureWidth, particleId / textureWidth);
    float neighborCount = texelFetch(u_neighborCounts, countCoord, 0).r;
    uint maxNeighbors = uint(neighborCount);
    
    // Process all neighbors
    for (uint i = 0u; i < maxNeighbors; i++) {
        ivec2 neighborCoord = ivec2((particleId * MAX_NEIGHBORS + i) % NEIGHBOR_TEX_WIDTH, 
                                   (particleId * MAX_NEIGHBORS + i) / NEIGHBOR_TEX_WIDTH);
        uvec4 neighborData = texelFetch(u_neighborData, neighborCoord, 0);
        
        uint neighborId = (neighborData.r << 16u) | (neighborData.g << 8u) | neighborData.b;
        float weight = float(neighborData.a) / 255.0;
        
        ivec2 neighborPosCoord = ivec2(neighborId % textureWidth, neighborId / textureWidth);
        vec3 neighborPos = texelFetch(u_particlePositions, neighborPosCoord, 0).xyz;
        
        force += calculateSocialForce(myPos, neighborPos, weight);
    }
    
    return force;
}

vec3 calculateMediumFidelityForces(uint particleId, vec3 myPos) {
    // Sample 50% of social neighbors
    vec3 socialForce = vec3(0.0);
    
    ivec2 countCoord = ivec2(particleId % textureWidth, particleId / textureWidth);
    float neighborCount = texelFetch(u_neighborCounts, countCoord, 0).r;
    uint maxNeighbors = uint(neighborCount);
    uint samplesToCheck = maxNeighbors / 2u;
    
    for (uint i = 0u; i < samplesToCheck; i++) {
        uint neighborIndex = (hash(particleId + u_frameNumber + i) % maxNeighbors);
        
        ivec2 neighborCoord = ivec2((particleId * MAX_NEIGHBORS + neighborIndex) % NEIGHBOR_TEX_WIDTH, 
                                   (particleId * MAX_NEIGHBORS + neighborIndex) / NEIGHBOR_TEX_WIDTH);
        uvec4 neighborData = texelFetch(u_neighborData, neighborCoord, 0);
        
        uint neighborId = (neighborData.r << 16u) | (neighborData.g << 8u) | neighborData.b;
        float weight = float(neighborData.a) / 255.0;
        
        ivec2 neighborPosCoord = ivec2(neighborId % textureWidth, neighborId / textureWidth);
        vec3 neighborPos = texelFetch(u_particlePositions, neighborPosCoord, 0).xyz;
        
        socialForce += calculateSocialForce(myPos, neighborPos, weight) * 2.0; // Compensate for sampling
    }
    
    // Add spatial approximation
    vec3 spatialForce = calculateSpatialApproximation(myPos);
    
    return socialForce + spatialForce;
}

vec3 calculateLowFidelityForces(vec3 myPos) {
    // Only use precomputed spatial approximations
    return calculateSpatialApproximation(myPos);
}

vec3 calculateSpatialApproximation(vec3 pos) {
    // Sample from 3D force field texture
    vec3 normalizedPos = (pos - u_worldBounds.min) / (u_worldBounds.max - u_worldBounds.min);
    normalizedPos = clamp(normalizedPos, vec3(0.0), vec3(1.0));
    
    return texture(u_spatialApproximation, normalizedPos).xyz;
}
```

#### Importance Classification System (CPU-based)
```javascript
class ParticleImportanceClassifier {
    constructor(gl, maxParticles) {
        this.gl = gl;
        
        // Create importance texture
        this.importanceTexture = gl.createTexture();
        const textureSize = Math.ceil(Math.sqrt(maxParticles));
        
        gl.bindTexture(gl.TEXTURE_2D, this.importanceTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, textureSize, textureSize, 0, 
                     gl.RED, gl.FLOAT, null);
        
        this.textureSize = textureSize;
        this.maxParticles = maxParticles;
    }
    
    updateImportance(particles, camera, userFocus) {
        const importance = new Float32Array(this.textureSize * this.textureSize);
        
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
                importance[i] = 0.0; // HIGH
            } else if (score > 0.01) {
                importance[i] = 1.0; // MEDIUM
            } else {
                importance[i] = 2.0; // LOW
            }
        }
        
        // Upload to GPU texture
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.importanceTexture);
        this.gl.texSubImage2D(this.gl.TEXTURE_2D, 0, 0, 0, 
                             this.textureSize, this.textureSize,
                             this.gl.RED, this.gl.FLOAT, importance);
    }
    
    // Alternative: Fragment shader-based importance classification
    updateImportanceGPU(particles, camera, userFocus) {
        // Use fragment shader to compute importance scores
        this.gl.useProgram(this.importanceProgram);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.importanceFramebuffer);
        
        // Set uniforms for camera and focus
        this.gl.uniform3fv(this.gl.getUniformLocation(this.importanceProgram, 'u_cameraPos'), 
                          camera.position.toArray());
        this.gl.uniform3fv(this.gl.getUniformLocation(this.importanceProgram, 'u_userFocus'), 
                          userFocus.toArray());
        
        // Render full-screen quad to compute importance
        this.renderFullScreenQuad();
    }
}
```

#### Spatial Approximation Precomputation (Fragment Shader Based)
```javascript
class SpatialApproximationGrid {
    constructor(gl, worldBounds, resolution) {
        this.gl = gl;
        this.resolution = resolution;
        this.worldBounds = worldBounds;
        this.cellSize = worldBounds.size / resolution;
        
        // Create 3D texture for precomputed forces
        this.forceFieldTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_3D, this.forceFieldTexture);
        gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA16F, 
                     resolution, resolution, resolution, 0, 
                     gl.RGBA, gl.HALF_FLOAT, null);
        
        this.setupPrecomputationShaders();
    }
    
    setupPrecomputationShaders() {
        // Fragment shader for precomputing spatial force field
        const fragmentSource = `
            precision highp float;
            
            uniform sampler2D u_particlePositions;
            uniform sampler2D u_particleMasses;
            uniform vec3 u_worldMin;
            uniform vec3 u_worldSize;
            uniform float u_gridResolution;
            uniform int u_numParticles;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / vec2(u_gridResolution);
                
                // Convert 2D fragment coord to 3D grid position
                float layer = floor(gl_FragCoord.z + 0.5) / u_gridResolution;
                vec3 gridPos = vec3(uv.x, uv.y, layer);
                vec3 worldPos = u_worldMin + gridPos * u_worldSize;
                
                vec3 totalForce = vec3(0.0);
                
                // Sample forces from all particles (or a representative subset)
                int sampleCount = min(u_numParticles, 1000); // Limit for performance
                for (int i = 0; i < sampleCount; i++) {
                    ivec2 particleCoord = ivec2(i % textureWidth, i / textureWidth);
                    vec3 particlePos = texelFetch(u_particlePositions, particleCoord, 0).xyz;
                    float mass = texelFetch(u_particleMasses, particleCoord, 0).r;
                    
                    vec3 direction = worldPos - particlePos;
                    float distance = length(direction) + 0.1; // Avoid division by zero
                    vec3 force = normalize(direction) * mass / (distance * distance);
                    
                    totalForce += force;
                }
                
                gl_FragColor = vec4(totalForce, 1.0);
            }
        `;
        
        this.precomputeProgram = this.createShaderProgram(vertexSource, fragmentSource);
    }
    
    precomputeForceField(particles) {
        // Method 1: Fragment shader precomputation (for dynamic updates)
        this.precomputeForceFieldGPU(particles);
        
        // Method 2: CPU precomputation (for static approximations)
        // this.precomputeForceFieldCPU(particles);
    }
    
    precomputeForceFieldGPU(particles) {
        this.gl.useProgram(this.precomputeProgram);
        
        // Upload particle data
        this.uploadParticleData(particles);
        
        // Render to 3D texture layers
        for (let layer = 0; layer < this.resolution; layer++) {
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.layerFramebuffers[layer]);
            this.gl.uniform1f(this.gl.getUniformLocation(this.precomputeProgram, 'u_layer'), layer);
            
            // Render full-screen quad for this layer
            this.renderFullScreenQuad();
        }
    }
    
    precomputeForceFieldCPU(particles) {
        // CPU fallback for platforms without 3D texture support
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
    
    computeAverageForceAtPosition(worldPos, particles) {
        let totalForce = {x: 0, y: 0, z: 0};
        let sampleCount = Math.min(particles.length, 500); // Limit for performance
        
        for (let i = 0; i < sampleCount; i++) {
            const particle = particles[i];
            const direction = {
                x: worldPos.x - particle.position.x,
                y: worldPos.y - particle.position.y,
                z: worldPos.z - particle.position.z
            };
            
            const distance = Math.sqrt(direction.x ** 2 + direction.y ** 2 + direction.z ** 2) + 0.1;
            const forceMagnitude = particle.mass / (distance * distance);
            
            totalForce.x += (direction.x / distance) * forceMagnitude;
            totalForce.y += (direction.y / distance) * forceMagnitude;
            totalForce.z += (direction.z / distance) * forceMagnitude;
        }
        
        return totalForce;
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

| Aspect | Plan A (Stochastic) | Plan C (Spatial Grid) | Plan D (Multi-Res) | Plan M (The Menace) |
|--------|-------------------|---------------------|-------------------|---------------------|
| **WebGL2 Knowledge Required** | Basic fragment shaders | Advanced fragment shaders + additive blending | Advanced fragment shaders + 3D textures | Advanced fragment shaders + MRT + texture pyramids |
| **Data Structure Complexity** | Simple textures | Spatial grid + texture-based sorting | Multi-level importance + approximation textures | Texture pyramid + reduction passes |
| **Shader Complexity** | Low (single fragment shader) | Medium-High (multi-pass fragment pipeline) | High (multi-program fragment pipeline) | High (multi-pass fragment pipeline) |
| **Debugging Difficulty** | Easy (visual validation) | Medium (spatial coherence + texture inspection) | Hard (quality validation across programs) | Medium (visualizable texture levels) |
| **Time to First Working Demo** | 1-2 weeks | 4-5 weeks | 5-7 weeks | 4-6 weeks |

### Performance Characteristics

| Metric | Plan A | Plan C | Plan D | Plan M |
|--------|--------|--------|--------|--------|
| **Target Particle Count** | 1M | 2M+ | 3M+ | 2M+ (hardware-dependent) |
| **Memory Bandwidth Reduction** | 4× (sampling) | 5-7× (spatial culling) | 2-20× (adaptive) | O(N log N) scaling; O(N) build cost via blending |
| **GPU Memory Usage** | High (full textures) | Medium (grid + sorted textures) | Low-Medium (importance-based + 3D approximation) | Medium-High (texture pyramid ~4/3× particle data) |
| **CPU Overhead** | Minimal | Medium (sorting fallback) | Medium (importance classification) | Low (GPU-side build via fragment shaders) |
| **Frame Rate Stability** | Good (consistent load) | Variable (distribution dependent) | Adaptive (quality scaling) | Good (consistent O(N log N) pattern) |

### Browser Compatibility & Future Support

| Consideration | Plan A | Plan C | Plan D | Plan M |
|---------------|--------|--------|--------|--------|
| **WebGL2 Support** | Universal ✓ | Requires EXT_color_buffer_float (widely supported) | Requires EXT_color_buffer_float + limited 3D texture support | Requires EXT_color_buffer_float + EXT_float_blend (widely supported) |
| **Compute Shader Requirement** | No | No (fragment shader based) | No (fragment shader based) | No (fragment shader based) |
| **Extension Dependencies** | None | EXT_color_buffer_float | EXT_color_buffer_float + OES_texture_3D | EXT_color_buffer_float + EXT_float_blend |
| **Mobile Performance** | Good | Good (efficient fragment pipeline) | Excellent (adaptive quality) | Good (efficient fragment pipeline, reasonable memory) |
| **WebGPU Migration Path** | Hard (texture-dependent) | Medium (fragment→compute, spatial structures) | Good (multi-program design) | Medium (fragment→compute, texture arrays→storage buffers) |
| **Long-term Viability** | Limited scaling | Good (proven spatial algorithms) | Excellent (adaptive approaches) | Excellent (proven algorithmic approach) |

### Development Risk Assessment

#### Plan A: Texture-Based Stochastic Sampling
**Low Risk ⭐⭐⭐⭐⭐**
- *Pros*: Proven techniques, easy debugging, incremental optimization path
- *Cons*: Limited scaling ceiling, memory inefficiency, visual sampling artifacts
- *Failure Modes*: Texture size limits, sampling jitter, poor cache utilization
- *Mitigation*: Blue noise sampling patterns, temporal coherence, progressive enhancement

#### Plan C: Fragment Shader Spatial Grid with Additive Blending
**Medium-High Risk ⭐⭐⭐**
- *Pros*: Excellent computational complexity, handles clustering well, proven algorithms
- *Cons*: Complex spatial data management, performance depends on distribution uniformity
- *Failure Modes*: Grid cell size tuning, particle clustering hotspots, sorting overhead
- *Mitigation*: Adaptive cell sizing, hybrid spatial-graph approaches, load balancing

#### Plan D: Fragment Shader Multi-Resolution Force Calculation
**High Risk ⭐⭐**
- *Pros*: Ultimate scalability, quality-performance trade-offs, user-adaptive
- *Cons*: Complex importance classification, multiple code paths, quality validation
- *Failure Modes*: Importance calculation overhead, visual artifacts at transitions, tuning complexity
- *Mitigation*: Conservative default parameters, extensive quality metrics, user feedback integration

#### Plan M: Fragment Shader Quadtree  
**Medium Risk ⭐⭐⭐**
- *Pros*: Excellent asymptotic scaling, uses standard WebGL2 features, GPU-resident algorithm, visualizable for debugging
- *Cons*: Multi-pass complexity, texture pyramid memory overhead, extension dependencies
- *Failure Modes*: Float blending unavailability, texture size limits, precision accumulation errors, sparse distribution inefficiency
- *Mitigation*: Extension detection with fallbacks, adaptive L0 sizing, careful numeric handling, hybrid approaches for sparse regions

### Recommended Implementation Strategy

#### Phase 1: Foundation (Weeks 1-4)
**Start with Plan A** - Establish working simulation with texture-based storage and stochastic sampling
- Validate core physics and rendering pipeline
- Implement performance monitoring and quality metrics
- Target: 100K particles at 60fps with basic social graph forces

#### Phase 2: Spatial Optimization (Weeks 5-8)  
**Implement Plan C** - Add fragment shader spatial grid while maintaining Plan A fallback
- Convert to spatial grid construction using fragment shaders and additive blending
- Implement grid-based culling and force calculation
- Target: 500K-1M particles with spatial optimization efficiency

#### Phase 3: Scaling Enhancement (Weeks 9-12)
**Add Plan D or Plan M Elements** - Integrate multi-resolution or hierarchical approaches
- Implement adaptive quality control (Plan D) or quadtree traversal (Plan M)
- Add adaptive resolution and load balancing
- Target: 1.5M+ particles with advanced scaling optimizations

#### Phase 4: Quality Scaling (Weeks 13-16)
**Integrate Plan D Concepts** - Add multi-resolution capabilities for ultimate scaling
- Implement importance classification and adaptive quality
- Add precomputed spatial approximations
- Target: 2M+ particles with quality-performance trade-offs

### Technical Decision Matrix

**Choose Plan A if:**
- Development timeline is tight (< 4 weeks)
- Team has limited advanced WebGL2 experience  
- Target is primarily desktop browsers with good GPU memory
- Particle count requirements are < 1M

**Choose Plan C if:**
- Particle distributions are spatially clustered (social graphs typically are)
- Need efficient fragment shader-based spatial optimization
- Acceptable to handle spatial data structure management using textures
- EXT_color_buffer_float extension is available

**Choose Plan D if:**
- Ultimate scalability is required (3M+ particles)
- Quality-performance trade-offs are acceptable
- Development team has advanced fragment shader programming experience
- Long-term maintenance and tuning resources are available

### Future Technology Considerations

**WebGPU Migration Readiness:**
- Plan C: Good (spatial structures translate well to compute)
- Plan D: Good (multi-program design fits compute model)
- Plan M: Good (fragment→compute shaders, texture pyramids→storage buffers)
- Plan C: Medium (spatial structures need redesign)
- Plan A: Poor (texture-dependent approach becomes obsolete)

**Machine Learning Integration Potential:**
- Plan D: Excellent (importance classification can use ML)
- Plan C: Medium (spatial clustering can benefit from ML)
- Plan M: Medium (ML could optimize theta parameters, L0 sizing, and hybrid switching)
- Plan A: Poor (limited optimization surface)

**Maintenance and Evolution:**
- Plan A: Good (simple to understand and modify)
- Plan M: Good (clear build/traverse separation, standard fragment shader techniques)
- Plan C: Fair (spatial logic complexity, but fragment shader based)
- Plan D: Poor (multiple interacting systems, complex fragment shader coordination)

## Plan M: "The Menace" — GPU-side Dynamic Quadtree (Fragment Shader Based)
*A full GPU-resident quadtree built as textures with multi-pass fragment shader reductions and per-particle traversal (Barnes–Hut style).* 

### Summary
Plan M builds a complete quadtree structure on the GPU using only standard WebGL2 fragment shaders. The quadtree is stored as a pyramid of textures, where each level contains node aggregates (mass, center-of-mass, particle count). The pipeline has two main stages: (1) Build - using additive blending to aggregate particles into leaf nodes, then multi-pass fragment shader reductions to build up the tree pyramid, and (2) Traverse - per-particle Barnes–Hut traversal using texture fetches across pyramid levels. This approach achieves O(N log N) scaling using only standard WebGL2 features.

### Key WebGL2 Features Required
- Fragment shaders with dynamic loops and texture fetches (standard WebGL2)
- Multiple Render Targets (MRT) for efficient multi-component aggregation
- Additive blending with floating-point render targets (EXT_color_buffer_float + EXT_float_blend)
- Sufficient texture size limits (MAX_TEXTURE_SIZE) for the quadtree base level
- Reasonable VRAM to hold the quadtree texture pyramid

All required features are part of standard WebGL2 or widely-supported extensions, making this approach much more portable than compute shader alternatives.

### Data Layouts and Formats
- **Lowest-level texture (L0)**: Power-of-two resolution covering simulation domain. Map particle world space → texel coordinates (x,y). Each texel represents a leaf node that aggregates particles in that spatial region.
- **Per-level textures**: Level L has resolution size_L = size_{L-1} / 2 (integer division), forming a texture pyramid where each texel represents a quadtree node.
- **Node data format**: Each texel stores RGBA32F containing:
  - R = sum_x (center-of-mass numerator x-component)  
  - G = sum_y (center-of-mass numerator y-component)
  - B = massSum (total mass of particles in this node)
  - A = particleCount (number of particles, stored as float)

**Indexing**: Use `texelFetch(texture, ivec2(x, y), 0)` for direct texel access. During reduction, each parent texel at level L+1 corresponds to a 2×2 block of child texels at level L with coordinates `(parentX*2 + {0,1}, parentY*2 + {0,1})`.

### Build Stage (Fragment Shader Pipeline)

#### 1. Leaf Level Aggregation (Particles → L0)
Use additive blending to aggregate particles into the lowest-level texture:

```glsl
// Vertex shader - render particles as points
attribute vec3 a_position;
attribute float a_mass;

uniform mat4 u_worldToGrid;  // Transform world coords to grid texture coords
uniform vec2 u_gridSize;     // Size of L0 texture

varying vec4 v_particleData; // xyz = center-of-mass contribution, w = mass

void main() {
    // Transform particle position to grid space
    vec4 gridPos = u_worldToGrid * vec4(a_position, 1.0);
    
    // Convert to texture coordinates (0 to gridSize)
    vec2 texCoord = gridPos.xy * u_gridSize;
    
    // Output as screen coordinates for point rendering
    gl_Position = vec4((texCoord / u_gridSize) * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = 1.0;
    
    // Pass data for fragment shader
    v_particleData = vec4(a_position.xy * a_mass, a_mass, 1.0); // weighted position + mass + count
}
```

```glsl
// Fragment shader - accumulate into leaf nodes
varying vec4 v_particleData;

void main() {
    // Output components that will be additively blended
    gl_FragColor = v_particleData; // sum_x, sum_y, mass, count
}
```

Render setup: Enable additive blending `(GL_ONE, GL_ONE)` and render to RGBA32F texture. Each particle contributes its weighted position and mass to the texel corresponding to its spatial location.

#### 2. Pyramid Reduction Passes (L0 → L1 → L2 → ... → Root)
Build upper levels using fragment shader reductions:

```glsl
// Fragment shader for reduction pass
uniform sampler2D u_previousLevel;
uniform vec2 u_previousLevelSize;

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
    
    gl_FragColor = aggregate;
}
```

For each level, render a full-screen quad that samples the 2×2 child blocks and outputs the aggregated parent values. Continue until reaching a 1×1 root texture.

### Traversal Stage (Fragment Shader Based)
Per-particle Barnes-Hut traversal using fragment shaders with texture pyramid access:

```glsl
// Fragment shader - one fragment per particle
uniform sampler2D u_particlePositions;
uniform sampler2D u_quadtreeLevels[MAX_LEVELS]; // Array of quadtree level textures
uniform float u_theta;                          // Barnes-Hut approximation parameter
uniform int u_numLevels;
uniform float u_cellSizes[MAX_LEVELS];          // Physical size of cells at each level

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    vec3 myPos = texelFetch(u_particlePositions, coord, 0).xyz;
    vec3 totalForce = vec3(0.0);
    
    // Traversal stack (fixed size for WebGL2 compatibility)
    const int MAX_STACK = 32;
    int stack[MAX_STACK];
    int stackTop = 0;
    
    // Start at root level (highest level index)
    stack[0] = u_numLevels - 1;  // level index
    stackTop = 1;
    
    while (stackTop > 0) {
        // Pop current level
        int level = stack[--stackTop];
        
        // Calculate which nodes to process at this level
        vec2 levelSize = textureSize(u_quadtreeLevels[level], 0);
        vec2 nodeCoord = (myPos.xy / u_cellSizes[level]) * levelSize;
        ivec2 nodeIndex = ivec2(floor(nodeCoord));
        
        // Clamp to valid range
        nodeIndex = clamp(nodeIndex, ivec2(0), ivec2(levelSize) - 1);
        
        // Fetch node data
        vec4 nodeData = texelFetch(u_quadtreeLevels[level], nodeIndex, 0);
        float massSum = nodeData.b;
        float particleCount = nodeData.a;
        
        if (particleCount > 0.0) {
            // Calculate center of mass
            vec2 centerOfMass = nodeData.rg / massSum;
            
            // Distance to center of mass
            float distance = length(myPos.xy - centerOfMass);
            float nodeSize = u_cellSizes[level];
            
            // Barnes-Hut criterion: if far enough, use approximation
            if ((nodeSize / distance) < u_theta || level == 0) {
                // Use this node for force approximation
                vec2 direction = centerOfMass - myPos.xy;
                float forceMagnitude = massSum / (distance * distance + 0.01); // softening
                totalForce.xy += normalize(direction) * forceMagnitude;
            } else {
                // Need to descend to children (if not at leaf level)
                if (level > 0 && stackTop < MAX_STACK - 4) {
                    // Push child levels onto stack
                    stack[stackTop++] = level - 1;
                }
            }
        }
    }
    
    gl_FragColor = vec4(totalForce, 0.0);
}
```

**Implementation Notes:**
- Each particle gets one fragment that performs Barnes-Hut traversal
- Uses fixed-size local array for traversal stack (WebGL2 compatible)
- Traverses from coarse to fine levels, using approximation when appropriate
- Texture array access allows reading from different pyramid levels efficiently

### WebGL2 Implementation Details

#### Pipeline Overview
1. **Setup Phase**: Create texture pyramid (all levels allocated)
2. **Build Phase**: 
   - Clear all level textures
   - Render particles to L0 using additive blending
   - For each level L0→L1→L2→...→root: render reduction pass
3. **Traverse Phase**: 
   - Render full-screen quad with per-particle traversal fragment shader
   - Output forces to force accumulation texture
4. **Integration Phase**: Standard velocity/position integration

#### JavaScript Pipeline Management
```javascript
class QuadtreePipeline {
    constructor(gl, worldBounds, maxParticles) {
        this.gl = gl;
        this.setupTexturePyramid(worldBounds, maxParticles);
        this.createShaderPrograms();
    }
    
    setupTexturePyramid(worldBounds, maxParticles) {
        // Calculate L0 size to fit maxParticles with reasonable density
        this.L0Size = Math.ceil(Math.sqrt(maxParticles * 4)); // 4x oversampling
        this.numLevels = Math.ceil(Math.log2(this.L0Size)) + 1;
        
        this.levelTextures = [];
        let currentSize = this.L0Size;
        
        for (let i = 0; i < this.numLevels; i++) {
            const texture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA32F, 
                              currentSize, currentSize, 0, 
                              this.gl.RGBA, this.gl.FLOAT, null);
            
            this.levelTextures.push({texture, size: currentSize});
            currentSize = Math.max(1, Math.floor(currentSize / 2));
        }
    }
    
    buildQuadtree(particlePositions) {
        // 1. Clear all textures
        this.clearAllLevels();
        
        // 2. Aggregate particles into L0
        this.aggregateParticles(particlePositions);
        
        // 3. Build pyramid via reduction passes
        for (let level = 0; level < this.numLevels - 1; level++) {
            this.runReductionPass(level, level + 1);
        }
    }
    
    aggregateParticles(positions) {
        this.gl.useProgram(this.aggregationProgram);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.L0Framebuffer);
        
        // Enable additive blending
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.ONE, this.gl.ONE);
        
        // Upload particle positions and render as points
        this.uploadParticleData(positions);
        this.gl.drawArrays(this.gl.POINTS, 0, positions.length);
        
        this.gl.disable(this.gl.BLEND);
    }
    
    runReductionPass(sourceLevel, targetLevel) {
        this.gl.useProgram(this.reductionProgram);
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.levelFramebuffers[targetLevel]);
        
        // Bind source level as texture
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.levelTextures[sourceLevel].texture);
        
        // Render full-screen quad
        this.renderFullScreenQuad();
    }
}
```

### Memory and Bandwidth Analysis
- For N ≈ 1,048,576 particles (L0 = 1024×1024 ≈ 1M texels):
    - RGBA32F per level = 16 bytes/texel
    - L0 = ~16MB, L1 = ~4MB, L2 = ~1MB, etc.
    - Total pyramid size ≈ 4/3 × L0 = ~21.3MB
- **Build bandwidth**: O(N) particle writes via blending + O(N) reduction reads/writes
- **Traversal bandwidth**: O(N log N) texture fetches (excellent cache locality within levels)
- **Compared to Plan A**: Higher memory usage but much better asymptotic scaling for large N

### Practical Considerations

#### Performance Characteristics
- **Build time**: O(N) for leaf aggregation + O(N) for pyramid (log N passes of decreasing size)
- **Traversal time**: O(N log N) with good cache locality 
- **Total per-frame**: O(N log N) which scales better than O(N×K) approaches at large N
- **Break-even point**: Typically outperforms linear methods around 500K-1M particles

#### WebGL2 Compatibility
- **Required extensions**: EXT_color_buffer_float, EXT_float_blend (widely supported)
- **Optional optimizations**: Multiple render targets for faster aggregation
- **Fallback strategy**: If float blending unavailable, use multiple passes with integer accumulation
- **Texture limits**: Requires MAX_TEXTURE_SIZE ≥ L0 size; typical limit is 4K-16K (sufficient for 1M-16M particles)

### Implementation Challenges and Solutions

#### Challenge: Precision Loss in Reductions
**Problem**: Floating-point accumulation can lose precision with many small contributions
**Solution**: Use double-precision emulation or careful ordering of operations. Consider using 32-bit integer fixed-point for center-of-mass calculations (Q16.16 format)

#### Challenge: Sparse Particle Distributions  
**Problem**: Large empty regions waste texture memory and processing
**Solution**: Use adaptive L0 sizing or hybrid approaches that fall back to Plan A for sparse regions

#### Challenge: Dynamic Particle Counts
**Problem**: Texture pyramid sized for max particles wastes memory with fewer particles
**Solution**: Multiple pre-allocated pyramid sizes, or dynamic allocation with texture resizing

#### Challenge: Barnes-Hut Parameter Tuning
**Problem**: Theta parameter affects both accuracy and performance significantly  
**Solution**: Adaptive theta based on local particle density, or user-configurable quality presets

### Validation and Debugging

#### Correctness Validation
- **Mass conservation**: Total mass at root should equal sum of particle masses
- **Center-of-mass accuracy**: Root center-of-mass should match CPU-computed global center-of-mass
- **Force consistency**: Compare forces against direct O(N²) computation on small test cases
- **Level consistency**: Each parent node's data should equal the sum of its children

#### Performance Validation  
- **Build time scaling**: Should scale as O(N) for reasonable N ranges
- **Traversal time scaling**: Should scale better than O(N×K) at large N
- **Memory usage**: Should use approximately 4/3 × L0 texture size
- **Frame rate stability**: Should maintain consistent frame times

#### Debug Visualizations
```glsl
// Debug fragment shader to visualize quadtree levels
uniform sampler2D u_levelTexture;
uniform int u_debugComponent; // 0=count, 1=mass, 2=COM_x, 3=COM_y

void main() {
    vec4 nodeData = texture2D(u_levelTexture, gl_FragCoord.xy / textureSize(u_levelTexture, 0));
    
    float value;
    if (u_debugComponent == 0) value = nodeData.a; // count
    else if (u_debugComponent == 1) value = nodeData.b; // mass  
    else if (u_debugComponent == 2) value = nodeData.r / nodeData.b; // COM x
    else value = nodeData.g / nodeData.b; // COM y
    
    gl_FragColor = vec4(value, value, value, 1.0);
}
```

### Integration with Other Plans

#### Fallback Strategy
- **Primary**: Use Plan M for particle counts > 1M when WebGL2 features available
- **Fallback**: Drop to Plan A (stochastic sampling) for devices without float blending
- **Emergency**: Use Plan A with reduced particle count for compatibility with older devices

#### Hybrid Approaches
- **Spatial-Quadtree Hybrid**: Use Plan C's spatial grid for uniform regions, Plan M's quadtree for clustered regions
- **Multi-Resolution Integration**: Combine with Plan D's importance system - use quadtree only for high-importance particles
- **Progressive Enhancement**: Start with Plan A, upgrade to Plan M when device capabilities detected

### Implementation Roadmap

#### Phase 1: Basic Quadtree (2-3 weeks)
- Implement texture pyramid creation and management
- Build particle aggregation using additive blending  
- Create basic reduction passes for pyramid building
- Validate mass conservation and center-of-mass accuracy

#### Phase 2: Traversal Implementation (2-3 weeks)  
- Implement Barnes-Hut traversal fragment shader
- Add theta parameter tuning and optimization
- Integrate with force accumulation and physics pipeline
- Performance profiling and optimization

#### Phase 3: Polish and Integration (1-2 weeks)
- Add debug visualizations and validation tools
- Implement fallback detection and graceful degradation
- Integration testing with existing codebase
- Documentation and performance benchmarking

### Pros and Cons Summary

#### Advantages
- **Asymptotic scaling**: O(N log N) performance beats linear approaches at large N
- **Standard WebGL2**: Uses only fragment shaders and common extensions
- **GPU-resident**: Entire quadtree lives on GPU, minimal CPU-GPU transfer
- **Cache-friendly**: Texture pyramid provides excellent spatial locality
- **Deterministic**: Same particle distribution always produces same quadtree
- **Debuggable**: Can visualize every level of the quadtree for validation

#### Disadvantages  
- **Memory overhead**: Texture pyramid requires ~4/3× memory compared to particle data
- **Build cost**: O(N) rebuild every frame (though this is highly parallel)
- **Complexity**: Multi-pass pipeline with careful numeric handling required
- **Extension dependency**: Requires EXT_color_buffer_float + EXT_float_blend
- **Sparse inefficiency**: Wastes memory for sparse particle distributions

#### When to Choose Plan M
- **Large particle counts**: > 1M particles where O(N log N) beats O(N×K)
- **Clustered distributions**: Social graphs typically have spatial clustering
- **Modern devices**: Target devices with good GPU memory and float extension support
- **Research/scaling focus**: When algorithmic scaling is more important than implementation simplicity

---