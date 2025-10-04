# Plan M Debugging & Verification Plan

## Phase 1: Static Analysis & Build ✅ COMPLETED
- [x] Build verification (npm run build)
- [x] Module import verification
- [x] Syntax validation

## Phase 2: Enhanced Debug Infrastructure

### 2.1 Add Module-Level Logging
Each pipeline module should log when called:
- `pipeline/aggregator.js` - Log entry/exit with context
- `pipeline/pyramid.js` - Log reduction passes
- `pipeline/traversal.js` - Log force calculations
- `pipeline/integrator.js` - Log integration steps
- `pipeline/bounds.js` - Log bounds updates
- `pipeline/renderer.js` - Log render calls

### 2.2 Shader Validation Tool
Create a utility to validate all shaders:
```javascript
// utils/shader-validator.js
- Compile each shader independently
- Report line numbers for errors
- Validate uniform locations
```

### 2.3 GPU State Validator
Create checkpoint validation:
```javascript
// utils/gpu-validator.js
- Read back texture data at key points
- Validate ranges (no NaN/Infinity)
- Check framebuffer completeness
- Validate texture bindings
```

## Phase 3: Runtime Testing

### 3.1 Unit Test Individual Modules
Test each pipeline function in isolation:

**Test: Aggregator**
```javascript
1. Initialize with known particle positions
2. Run aggregateL0()
3. Read back L0 texture
4. Verify particle counts per cell
5. Verify center-of-mass calculations
```

**Test: Pyramid Builder**
```javascript
1. Create synthetic L0 with known values
2. Run pyramidReduce() for each level
3. Read back each level
4. Verify reduction math (2x2 → 1 aggregation)
```

**Test: Traversal**
```javascript
1. Set up simple 2-particle scenario
2. Run calculateForces()
3. Read back force texture
4. Verify force direction and magnitude
```

**Test: Integrator**
```javascript
1. Set known velocities and forces
2. Run integratePhysics()
3. Read back new positions/velocities
4. Verify integration math
```

### 3.2 Integration Tests
Test pipeline combinations:

**Test: Full Quadtree Build**
```javascript
1. Initialize particles
2. Run buildQuadtree()
3. Validate all levels populated
4. Check mass conservation
5. Verify spatial distribution
```

**Test: Full Simulation Step**
```javascript
1. Run complete step()
2. Verify particles moved
3. Check velocity changes
4. Validate no particles escaped bounds
```

### 3.3 Regression Tests
```javascript
1. Run 100 frames
2. Check for:
   - WebGL errors
   - NaN/Infinity in positions
   - Particle escapes
   - Performance degradation
```

## Phase 4: Visual Verification

### 4.1 Create Debug Visualization
Add visualization modes:
- **Heatmap**: Show L0 particle density
- **Level View**: Visualize each quadtree level
- **Force Vectors**: Draw force arrows on particles
- **Bounds**: Show world bounds
- **Stats**: FPS, particle count, GPU time

### 4.2 Visual Tests
```javascript
Test Cases:
1. Uniform Distribution - particles should stay distributed
2. Cluster Formation - gravity should form clusters
3. Binary System - two clusters should orbit
4. Collision - clusters should merge
5. Escape Velocity - high-velocity particles behavior
```

## Phase 5: Performance Profiling

### 5.1 Timing Breakdown
Measure each pipeline stage:
```javascript
- Quadtree Build (L0 aggregation + pyramid)
- Force Calculation (traversal)
- Integration (velocity + position)
- Rendering
- Total frame time
```

### 5.2 GPU Profiling
```javascript
- Check for GPU stalls
- Validate texture sizes
- Monitor WebGL calls
- Profile shader execution
```

### 5.3 Scalability Test
```javascript
Test with:
- 1,000 particles
- 10,000 particles
- 50,000 particles
- 100,000 particles
- Measure: FPS, frame time, build time
```

## Phase 6: Error Handling & Edge Cases

### 6.1 Test Error Scenarios
```javascript
1. Missing WebGL extensions
2. Shader compilation failures
3. Texture allocation failures
4. Zero particles
5. Single particle
6. All particles at same position
```

### 6.2 Boundary Conditions
```javascript
1. Particles at world bounds
2. Very high velocities
3. Very high particle densities
4. Empty quadtree cells
```

## Implementation Strategy

### Quick Start (Minimal Viable Test)
1. Start a dev server
2. Open browser console
3. Check for:
   - Module load errors
   - WebGL initialization
   - Shader compilation
   - First frame renders

### Commands to Run

```bash
# 1. Build and serve
npm run build
# Serve public/ directory (if you have a server)

# 2. Browser console checks
# - No errors on load?
# - "Plan M initialized successfully"?
# - Particles visible?
# - FPS counter working?

# 3. Add debug mode
# In browser console:
window.planM.options.debugSkipQuadtree = false
# Watch for quadtree build logs
```

### Debug Checklist

#### Initialization Phase
- [ ] WebGL2 context created
- [ ] Extensions loaded (EXT_color_buffer_float, EXT_float_blend)
- [ ] All shaders compiled without errors
- [ ] All programs linked successfully
- [ ] Textures created (levels, positions, velocities, forces)
- [ ] VAOs created (quad, particles)
- [ ] Initial particle data uploaded

#### First Frame
- [ ] buildQuadtree() executes
- [ ] L0 aggregation completes (check texture)
- [ ] Pyramid reduction completes (check all levels)
- [ ] calculateForces() executes
- [ ] integratePhysics() executes
- [ ] renderParticles() executes
- [ ] Particles visible on screen

#### Ongoing Operation
- [ ] No WebGL errors per frame
- [ ] Positions remain finite (no NaN)
- [ ] FPS stable
- [ ] Particles move realistically
- [ ] Bounds update correctly

### Critical Breakpoints

If issues occur, add breakpoints at:
1. `index.js:108` - After init()
2. `pipeline/aggregator.js:60` - After L0 draw
3. `pipeline/pyramid.js:23` - After each reduction
4. `pipeline/traversal.js:44` - After force calculation
5. `pipeline/renderer.js:43` - After render draw

### GPU Readback Points

Read back textures to validate:
```javascript
// After L0 aggregation
readTexture(levelTextures[0]) // Check for particles
// After pyramid
readTexture(levelTextures[numLevels-1]) // Should have root
// After forces
readTexture(forceTexture) // Check for non-zero forces
// After integration
readTexture(positionTextures.getCurrentTexture()) // Check movement
```

## Success Criteria

### Must Have (MVP)
- ✅ Code compiles
- ⏳ Simulation starts without errors
- ⏳ Particles visible on screen
- ⏳ Particles move over time
- ⏳ No WebGL errors in console

### Should Have
- ⏳ Quadtree builds successfully
- ⏳ Forces calculated correctly
- ⏳ Particles cluster under gravity
- ⏳ Performance: >30 FPS with 50K particles
- ⏳ No memory leaks over 1000 frames

### Nice to Have
- ⏳ All test cases pass
- ⏳ Debug visualizations working
- ⏳ Performance profiling complete
- ⏳ Documentation updated
