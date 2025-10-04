# Concrete Plan to Fix the T-Shape Bug

## The Problem
Particles show as "sparkly T shape in corner" instead of moving under gravity.

## Root Causes (in order of likelihood)

### 1. Render Coordinate Bug (MOST LIKELY)
**Symptom**: T-shape in corner means particles are being projected incorrectly
**Cause**: Render shader uses broken orthographic projection instead of Three.js camera
**Fix**: Replace ortho projection with proper camera matrix transform

### 2. Rendering Before Three.js Clears
**Symptom**: Particles don't appear at all, or flicker
**Cause**: Three.js clears the screen after we draw
**Fix**: Disable autoClear and manually control clear order

### 3. Particles Not Moving
**Symptom**: Particles stuck in place
**Cause**: Integration or force calculation broken
**Fix**: Check each pipeline stage

## Step-by-Step Fix Plan

### Phase 1: Fix Rendering (Makes particles visible in correct location)

#### Fix 1.1: Use Camera Matrix Instead of Ortho Projection
**File**: `src/plan-m/shaders/render.vert.js`
**Change**: Replace the ortho projection with:
```glsl
uniform mat4 u_projectionView;  // Add this uniform

void main() {
  // ... existing code to fetch position ...
  
  // Replace the ortho mapping (lines 34-40) with:
  gl_Position = u_projectionView * vec4(worldPos, 1.0);
  gl_PointSize = u_pointSize;
}
```

#### Fix 1.2: Pass Camera Matrix to Shader
**File**: `src/plan-m/pipeline/renderer.js`
**Change**: After line 37, add:
```javascript
// Compute projection-view matrix
camera.updateMatrixWorld();
camera.updateProjectionMatrix();
const pvMatrix = new THREE.Matrix4();
pvMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);

// Pass to shader
const u_projectionView = gl.getUniformLocation(ctx.programs.render, 'u_projectionView');
gl.uniformMatrix4fv(u_projectionView, false, pvMatrix.elements);
```

**Remove**: Lines 36-43 (old worldMin/worldMax uniforms - no longer needed)

#### Fix 1.3: Fix Render Order
**File**: `src/index.js`
**Already done**: `renderer.autoClear = false` and manual clearing

**Result after Phase 1**: Particles should be visible in center of screen (though maybe not moving yet)

---

### Phase 2: Verify Particle Movement (Debug which stage is broken)

#### Test 2.1: Are Particles Initialized?
**Add to**: `src/plan-m/index.js` in `initializeParticles()` after line 391:
```javascript
console.log(`P0: pos=[${positions[0]}, ${positions[1]}, ${positions[2]}] vel=[${velocities[0]}, ${velocities[1]}, ${velocities[2]}]`);
```
**Expected**: Non-zero positions around origin, small velocities

#### Test 2.2: Are Forces Calculated?
**Add to**: `src/plan-m/pipeline/traversal.js` after line 50:
```javascript
if (ctx.frameCount === 0) {
  const f = new Float32Array(4);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, f);
  console.log(`Force on P0: [${f[0]}, ${f[1]}, ${f[2]}]`);
}
```
**Expected**: Small non-zero forces (e.g., 0.0001 magnitude)
**If zero**: Quadtree or traversal shader is broken

#### Test 2.3: Do Velocities Change?
**Add to**: `src/plan-m/pipeline/integrator.js` after velocity integration:
```javascript
if (ctx.frameCount === 0) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.velocityTextures.getTargetFramebuffer());
  const v = new Float32Array(4);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, v);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  console.log(`New vel P0: [${v[0]}, ${v[1]}, ${v[2]}]`);
}
```
**Expected**: Velocity slightly different from initial
**If same**: Velocity integration shader is broken

#### Test 2.4: Do Positions Change?
**Add to**: `src/plan-m/index.js` in `update()`:
```javascript
if (this.frameCount === 60) {
  const gl = this.gl;
  gl.bindFramebuffer(gl.FRAMEBUFFER, this.positionTextures.framebuffers[this.positionTextures.currentIndex]);
  const p = new Float32Array(4);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, p);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  console.log(`Frame 60: P0 at [${p[0]}, ${p[1]}, ${p[2]}]`);
}
```
**Expected**: Position different from frame 0
**If same**: Position integration is broken

---

### Phase 3: Fix Integration Issues (If particles don't move)

#### Fix 3.1: Check Integration Shaders
**Files**: `src/plan-m/shaders/vel_integrate.frag.js` and `pos_integrate.frag.js`

**Verify**:
1. They read from correct textures
2. They write to framebuffer (not reading from it)
3. dt is reasonable (not 0 or NaN)
4. Ping-pong textures are swapped after each integration

#### Fix 3.2: Check Texture Ping-Pong
**File**: `src/plan-m/pipeline/integrator.js`

**Verify**: After each integration, textures are swapped:
```javascript
ctx.velocityTextures.swap();  // After velocity integration
ctx.positionTextures.swap();  // After position integration
```

#### Fix 3.3: Check dt Value
**Add to**: `src/plan-m/index.js` in `constructor`:
```javascript
console.log('dt:', this.options.dt, 'G:', this.options.gravityStrength);
```
**Expected**: dt around 0.016, G around 0.0003
**If wrong**: Adjust in options

---

### Phase 4: Fix Force Calculation (If forces are zero)

#### Fix 4.1: Check Quadtree Build
**Add to**: `src/plan-m/index.js` after `buildQuadtree()` call:
```javascript
// Read L0 to verify particles are aggregated
const gl = this.gl;
gl.bindFramebuffer(gl.FRAMEBUFFER, this.levelFramebuffers[0]);
const l0 = new Float32Array(4);
gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, l0);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
if (this.frameCount === 0) console.log(`L0[0,0]: [${l0[0]}, ${l0[1]}, ${l0[2]}, ${l0[3]}]`);
```
**Expected**: Non-zero values in L0
**If zero**: Aggregation is broken

#### Fix 4.2: Check Traversal Shader Uniforms
**File**: `src/plan-m/pipeline/traversal.js`

**Verify**: All uniforms are set correctly:
- Particle count matches texture
- World bounds are correct
- Cell sizes are correct
- All 8 quadtree level textures are bound

#### Fix 4.3: Check Traversal Shader Logic
**File**: `src/plan-m/shaders/traversal.frag.js`

**Common bugs**:
- Reading from wrong texture coordinate
- Softening parameter too large (makes forces zero)
- Theta parameter too small (forces BH approximation to always fail)
- G constant is zero or NaN

---

### Phase 5: Quick Fixes to Try First

#### Quick Fix 1: Increase Point Size
**File**: `src/plan-m/index.js` in `constructor`:
```javascript
pointSize: 3.0,  // Change from 2.0 to 3.0
```

#### Quick Fix 2: Simplify - Disable Quadtree
**Add to**: Browser console:
```javascript
plans.m.options.debugSkipQuadtree = true;
```
This tests if the problem is in quadtree or integration.

#### Quick Fix 3: Use Fixed Bounds
**File**: `src/plan-m/index.js` in `initializeParticles()`:
Replace lines 381-384 with:
```javascript
this.options.worldBounds = {
  min: [-10, -10, -10],
  max: [10, 10, 10]
};
```

---

## Execution Order

1. **Do Quick Fix 1** (point size) - 30 seconds
2. **Do Fix 1.1, 1.2, 1.3** (camera rendering) - 5 minutes - **THIS IS THE MAIN FIX**
3. **Build and test** - 1 minute
4. **If still broken, add Tests 2.1-2.4** - 5 minutes
5. **Based on test results, do Phase 3 or 4 fixes** - 10 minutes

## Expected Outcome

After Fix 1.1-1.3 (camera rendering):
- Particles appear in center of screen
- They move with camera as you orbit
- They may or may not be moving (need Phase 2 tests to verify)

After all fixes:
- Particles cluster toward center of mass
- System slowly collapses under gravity
- Particles move smoothly, not teleporting

## Most Likely Issue

Based on "T shape in corner", I'd bet money on:
**The render shader's orthographic projection is broken**

The fix is to replace it with proper camera matrix transform (Fix 1.1 and 1.2).

This should take 5 minutes to implement and will probably fix 90% of the problem.
