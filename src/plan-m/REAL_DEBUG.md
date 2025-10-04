# Real Debugging Plan: Fix the Broken Particle Simulation

## Symptom: "Sparkly T shape in corner" instead of particles moving under gravity

## Root Cause Analysis

### Issue 1: Renderer draws directly to screen, bypassing Three.js
**Problem**: The renderer writes directly to the default framebuffer (line 23 in `renderer.js`), which happens BEFORE Three.js renders its scene. This means:
- Three.js clears the screen after we draw
- Our particles get erased
- Or they appear in wrong coordinate space vs Three.js objects

**Fix Options:**
A. Render AFTER Three.js (inject into Three.js render pipeline)
B. Render to a texture, then display via Three.js material
C. Disable Three.js clear and coordinate correctly

### Issue 2: Coordinate System Mismatch
**Problem**: The render shader uses simple orthographic projection (lines 34-40 in `render.vert.js`):
```glsl
vec2 norm = (worldPos.xy - lo) / extent;
vec2 clipXY = clamp(norm * 2.0 - 1.0, vec2(-1.0), vec2(1.0));
gl_Position = vec4(clipXY, 0.0, 1.0);
```

This is a 2D ortho projection, but Three.js uses a perspective camera. The particles are:
- Being projected to 2D clip space [-1,1]
- Ignoring the Z coordinate
- Not using the Three.js camera matrix at all

**Why T-shape in corner?**
- World bounds might be calculated wrong
- Particles might all be at (0,0) or similar positions
- Coordinate mapping is broken

### Issue 3: Camera is completely ignored
The renderer gets the camera (line 5) but never uses it. No projection matrix, no view matrix.

### Issue 4: Bounds might be wrong
Initial bounds are set to `[-10, -10, -10] to [10, 10, 10]` but particles are spawned in a ~3 unit radius circle. Then bounds are recalculated but maybe broken.

## Concrete Debugging Steps

### Step 1: Add Console Logging to See What's Happening

Add to `renderer.js` line 58:
```javascript
if (ctx.frameCount < 3) {
  console.log(`Plan M: Rendered ${ctx.options.particleCount} particles`);
  console.log(`World bounds:`, ctx.options.worldBounds);
  console.log(`Texture size: ${ctx.textureWidth}x${ctx.textureHeight}`);
  console.log(`Drawing ${ctx.options.particleCount} points`);
}
```

### Step 2: Check If Particles Are Actually Moving

Add to `index.js` in the `update()` method (after line 430):
```javascript
// Debug: log first particle position every 60 frames
if (this.frameCount % 60 === 0) {
  const gl = this.gl;
  gl.bindFramebuffer(gl.FRAMEBUFFER, this.positionTextures.framebuffers[this.positionTextures.currentIndex]);
  const px = new Float32Array(4);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  console.log(`Frame ${this.frameCount}: Particle 0 at [${px[0].toFixed(2)}, ${px[1].toFixed(2)}, ${px[2].toFixed(2)}]`);
}
```

### Step 3: Fix the Rendering (Choose ONE approach)

#### Option A: Quick Fix - Disable Three.js auto-clear
In `src/index.js`, after creating the renderer:
```javascript
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.autoClear = false; // ADD THIS LINE
```

Then in the render loop, manually control clearing:
```javascript
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  
  // Clear manually
  renderer.clear();
  
  // Update and render Plan M (draws particles first)
  if (activePlan && activePlan.update) activePlan.update();
  
  // Render Three.js scene (grid, axes, etc.)
  renderer.render(scene, camera);
}
```

#### Option B: Better Fix - Use Three.js coordinate system
Replace the render shader to use Three.js camera:

1. Add uniform for projection-view matrix in `render.vert.js`:
```glsl
uniform mat4 u_projectionView;
```

2. Transform vertices properly:
```glsl
vec4 worldPos4 = vec4(worldPos, 1.0);
gl_Position = u_projectionView * worldPos4;
```

3. Update `renderer.js` to pass the matrix:
```javascript
// After line 37
camera.updateMatrixWorld();
const pvMatrix = new Float32Array(16);
const pv = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
pv.toArray(pvMatrix);

const u_projectionView = gl.getUniformLocation(ctx.programs.render, 'u_projectionView');
gl.uniformMatrix4fv(u_projectionView, false, pvMatrix);
```

### Step 4: Check if Forces Are Being Calculated

Add to `pipeline/traversal.js` after the draw call (after line 44):
```javascript
// Debug: read back one force value
if (ctx.frameCount < 3) {
  const gl = ctx.gl;
  const px = new Float32Array(4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.forceTexture.framebuffer);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
  console.log(`Force on particle 0: [${px[0].toFixed(4)}, ${px[1].toFixed(4)}, ${px[2].toFixed(4)}]`);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
```

### Step 5: Check if Integration is Working

Add to `pipeline/integrator.js` after velocity integration (around line 30):
```javascript
// Debug: check if velocities are changing
if (ctx.frameCount < 3) {
  const gl = ctx.gl;
  const px = new Float32Array(4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.velocityTextures.framebuffers[1 - ctx.velocityTextures.currentIndex]);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
  console.log(`Velocity of particle 0: [${px[0].toFixed(4)}, ${px[1].toFixed(4)}, ${px[2].toFixed(4)}]`);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
```

### Step 6: Verify Particle Initialization

Add to `index.js` in `initializeParticles()` after line 392:
```javascript
console.log(`Initialized particles with positions:`, {
  first: [positions[0], positions[1], positions[2], positions[3]],
  bounds: this.options.worldBounds,
  center: center
});
```

## Quick Test Script

Add this to browser console after pressing `4`:

```javascript
// Get Plan M instance
const planM = plans.m;

// Check if initialized
console.log('Initialized:', planM.isInitialized);
console.log('Running:', planM.running);

// Check particle count
console.log('Particles:', planM.options.particleCount);

// Check world bounds
console.log('Bounds:', planM.options.worldBounds);

// Read first particle position
const gl = planM.gl;
gl.bindFramebuffer(gl.FRAMEBUFFER, planM.positionTextures.framebuffers[planM.positionTextures.currentIndex]);
const pos = new Float32Array(4);
gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pos);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
console.log('Particle 0 position:', pos);

// Read first particle velocity
gl.bindFramebuffer(gl.FRAMEBUFFER, planM.velocityTextures.framebuffers[planM.velocityTextures.currentIndex]);
const vel = new Float32Array(4);
gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, vel);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
console.log('Particle 0 velocity:', vel);

// Read first particle force
gl.bindFramebuffer(gl.FRAMEBUFFER, planM.forceTexture.framebuffer);
const force = new Float32Array(4);
gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, force);
gl.bindFramebuffer(gl.FRAMEBUFFER, null);
console.log('Particle 0 force:', force);
```

## Expected Behavior vs Actual

### Expected:
- Particles spawn in a disk around origin
- Initial velocities are small random vectors
- Forces point toward center of mass
- Particles slowly drift and cluster
- Visible in center of screen with Three.js camera

### Actual ("T shape in corner"):
Likely causes:
1. **All particles at (0,0,0)** - initialization bug
2. **Bounds are [0,0] to [epsilon,epsilon]** - bounds calculation bug
3. **Rendering to wrong framebuffer** - Three.js clears screen
4. **Coordinate transform broken** - particles mapped to wrong screen space
5. **Camera matrix not applied** - particles ignore perspective

## Priority Fixes (Do These First)

1. **Fix rendering order** - Option A above (disable autoClear)
2. **Add position logging** - Step 2 above
3. **Add bounds logging** - Step 1 above
4. **Run test script** - see what's actually in the textures

Then based on the logs, you'll know:
- Are particles initialized correctly?
- Are they moving?
- Where are they being drawn?
- What are the bounds?

## Most Likely Bug

Based on "T shape in corner", I bet:
- **Bounds are wrong** (very small or negative)
- Particles are at reasonable world positions (e.g., -3 to +3)
- But the ortho projection maps them incorrectly
- Result: all particles squeezed into corner

**Test this:** Change `initializeParticles()` to hardcode bounds:
```javascript
this.options.worldBounds = {
  min: [-5, -5, -5],
  max: [5, 5, 5]
};
```

Then particles should be visible.
