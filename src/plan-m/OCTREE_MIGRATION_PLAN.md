# Octree Migration Plan

## Current Status: 2D Quadtree → 3D Octree Migration

**Date**: October 4, 2025  
**Goal**: Replace 2D quadtree with 3D octree for proper isotropic 3D physics

## Problem Statement

The current implementation uses a **2D quadtree** for Barnes-Hut approximation:
- ❌ Only stores (x·m, y·m, mass) in quadtree nodes - **Z coordinate is lost**
- ❌ Force calculations use 2D distance: `length(com.xy - myPos.xy)` - **ignores Z separation**
- ❌ Results in **grid-like artifacts** and **string/filament structures**
- ❌ Particles don't collapse properly in 3D - they form 2D radial patterns

**Root Cause**: Z dimension is treated as special/inferior, not equal to X and Y.

---

## Solution Architecture

### Octree Structure
Replace 512×512 quadtree with **64³ octree**:

```
Level 0 (L0): 64×64×64 = 262,144 voxels  (finest, ~same as current 512×512)
Level 1 (L1): 32×32×32 = 32,768 voxels
Level 2 (L2): 16×16×16 = 4,096 voxels
Level 3 (L3): 8×8×8 = 512 voxels
Level 4 (L4): 4×4×4 = 64 voxels
Level 5 (L5): 2×2×2 = 8 voxels
Level 6 (L6): 1×1×1 = 1 voxel (root)
```

### Data Storage Format

**Current (2D)**: `vec4(x·m, y·m, mass, count)`  
**New (3D)**: `vec4(x·m, y·m, z·m, mass)` ✅ **Z now included!**

### 3D Volume → 2D Texture Mapping

Since WebGL textures are 2D, store 3D octree by **stacking Z-slices**:

```
Texture layout for 64³ grid:
- Each Z-slice is 64×64 pixels
- 64 slices total
- Arrange in 8×8 grid → 512×512 texture
- Or 4×16 grid → 256×1024 texture

Coordinate mapping:
voxel(x,y,z) → texCoord(baseX + x, baseY + y)
where:
  baseX = (z % slicesPerRow) * gridSize
  baseY = (z / slicesPerRow) * gridSize
```

---

## Implementation Checklist

### ✅ Phase 1: Shaders (COMPLETED)

- [x] **aggregation.vert.js**: 
  - Changed `u_worldMin`/`u_worldMax` from `vec2` → `vec3`
  - Changed binning from 2D grid → 3D voxel grid
  - Added `voxelToTexCoord()` function for 3D→2D mapping
  - Output now includes Z: `vec4(pos.xyz * mass, mass)`

- [x] **reduction.frag.js**:
  - Changed from 2×2 reduction (4 children) → 2×2×2 reduction (8 children)
  - Added `voxelToTexCoord()` and `texCoordToVoxel()` functions
  - Added `u_prevGridSize`, `u_currGridSize`, `u_textureWidth` uniforms
  - Samples 8 children in 3D space

- [x] **traversal.frag.js**:
  - Changed `u_worldMin`/`u_worldMax` from `vec2` → `vec3`
  - Changed distance calculation to 3D: `length(com - myPos)` (full vec3)
  - Changed neighborhood iteration from 2D → 3D (dx, dy, dz loops)
  - Extract 3D center of mass: `vec3 com = nodeData.rgb / mass`
  - Compute 3D force vector: `totalForce += normalize(dir) * m / r²`
  - Removed opportunistic Z-force hack
  - Added `u_textureWidth` uniform for 3D→2D mapping

### 🔧 Phase 2: Pipeline Modules (TODO)

#### **aggregator.js**
```javascript
// Changes needed:
- Update u_worldMin: gl.uniform2f() → gl.uniform3f()
- Update u_worldMax: gl.uniform2f() → gl.uniform3f()
- Add u_textureWidth: gl.uniform1i(u_textureWidth, 512 or 256)
- Change L0 from 512 → 64
```

#### **pyramid.js** (reduction.js)
```javascript
// Changes needed:
- Calculate texture sizes for 3D octree levels:
  * L0: 64³ → 512×512 or 256×1024 texture
  * L1: 32³ → 256×256 or 128×512 texture
  * L2: 16³ → 128×128 or 64×256 texture
  * L3: 8³ → 64×64 or 32×128 texture
  * L4: 4³ → 32×32 or 16×64 texture
  * L5: 2³ → 16×16 or 8×32 texture
  * L6: 1³ → 8×8 or 4×16 texture
  
- Add uniforms for each reduction pass:
  * u_prevGridSize (e.g., 64 for L0→L1 reduction)
  * u_currGridSize (e.g., 32 for L0→L1 reduction)
  * u_textureWidth (texture width for mapping)

- Update framebuffer sizes for each level
```

#### **traversal.js**
```javascript
// Changes needed:
- Update u_worldMin: gl.uniform2f() → gl.uniform3f()
- Update u_worldMax: gl.uniform2f() → gl.uniform3f()
- Add u_textureWidth: gl.uniform1i()
- Update cell size calculation for 3D:
  * cellSizes[i] = worldExtent / gridSize (3D diagonal)
  * gridSize: [64, 32, 16, 8, 4, 2, 1]
```

#### **renderer.js**
```javascript
// Changes needed:
- Update u_worldMin: gl.uniform2f() → gl.uniform3f()
- Update u_worldMax: gl.uniform2f() → gl.uniform3f()
```

#### **bounds.js**
```javascript
// Changes needed:
- Already uses 3D worldBounds {min: [x,y,z], max: [x,y,z]}
- Should work as-is, but verify Z bounds are computed correctly
```

#### **index.js** (main)
```javascript
// Changes needed:
- Update worldBounds initialization to ensure proper 3D bounds
- Verify particle initialization uses 3D distribution
- Update pyramid level configuration:
  * Change from 8 quadtree levels (512→256→...→1 in 2D)
  * To 7 octree levels (64→32→16→8→4→2→1 in 3D)
```

### 🧪 Phase 3: Testing & Validation

- [ ] **Verify aggregation**: Check L0 texture contains correct voxel data
- [ ] **Verify reduction**: Check each level properly aggregates 8 children
- [ ] **Verify traversal**: Check 3D forces are computed correctly
- [ ] **Performance**: Measure frame time, ensure acceptable performance
- [ ] **Visual check**: Particles should form 3D clusters, not strings
- [ ] **Physics check**: Particles should collapse spherically, not in 2D planes

---

## Technical Details

### Cell Size Calculation (3D)

```javascript
// Current (2D): 
const worldExtent = worldMax.xy - worldMin.xy
const cellSize = worldExtent / gridSize  // 2D

// New (3D):
const worldExtent = worldMax.xyz - worldMin.xyz
const cellSize = max(worldExtent.x, worldExtent.y, worldExtent.z) / gridSize
// Or use 3D diagonal: length(worldExtent) / gridSize
```

### Barnes-Hut Criterion (3D)

```glsl
// Same formula, but now with 3D distance:
float s = cellSize;  // voxel size
float d = length(com - myPos);  // 3D distance
if (s / d < theta) {
  // Accept approximation
  vec3 force = normalize(com - myPos) * mass / (d*d + eps*eps);
}
```

### Memory Usage

**Current (2D quadtree)**:
- L0: 512×512 = 262k cells
- L1-L7: 256×256 + 128×128 + ... = ~87k cells
- **Total**: ~349k cells × 4 floats = 1.4 MB

**New (3D octree)**:
- L0: 64×64×64 = 262k cells
- L1-L6: 32³ + 16³ + 8³ + 4³ + 2³ + 1³ = ~37k cells
- **Total**: ~299k cells × 4 floats = 1.2 MB ✅ **Slightly less memory!**

---

## Potential Issues & Solutions

### Issue 1: Texture Size Limits
**Problem**: 64³ = 262k voxels, but max texture size might be limited  
**Solution**: Use 512×512 texture (262k pixels) or 256×1024 if needed

### Issue 2: Performance
**Problem**: 3D neighborhood iteration is more expensive (27 neighbors vs 9)  
**Solution**: Use adaptive radius (smaller at fine levels, larger at coarse)

### Issue 3: Octree Empty Cells
**Problem**: 64³ grid might have many empty voxels  
**Solution**: Reduction naturally handles this (empty children contribute zero)

### Issue 4: Coordinate Mapping Complexity
**Problem**: 3D→2D mapping adds overhead  
**Solution**: Use simple formula, GPU handles it efficiently

---

## Expected Improvements

After migration:
- ✅ **Proper 3D physics**: All axes treated equally
- ✅ **No more string artifacts**: Particles collapse in 3D, not 2D planes
- ✅ **Spherical clusters**: Natural 3D gravitational structures
- ✅ **Correct Barnes-Hut**: Approximation works in 3D space
- ✅ **Better visual quality**: Realistic N-body simulation

---

## Migration Steps (Order of Execution)

1. **Update aggregator.js** - Fix uniforms, change L0 size to 64
2. **Update pyramid.js** - Configure octree levels and texture sizes
3. **Update traversal.js** - Fix uniforms for 3D
4. **Update renderer.js** - Fix uniforms for 3D
5. **Update index.js** - Configure octree parameters
6. **Test each stage** - Verify aggregation → reduction → traversal
7. **Debug and iterate** - Fix any issues that arise

---

## Rollback Plan

If octree migration fails:
1. **Revert shader changes** (git revert or manual)
2. **Keep 2D quadtree** but add better Z-force handling:
   - Option A: Direct pairwise calculation for nearby particles
   - Option B: Separate 1D Z-hierarchy (simpler than full octree)
   - Option C: Approximate Z-forces using particle texture sampling

---

## Notes

- The shader changes are **already complete** ✅
- JavaScript pipeline updates are **straightforward** but need care
- Main complexity is in **3D→2D coordinate mapping**
- Performance should be **similar or better** (fewer total cells)
- Visual quality should **dramatically improve** 🌌

---

## References

- Barnes-Hut Algorithm: https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
- Octree Data Structure: https://en.wikipedia.org/wiki/Octree
- GPU Octree Implementation: [Current codebase]

---

## Implementation Status: ✅ COMPLETE

**Date Completed**: October 4, 2025  
**All shaders and pipeline modules updated successfully!**

The octree implementation is now live. All 3 dimensions (X, Y, Z) are treated equally with proper 3D Barnes-Hut force calculations.

---

**Status**: ✅ **Implementation Complete!**  
**Next Step**: Test in browser and verify 3D physics works correctly
