# Plan A - Texture-Based Brute Force with Stochastic Sampling

Plan A is a GPU-driven particle system that uses texture-based storage and stochastic neighbor sampling to simulate particle interactions efficiently.

## Features

- **GPU-Accelerated**: All particle physics computed on GPU using WebGL2 compute shaders
- **Ping-Pong Textures**: Position and velocity data stored in RGBA32F textures with ping-pong buffering
- **Stochastic Sampling**: Reduces O(N²) complexity by sampling only a fraction of neighbors per frame
- **Temporal Accumulation**: Approximates full N×N forces over time through partial sampling
- **Fallback Support**: Automatic fallback to 8-bit texture packing for devices without float texture support
- **Performance Monitoring**: Built-in FPS, frame time, and memory usage tracking

## Quick Start

```javascript
import PlanA from './src/plan-a/index.js';

// Get WebGL2 context
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl2');

// Create and initialize Plan A
const planA = new PlanA(gl, {
  particleCount: 100000,
  samplingFraction: 0.25,
  worldBounds: { min: [-10, -10, -10], max: [10, 10, 10] }
});

await planA.init();

// Animation loop
function animate() {
  // Update physics
  planA.step();
  
  // Render particles
  const projectionView = getProjectionViewMatrix(); // Your camera matrix
  planA.render(projectionView);
  
  requestAnimationFrame(animate);
}
animate();
```

## API Reference

### Constructor
```javascript
new PlanA(gl, options)
```

### Methods

#### `async init()`
Initialize GPU resources and validate capabilities. Must be called before using other methods.

#### `step()`
Run one simulation step (update + internal state advancement).

#### `render(projectionViewMatrix)`
Render particles using provided camera matrix.

#### `resize(newParticleCount)`
Safely resize particle count with resource reallocation.

#### `setSamplingFraction(fraction)`
Update sampling fraction at runtime (0.01-1.0).

#### `readback(maxCount = 1000)`
Read particle positions back to CPU for testing/debugging (slow operation).

#### `getMetrics()`
Get performance metrics: `{fps, frameTime, memoryMB, particleCount, samplingFraction, frameCount}`.

#### `dispose()`
Clean up all GPU resources.

## Performance Guidelines

### Recommended Settings by Hardware

| Hardware Class | Particle Count | Sampling Fraction | Expected FPS |
|----------------|----------------|-------------------|--------------|
| Desktop High   | 1,000,000      | 0.25              | 60+ fps      |
| Desktop Mid    | 500,000        | 0.25              | 60+ fps      |
| Desktop Low    | 100,000        | 0.25              | 30+ fps      |
| Mobile High    | 50,000         | 0.20              | 30+ fps      |
| Mobile Low     | 10,000         | 0.15              | 30+ fps      |

## Testing

```javascript
// Basic smoke test
const planA = new PlanA(gl, { particleCount: 1000 });
await planA.init();
planA.step();
const particles = planA.readback(10);
console.log('First 10 particles:', particles);
```

For comprehensive testing, see the test specifications in `docs/1-plan.md`.
