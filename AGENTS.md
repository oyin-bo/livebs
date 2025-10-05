# N-body problem

## Overview: Plan M: Barnes-Hut with Octree

GPU-resident hierarchical force approximation via octree aggregation and tree traversal. Achieves O(N log N) complexity for gravitational N-body dynamics. Target: 50K particles at interactive framerates with physical fidelity.

## Immediate goal: Accessible solution for 50K particles

Render 50K gravitationally interacting particles at ≥10 FPS with stable Barnes-Hut approximation (θ=0.5). Maintain numerical fidelity: conserved total mass, bounded COM drift, reproducible trajectories under fixed seed.

## Debugging tools

The main and core debugging tools is Playwright driving Chrome-based browser, or Firefox-based browser. Both options are valid.

Only minor changes can be accepted without Playwright validation using browser tool.

ALWAYS use Playwright tool for diagnostics, debugging and verification of any changes.

## Mandatory Smoke Test Protocol

A simulation run is **FAILED** if any of these conditions are met across a sequence of **2-3 independent captures**:

- **Visual Failure:** Any captured frame is blank or shows grid/axis artifacts.
- **Numerical Failure:** Any captured frame has `totalMass <= 0` or an out-of-bounds Center of Mass (COM).
- **Error Failure:** Any WebGL error appears in the console during the capture sequence.

A run only **PASSES** if all taken captures meet all criteria.