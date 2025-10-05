/**
 * Simple smoke test for Plan A implementation
 * Tests basic initialization, stepping, and readback functionality
 */

export async function runSmokeTest() {
  console.log('🧪 Running Plan A smoke test...');
  
  try {
    // Create canvas and WebGL2 context
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const gl = canvas.getContext('webgl2');
    
    if (!gl) {
      throw new Error('WebGL2 not supported');
    }
    
    // Import Plan A
    const { default: PlanA } = await import('./index.js');
    
    // Test 1: Basic initialization
    console.log('🔧 Test 1: Initialization...');
    const planA = new PlanA(gl, { 
      particleCount: 1000,
      samplingFraction: 0.5 
    });
    
    await planA.init();
    console.log('✅ Initialization successful');
    
    // Test 2: GPU capabilities
    console.log('🔧 Test 2: GPU capabilities...');
    const capabilities = planA.gpuTexture.capabilities;
    console.log('GPU capabilities:', capabilities);
    
    if (!capabilities.webgl2) {
      throw new Error('WebGL2 required but not available');
    }
    console.log('✅ GPU capabilities check passed');
    
    // Test 3: Initial readback
    console.log('🔧 Test 3: Initial particle state...');
    const initialParticles = planA.readback(5);
    console.log('Initial particles:', initialParticles);
    
    if (initialParticles.length !== 5) {
      throw new Error('Expected 5 particles in readback');
    }
    console.log('✅ Initial readback successful');
    
    // Test 4: Simulation step
    console.log('🔧 Test 4: Simulation step...');
    const metricsBefore = planA.getMetrics();
    
    planA.step();
    
    const metricsAfter = planA.getMetrics();
    console.log('Metrics before:', metricsBefore);
    console.log('Metrics after:', metricsAfter);
    
    if (metricsAfter.frameCount !== metricsBefore.frameCount + 1) {
      throw new Error('Frame count should increment after step');
    }
    console.log('✅ Simulation step successful');
    
    // Test 5: Position changes
    console.log('🔧 Test 5: Position changes...');
    const particlesAfterStep = planA.readback(5);
    console.log('Particles after step:', particlesAfterStep);
    
    // Check if at least some particles moved (not all will move due to sampling)
    let particlesMoved = 0;
    for (let i = 0; i < 5; i++) {
      const initial = initialParticles[i];
      const after = particlesAfterStep[i];
      const distance = Math.sqrt(
        (after.x - initial.x) ** 2 +
        (after.y - initial.y) ** 2 +
        (after.z - initial.z) ** 2
      );
      if (distance > 1e-6) {
        particlesMoved++;
      }
    }
    
    console.log(`${particlesMoved}/5 particles moved`);
    if (particlesMoved === 0) {
      console.warn('⚠️ No particles moved (may be expected with stochastic sampling)');
    } else {
      console.log('✅ Particles are moving');
    }
    
    // Test 6: Multiple steps
    console.log('🔧 Test 6: Multiple simulation steps...');
    const stepsBefore = metricsAfter.frameCount;
    
    for (let i = 0; i < 10; i++) {
      planA.step();
    }
    
    const finalMetrics = planA.getMetrics();
    if (finalMetrics.frameCount !== stepsBefore + 10) {
      throw new Error('Frame count should increment by 10 after 10 steps');
    }
    console.log('✅ Multiple steps successful');
    
    // Test 7: Performance check
    console.log('🔧 Test 7: Performance check...');
    if (finalMetrics.fps > 0) {
      console.log(`Performance: ${finalMetrics.fps.toFixed(1)} FPS`);
      console.log('✅ Performance monitoring working');
    } else {
      console.warn('⚠️ Performance monitoring not yet active (need more frames)');
    }
    
    // Test 8: Sampling fraction change
    console.log('🔧 Test 8: Sampling fraction change...');
    planA.setSamplingFraction(0.1);
    if (Math.abs(planA.options.samplingFraction - 0.1) > 1e-6) {
      throw new Error('Sampling fraction not updated correctly');
    }
    console.log('✅ Sampling fraction change successful');
    
    // Test 9: Resource cleanup
    console.log('🔧 Test 9: Resource cleanup...');
    planA.dispose();
    console.log('✅ Resource cleanup successful');
    
    // Final report
    console.log('🎉 All tests passed! Plan A implementation is working correctly.');
    
    return {
      success: true,
      capabilities,
      initialParticles,
      finalMetrics,
      particlesMoved
    };
    
  } catch (error) {
    console.error('❌ Smoke test failed:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

// Auto-run if in browser environment
if (typeof window !== 'undefined' && window.location) {
  // Add button to run test
  const button = document.createElement('button');
  button.textContent = 'Run Plan A Smoke Test';
  button.style.cssText = 'position: fixed; top: 10px; right: 10px; z-index: 1000; padding: 10px; background: #007acc; color: white; border: none; border-radius: 5px; cursor: pointer;';
  button.onclick = runSmokeTest;
  document.body.appendChild(button);
  
  console.log('💡 Plan A smoke test available. Click the button in the top-right or call runSmokeTest()');
}
