/**
 * Performance monitoring and benchmarking for Plan A
 * Tracks FPS, frame timing, memory usage, and provides metrics
 */

export class PerformanceMonitor {
  constructor() {
    this.frameCount = 0;
    this.startTime = performance.now();
    this.frameTimes = [];
    this.maxFrameHistory = 60;
    this.frameStart = 0;
    
    // GPU timer query extension (if available)
    this.gpuTimerExt = null;
    this.gpuQueries = [];
    this.gpuTimings = [];
  }

  /**
   * Initialize GPU timing if available
   */
  initGPUTiming(gl) {
    try {
      const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
      
      // Validate that all required methods exist
      if (ext && 
          typeof ext.createQueryEXT === 'function' &&
          typeof ext.deleteQueryEXT === 'function' &&
          typeof ext.beginQueryEXT === 'function' &&
          typeof ext.endQueryEXT === 'function' &&
          typeof ext.queryCounterEXT === 'function' &&
          typeof ext.getQueryObjectEXT === 'function' &&
          ext.TIME_ELAPSED_EXT !== undefined) {
        
        this.gpuTimerExt = ext;
        console.log('GPU timing enabled');
      } else {
        console.log('GPU timing extension incomplete or unavailable');
        this.gpuTimerExt = null;
      }
    } catch (error) {
      console.warn('GPU timing initialization failed:', error);
      this.gpuTimerExt = null;
    }
  }

  /**
   * Begin frame measurement
   */
  beginFrame() {
    this.frameStart = performance.now();
    
    // Start GPU timing if available
    if (this.gpuTimerExt) {
      try {
        const query = this.gpuTimerExt.createQueryEXT();
        this.gpuTimerExt.beginQueryEXT(this.gpuTimerExt.TIME_ELAPSED_EXT, query);
        this.gpuQueries.push({ query, frameCount: this.frameCount });
      } catch (error) {
        console.warn('GPU timing beginFrame failed:', error);
        // Disable GPU timing if it fails
        this.gpuTimerExt = null;
      }
    }
  }

  /**
   * End frame measurement
   */
  endFrame() {
    const frameTime = performance.now() - this.frameStart;
    this.frameTimes.push(frameTime);
    
    // Maintain rolling window
    if (this.frameTimes.length > this.maxFrameHistory) {
      this.frameTimes.shift();
    }
    
    this.frameCount++;

    // End GPU timing if available
    if (this.gpuTimerExt) {
      try {
        this.gpuTimerExt.endQueryEXT(this.gpuTimerExt.TIME_ELAPSED_EXT);
      } catch (error) {
        console.warn('GPU timing endFrame failed:', error);
        this.gpuTimerExt = null;
      }
    }

    // Process completed GPU queries
    this.processGPUQueries();
  }

  /**
   * Process completed GPU timing queries
   */
  processGPUQueries() {
    if (!this.gpuTimerExt) return;

    // Check for completed queries
    for (let i = this.gpuQueries.length - 1; i >= 0; i--) {
      const queryInfo = this.gpuQueries[i];
      
      try {
        if (this.gpuTimerExt.getQueryObjectEXT(queryInfo.query, this.gpuTimerExt.QUERY_RESULT_AVAILABLE_EXT)) {
          const gpuTime = this.gpuTimerExt.getQueryObjectEXT(queryInfo.query, this.gpuTimerExt.QUERY_RESULT_EXT);
          this.gpuTimings.push(gpuTime / 1000000); // Convert to milliseconds
          
          // Maintain rolling window
          if (this.gpuTimings.length > this.maxFrameHistory) {
            this.gpuTimings.shift();
          }
          
          this.gpuTimerExt.deleteQueryEXT(queryInfo.query);
          this.gpuQueries.splice(i, 1);
        }
      } catch (error) {
        console.warn('GPU timing query processing failed:', error);
        // Clean up the failed query
        try {
          this.gpuTimerExt.deleteQueryEXT(queryInfo.query);
        } catch (deleteError) {
          // Ignore delete errors
        }
        this.gpuQueries.splice(i, 1);
        // Disable GPU timing if it keeps failing
        this.gpuTimerExt = null;
        break;
      }
    }
  }

  /**
   * Get current performance metrics
   */
  getMetrics() {
    if (this.frameTimes.length === 0) {
      return {
        fps: 0,
        frameTime: 0,
        frameTimeP95: 0,
        frameTimeP99: 0,
        gpuTimeMs: 0,
        totalFrames: this.frameCount,
        uptimeSeconds: 0
      };
    }

    const avgFrameTime = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    const sortedFrameTimes = [...this.frameTimes].sort((a, b) => a - b);
    const p95Index = Math.floor(sortedFrameTimes.length * 0.95);
    const p99Index = Math.floor(sortedFrameTimes.length * 0.99);
    
    const avgGpuTime = this.gpuTimings.length > 0 
      ? this.gpuTimings.reduce((a, b) => a + b, 0) / this.gpuTimings.length 
      : 0;

    return {
      fps: 1000 / avgFrameTime,
      frameTime: avgFrameTime,
      frameTimeP95: sortedFrameTimes[p95Index] || 0,
      frameTimeP99: sortedFrameTimes[p99Index] || 0,
      gpuTimeMs: avgGpuTime,
      totalFrames: this.frameCount,
      uptimeSeconds: (performance.now() - this.startTime) / 1000
    };
  }

  /**
   * Estimate memory usage for given configuration
   */
  estimateMemoryUsage(particleCount, useFloat) {
    const bytesPerTexel = useFloat ? 16 : 4; // RGBA32F vs RGBA8
    const textureSize = Math.ceil(Math.sqrt(particleCount));
    const texelsPerTexture = textureSize * textureSize;
    const textureCount = 4; // positions, velocities × 2 (ping-pong)
    
    return {
      textureSizePx: textureSize,
      totalTexels: texelsPerTexture * textureCount,
      memoryMB: (texelsPerTexture * textureCount * bytesPerTexel) / (1024 * 1024)
    };
  }

  /**
   * Reset all measurements
   */
  reset() {
    this.frameCount = 0;
    this.startTime = performance.now();
    this.frameTimes = [];
    this.gpuTimings = [];
    
    // Clean up any pending GPU queries
    if (this.gpuTimerExt) {
      this.gpuQueries.forEach(queryInfo => {
        this.gpuTimerExt.deleteQueryEXT(queryInfo.query);
      });
      this.gpuQueries = [];
    }
  }

  /**
   * Clean up resources
   */
  dispose() {
    this.reset();
  }
}

// Backward compatibility with existing Bench class
export class Bench extends PerformanceMonitor {
  startFrame() {
    this.beginFrame();
  }

  endFrame() {
    super.endFrame();
  }

  getStats() {
    const metrics = this.getMetrics();
    return {
      mean: metrics.frameTime,
      median: metrics.frameTime, // Approximation
      p90: metrics.frameTimeP95, // Close enough
      samples: this.frameTimes.length
    };
  }

  toCSVLine(info = {}) {
    const s = this.getStats();
    const parts = [new Date().toISOString(), s.mean.toFixed(3), s.median.toFixed(3), s.p90.toFixed(3), s.samples];
    for (const k of Object.keys(info)) parts.push(info[k]);
    return parts.join(',');
  }
}
