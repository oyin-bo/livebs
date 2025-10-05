// Render particles from the current position texture to the default framebuffer
export function renderParticles(ctx) {
  if (!ctx.renderer || !ctx.scene) return;

  const camera = ctx.getCameraFromScene();
  if (!camera) {
    console.warn('Plan M: No camera found for rendering');
    return;
  }

  const gl = ctx.gl;

  // Save WebGL state
  const oldViewport = gl.getParameter(gl.VIEWPORT);
  const oldProgram = gl.getParameter(gl.CURRENT_PROGRAM);
  const oldFramebuffer = gl.getParameter(gl.FRAMEBUFFER_BINDING);

  try {
    // Use render program
    gl.useProgram(ctx.programs.render);

    // Default framebuffer (screen) - DON'T restore old viewport, use full canvas!
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.disable(gl.SCISSOR_TEST);

    // Bind particle positions
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, ctx.positionTextures.getCurrentTexture());

    // Calculate projection-view matrix
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();
    const projectionMatrix = camera.projectionMatrix;
    const viewMatrix = camera.matrixWorldInverse;
    const projectionViewMatrix = new Float32Array(16);
    
    // Multiply projection * view manually (column-major order)
    const p = projectionMatrix.elements;
    const v = viewMatrix.elements;
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        projectionViewMatrix[col * 4 + row] = 
          p[0 * 4 + row] * v[col * 4 + 0] +
          p[1 * 4 + row] * v[col * 4 + 1] +
          p[2 * 4 + row] * v[col * 4 + 2] +
          p[3 * 4 + row] * v[col * 4 + 3];
      }
    }

    // Uniforms
    const u_positions = gl.getUniformLocation(ctx.programs.render, 'u_positions');
    const u_texSize = gl.getUniformLocation(ctx.programs.render, 'u_texSize');
    const u_pointSize = gl.getUniformLocation(ctx.programs.render, 'u_pointSize');
    const u_projectionView = gl.getUniformLocation(ctx.programs.render, 'u_projectionView');
    const u_worldMin = gl.getUniformLocation(ctx.programs.render, 'u_worldMin');
    const u_worldMax = gl.getUniformLocation(ctx.programs.render, 'u_worldMax');

    gl.uniform1i(u_positions, 0);
    gl.uniform2f(u_texSize, ctx.textureWidth, ctx.textureHeight);
    gl.uniform1f(u_pointSize, ctx.options.pointSize);
    if (u_projectionView) gl.uniformMatrix4fv(u_projectionView, false, projectionViewMatrix);
    if (u_worldMin) gl.uniform3f(u_worldMin, ctx.options.worldBounds.min[0], ctx.options.worldBounds.min[1], ctx.options.worldBounds.min[2]);
    if (u_worldMax) gl.uniform3f(u_worldMax, ctx.options.worldBounds.max[0], ctx.options.worldBounds.max[1], ctx.options.worldBounds.max[2]);

    // Blending for particles
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);

    // Draw
    gl.bindVertexArray(ctx.particleVAO);
    gl.drawArrays(gl.POINTS, 0, ctx.options.particleCount);
    gl.bindVertexArray(null);

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);

    if (ctx.frameCount < 3) {
      console.log(`Plan M: Rendered ${ctx.options.particleCount} particles at frame ${ctx.frameCount}`);
      console.log(`Plan M: Camera position:`, camera.position);
      console.log(`Plan M: Point size:`, ctx.options.pointSize);
      console.log(`Plan M: World bounds:`, ctx.options.worldBounds);
      console.log(`Plan M: Texture size: ${ctx.textureWidth}x${ctx.textureHeight}`);
      console.log(`Plan M: Viewport: ${gl.drawingBufferWidth}x${gl.drawingBufferHeight}`);
    }
  } catch (error) {
    console.error('Plan M render error:', error);
  } finally {
    // Don't restore viewport - we want full canvas size for rendering
    // gl.viewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
    gl.useProgram(oldProgram);
    gl.bindFramebuffer(gl.FRAMEBUFFER, oldFramebuffer);
  }
}
