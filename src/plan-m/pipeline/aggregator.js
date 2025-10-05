// Aggregate particles into L0 using additive blending.
// Each particle contributes: (sum_x, sum_y, mass, count)
export function aggregateParticlesIntoL0(ctx) {
  const gl = ctx.gl;
  gl.useProgram(ctx.programs.aggregation);

  // Avoid feedback: ensure no textures are bound except the ones we set below
  ctx.unbindAllTextures();

  // Bind L0 framebuffer and set viewport
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.levelFramebuffers[0]);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  const L0 = ctx.levelTextures[0].size;
  gl.viewport(0, 0, L0, L0);
  gl.disable(gl.SCISSOR_TEST);
  gl.scissor(0, 0, L0, L0);
  ctx.checkFBO('aggregate L0 (after bind)');
  ctx.checkGl('aggregate L0 (after bind)');

  // Enable additive blending for accumulation
  gl.disable(gl.DEPTH_TEST);
  if (!ctx._disableFloatBlend) {
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
  } else {
    gl.disable(gl.BLEND);
  }
  ctx.checkGl('aggregate L0 (after blend setup)');

  // Bind positions texture
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ctx.positionTextures.getCurrentTexture());
  const u_positions = gl.getUniformLocation(ctx.programs.aggregation, 'u_positions');
  gl.uniform1i(u_positions, 0);
  ctx.checkGl('aggregate L0 (after bind positions)');

  // Set uniforms
  const u_texSize = gl.getUniformLocation(ctx.programs.aggregation, 'u_texSize');
  const u_worldMin = gl.getUniformLocation(ctx.programs.aggregation, 'u_worldMin');
  const u_worldMax = gl.getUniformLocation(ctx.programs.aggregation, 'u_worldMax');
  const u_gridSize = gl.getUniformLocation(ctx.programs.aggregation, 'u_gridSize');
  const u_slicesPerRow = gl.getUniformLocation(ctx.programs.aggregation, 'u_slicesPerRow');
  
  gl.uniform2f(u_texSize, ctx.textureWidth, ctx.textureHeight);
  gl.uniform3f(u_worldMin, 
    ctx.options.worldBounds.min[0], 
    ctx.options.worldBounds.min[1],
    ctx.options.worldBounds.min[2]);
  gl.uniform3f(u_worldMax, 
    ctx.options.worldBounds.max[0], 
    ctx.options.worldBounds.max[1],
    ctx.options.worldBounds.max[2]);
  gl.uniform1f(u_gridSize, ctx.octreeGridSize);
  gl.uniform1f(u_slicesPerRow, ctx.octreeSlicesPerRow);
  ctx.checkGl('aggregate L0 (after set uniforms)');

  // Assert we're not simultaneously sampling and rendering into the same texture
  const attachedTex = gl.getFramebufferAttachmentParameter(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
  const posTex = ctx.positionTextures.getCurrentTexture();
  if (attachedTex === posTex) {
    console.error('Plan M: FEEDBACK DETECTED - L0 FBO is the same texture as positions being sampled.');
  }

  // Draw particles as points using gl_VertexID
  console.log('Plan M draw: aggregateParticlesIntoL0');
  ctx.checkFBO('aggregate L0');
  ctx.checkGl('aggregate L0 (before draw)');
  gl.bindVertexArray(ctx.particleVAO);
  gl.drawArrays(gl.POINTS, 0, ctx.options.particleCount);
  gl.bindVertexArray(null);
  ctx.checkGl('aggregate L0 (after draw)');

  gl.disable(gl.BLEND);
  // Unbind input texture
  ctx.unbindAllTextures();
  const err = ctx.checkGl('aggregateParticlesIntoL0');
  if (err !== gl.NO_ERROR) {
    if (!ctx._disableFloatBlend) {
      console.warn('Plan M: Disabling float blending for L0 accumulation due to GL error. Falling back to non-blended writes (reduced accuracy).');
      ctx._disableFloatBlend = true;
    } else {
      console.warn('Plan M: Disabling quadtree due to persistent GL errors in aggregation. Forces will be cleared each frame.');
      ctx._quadtreeDisabled = true;
    }
  }
  // Unbind FBO
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
