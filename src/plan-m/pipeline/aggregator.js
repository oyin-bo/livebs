// Aggregate particles into L0 using additive blending.
// Each particle contributes: (sum_x, sum_y, sum_z, mass)
export function aggregateParticlesIntoL0(ctx) {
  const gl = ctx.gl;
  
  // DEBUG: Check first 4 particles' positions before aggregation
  if (ctx.frameCount < 3) {
    // Temporarily bind position texture for readback
    const currentPos = ctx.positionTextures.currentIndex;
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.positionTextures.framebuffers[currentPos]);
    const posData = new Float32Array(16); // 4 particles * 4 components
    gl.readPixels(0, 0, 4, 1, gl.RGBA, gl.FLOAT, posData);
    console.log(`[DEBUG Frame ${ctx.frameCount}] P0-3 positions: P0=[${posData[0].toFixed(2)}, ${posData[1].toFixed(2)}, ${posData[2].toFixed(2)}] mass=${posData[3].toFixed(2)}, P1=[${posData[4].toFixed(2)}, ${posData[5].toFixed(2)}, ${posData[6].toFixed(2)}] mass=${posData[7].toFixed(2)}`);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  
  gl.useProgram(ctx.programs.aggregation);

  // Avoid feedback: ensure no textures are bound except the ones we set below
  ctx.unbindAllTextures();

  // Bind L0 framebuffer and set viewport
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.levelFramebuffers[0]);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  const L0 = ctx.levelTextures[0].size;
  gl.viewport(0, 0, L0, L0);
  gl.disable(gl.SCISSOR_TEST);
  ctx.checkFBO('aggregate L0 (after bind)');

  // Enable additive blending for accumulation
  gl.disable(gl.DEPTH_TEST);
  if (!ctx._disableFloatBlend) {
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
  } else {
    gl.disable(gl.BLEND);
  }

  // Bind positions texture
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ctx.positionTextures.getCurrentTexture());
  const u_positions = gl.getUniformLocation(ctx.programs.aggregation, 'u_positions');
  gl.uniform1i(u_positions, 0);

  // Set uniforms
  const u_texSize = gl.getUniformLocation(ctx.programs.aggregation, 'u_texSize');
  const u_worldMin = gl.getUniformLocation(ctx.programs.aggregation, 'u_worldMin');
  const u_worldMax = gl.getUniformLocation(ctx.programs.aggregation, 'u_worldMax');
  const u_gridSize = gl.getUniformLocation(ctx.programs.aggregation, 'u_gridSize');
  const u_textureWidth = gl.getUniformLocation(ctx.programs.aggregation, 'u_textureWidth');
  gl.uniform2f(u_texSize, ctx.textureWidth, ctx.textureHeight);
  gl.uniform3f(u_worldMin, ctx.options.worldBounds.min[0], ctx.options.worldBounds.min[1], ctx.options.worldBounds.min[2]);
  gl.uniform3f(u_worldMax, ctx.options.worldBounds.max[0], ctx.options.worldBounds.max[1], ctx.options.worldBounds.max[2]);
  gl.uniform1f(u_gridSize, ctx.octreeGridSize); // Use octree grid size (64 for 64³)
  gl.uniform1i(u_textureWidth, L0);

  // Assert we're not simultaneously sampling and rendering into the same texture
  const attachedTex = gl.getFramebufferAttachmentParameter(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
  const posTex = ctx.positionTextures.getCurrentTexture();
  if (attachedTex === posTex) {
    console.error('Plan M: FEEDBACK DETECTED - L0 FBO is the same texture as positions being sampled.');
  }

    // Draw particles as points using attribute from VAO
    // Use particleVAO which has a_particleIndex attribute configured
    console.log('Plan M draw: aggregateParticlesIntoL0');
    ctx.checkFBO('aggregate L0');
    
    gl.bindVertexArray(ctx.particleVAO);
    gl.drawArrays(gl.POINTS, 0, ctx.options.particleCount);  // DEBUG: Readback first 8 L0 voxels to verify data is written
  if (ctx.frameCount < 3) {
    const L0data = new Float32Array(32); // 8 voxels * 4 components
    gl.readPixels(0, 0, 8, 1, gl.RGBA, gl.FLOAT, L0data);
    console.log(`[DEBUG Frame ${ctx.frameCount}] L0[0-7] masses: [${L0data[3].toFixed(4)}, ${L0data[7].toFixed(4)}, ${L0data[11].toFixed(4)}, ${L0data[15].toFixed(4)}, ${L0data[19].toFixed(4)}, ${L0data[23].toFixed(4)}, ${L0data[27].toFixed(4)}, ${L0data[31].toFixed(4)}]`);
  }

  gl.disable(gl.BLEND);
  // Unbind input texture
  ctx.unbindAllTextures();
  ctx.checkGl('aggregateParticlesIntoL0');
  // Unbind FBO
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
