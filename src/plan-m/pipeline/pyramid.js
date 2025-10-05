// Build the octree pyramid via 2x2x2 reduction passes (8 children → 1 parent)
export function runReductionPass(ctx, sourceLevel, targetLevel) {
  const gl = ctx.gl;
  gl.useProgram(ctx.programs.reduction);
  // Avoid feedback
  ctx.unbindAllTextures();

  // Bind target framebuffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.levelFramebuffers[targetLevel]);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  gl.viewport(0, 0, ctx.levelTextures[targetLevel].size, ctx.levelTextures[targetLevel].size);
  gl.disable(gl.SCISSOR_TEST);
  
  // Bind source texture
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ctx.levelTextures[sourceLevel].texture);
  
  const u_previousLevel = gl.getUniformLocation(ctx.programs.reduction, 'u_previousLevel');
  const u_prevGridSize = gl.getUniformLocation(ctx.programs.reduction, 'u_prevGridSize');
  const u_currGridSize = gl.getUniformLocation(ctx.programs.reduction, 'u_currGridSize');
  const u_textureWidth = gl.getUniformLocation(ctx.programs.reduction, 'u_textureWidth');
  
  gl.uniform1i(u_previousLevel, 0);
  gl.uniform1i(u_prevGridSize, ctx.levelGridSizes[sourceLevel]);
  gl.uniform1i(u_currGridSize, ctx.levelGridSizes[targetLevel]);
  gl.uniform1i(u_textureWidth, ctx.levelTextures[targetLevel].size);
  
  // Render full-screen quad
  console.log(`Plan M draw: octree reduction ${sourceLevel}->${targetLevel} (${ctx.levelGridSizes[sourceLevel]}³ → ${ctx.levelGridSizes[targetLevel]}³)`);
  ctx.checkFBO(`reduction ${sourceLevel}->${targetLevel}`);
  gl.bindVertexArray(ctx.quadVAO);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
  // Unbind source texture
  ctx.unbindAllTextures();
  ctx.checkGl(`runReductionPass ${sourceLevel}->${targetLevel}`);
  // Unbind FBO
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
