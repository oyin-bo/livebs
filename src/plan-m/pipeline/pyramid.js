// Build the quadtree pyramid via 2x reduction passes
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
  gl.scissor(0, 0, ctx.levelTextures[targetLevel].size, ctx.levelTextures[targetLevel].size);
  
  // Bind source texture
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ctx.levelTextures[sourceLevel].texture);
  
  const u_previousLevel = gl.getUniformLocation(ctx.programs.reduction, 'u_previousLevel');
  gl.uniform1i(u_previousLevel, 0);
  
  // Render full-screen quad
  console.log(`Plan M draw: reduction ${sourceLevel}->${targetLevel}`);
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
