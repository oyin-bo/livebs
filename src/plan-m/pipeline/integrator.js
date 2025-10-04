// Velocity and position integration passes
export function integratePhysics(ctx) {
  const gl = ctx.gl;

  // 1) Update velocities using forces
  gl.useProgram(ctx.programs.velIntegrate);
  ctx.unbindAllTextures();
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.velocityTextures.getTargetFramebuffer());
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  gl.viewport(0, 0, ctx.textureWidth, ctx.textureHeight);
  gl.disable(gl.SCISSOR_TEST);
  gl.scissor(0, 0, ctx.textureWidth, ctx.textureHeight);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ctx.velocityTextures.getCurrentTexture());
  gl.uniform1i(gl.getUniformLocation(ctx.programs.velIntegrate, 'u_velocity'), 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, ctx.forceTexture.texture);
  gl.uniform1i(gl.getUniformLocation(ctx.programs.velIntegrate, 'u_force'), 1);

  gl.uniform2f(gl.getUniformLocation(ctx.programs.velIntegrate, 'u_texSize'), ctx.textureWidth, ctx.textureHeight);
  gl.uniform1i(gl.getUniformLocation(ctx.programs.velIntegrate, 'u_particleCount'), ctx.options.particleCount);
  gl.uniform1f(gl.getUniformLocation(ctx.programs.velIntegrate, 'u_dt'), ctx.options.dt);
  gl.uniform1f(gl.getUniformLocation(ctx.programs.velIntegrate, 'u_damping'), ctx.options.damping);
  gl.uniform1f(gl.getUniformLocation(ctx.programs.velIntegrate, 'u_maxSpeed'), ctx.options.maxSpeed);
  gl.uniform1f(gl.getUniformLocation(ctx.programs.velIntegrate, 'u_maxAccel'), ctx.options.maxAccel);

  gl.bindVertexArray(ctx.quadVAO);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
  // Unbind textures used in this pass
  ctx.unbindAllTextures();
  ctx.checkGl('velIntegrate');
  ctx.velocityTextures.swap();

  // 2) Update positions using new velocities
  gl.useProgram(ctx.programs.posIntegrate);
  ctx.unbindAllTextures();
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.positionTextures.getTargetFramebuffer());
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  gl.viewport(0, 0, ctx.textureWidth, ctx.textureHeight);
  gl.disable(gl.SCISSOR_TEST);
  gl.scissor(0, 0, ctx.textureWidth, ctx.textureHeight);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ctx.positionTextures.getCurrentTexture());
  gl.uniform1i(gl.getUniformLocation(ctx.programs.posIntegrate, 'u_positions'), 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, ctx.velocityTextures.getCurrentTexture());
  gl.uniform1i(gl.getUniformLocation(ctx.programs.posIntegrate, 'u_velocity'), 1);

  gl.uniform2f(gl.getUniformLocation(ctx.programs.posIntegrate, 'u_texSize'), ctx.textureWidth, ctx.textureHeight);
  gl.uniform1i(gl.getUniformLocation(ctx.programs.posIntegrate, 'u_particleCount'), ctx.options.particleCount);
  gl.uniform1f(gl.getUniformLocation(ctx.programs.posIntegrate, 'u_dt'), ctx.options.dt);

  gl.bindVertexArray(ctx.quadVAO);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
  // Unbind textures used in this pass
  ctx.unbindAllTextures();
  ctx.checkGl('posIntegrate');
  ctx.positionTextures.swap();
  // Unbind FBO
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
