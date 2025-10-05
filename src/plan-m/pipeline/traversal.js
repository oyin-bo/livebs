// Barnes–Hut traversal to compute force texture
export function calculateForces(ctx) {
  const gl = ctx.gl;
  gl.useProgram(ctx.programs.traversal);
  // Avoid feedback
  ctx.unbindAllTextures();

  // Bind force framebuffer
  gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.forceTexture.framebuffer);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  gl.viewport(0, 0, ctx.textureWidth, ctx.textureHeight);
  gl.disable(gl.SCISSOR_TEST);
  gl.scissor(0, 0, ctx.textureWidth, ctx.textureHeight);

  // Bind particle positions
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, ctx.positionTextures.getCurrentTexture());
  gl.uniform1i(gl.getUniformLocation(ctx.programs.traversal, 'u_particlePositions'), 0);

  // Bind quadtree levels
  for (let i = 0; i < Math.min(8, ctx.numLevels); i++) {
    gl.activeTexture(gl.TEXTURE1 + i);
    gl.bindTexture(gl.TEXTURE_2D, ctx.levelTextures[i].texture);
    const loc = gl.getUniformLocation(ctx.programs.traversal, `u_quadtreeLevel${i}`);
    if (loc) gl.uniform1i(loc, 1 + i);
  }

  // Uniforms
  gl.uniform1f(gl.getUniformLocation(ctx.programs.traversal, 'u_theta'), ctx.options.theta);
  gl.uniform1i(gl.getUniformLocation(ctx.programs.traversal, 'u_numLevels'), ctx.numLevels);
  gl.uniform2f(gl.getUniformLocation(ctx.programs.traversal, 'u_texSize'), ctx.textureWidth, ctx.textureHeight);
  gl.uniform1i(gl.getUniformLocation(ctx.programs.traversal, 'u_particleCount'), ctx.options.particleCount);
  gl.uniform3f(gl.getUniformLocation(ctx.programs.traversal, 'u_worldMin'), 
    ctx.options.worldBounds.min[0], 
    ctx.options.worldBounds.min[1],
    ctx.options.worldBounds.min[2]);
  gl.uniform3f(gl.getUniformLocation(ctx.programs.traversal, 'u_worldMax'), 
    ctx.options.worldBounds.max[0], 
    ctx.options.worldBounds.max[1],
    ctx.options.worldBounds.max[2]);
  gl.uniform1f(gl.getUniformLocation(ctx.programs.traversal, 'u_softening'), ctx.options.softening);
  gl.uniform1f(gl.getUniformLocation(ctx.programs.traversal, 'u_G'), ctx.options.gravityStrength);

  // Cell sizes, grid sizes, and slices per row per level
  const cellSizes = new Float32Array(8);
  const gridSizes = new Float32Array(8);
  const slicesPerRow = new Float32Array(8);
  
  // World extent (use max dimension for isotropic cell size)
  const worldExtent = [
    ctx.options.worldBounds.max[0] - ctx.options.worldBounds.min[0],
    ctx.options.worldBounds.max[1] - ctx.options.worldBounds.min[1],
    ctx.options.worldBounds.max[2] - ctx.options.worldBounds.min[2]
  ];
  const worldSize = Math.max(...worldExtent);
  
  // Calculate per-level parameters
  let currentGridSize = ctx.octreeGridSize;
  let currentSlicesPerRow = ctx.octreeSlicesPerRow;
  let cellSize = worldSize / currentGridSize;
  
  for (let i = 0; i < 8; i++) {
    cellSizes[i] = cellSize;
    gridSizes[i] = currentGridSize;
    slicesPerRow[i] = currentSlicesPerRow;
    
    // Next level: halve grid dimensions
    currentGridSize = Math.max(1, Math.floor(currentGridSize / 2));
    currentSlicesPerRow = Math.max(1, Math.floor(currentSlicesPerRow / 2));
    cellSize *= 2.0;
  }
  
  gl.uniform1fv(gl.getUniformLocation(ctx.programs.traversal, 'u_cellSizes'), cellSizes);
  gl.uniform1fv(gl.getUniformLocation(ctx.programs.traversal, 'u_gridSizes'), gridSizes);
  gl.uniform1fv(gl.getUniformLocation(ctx.programs.traversal, 'u_slicesPerRow'), slicesPerRow);

  // Draw quad
  console.log('Plan M draw: calculateForces');
  ctx.checkFBO('calculateForces');
  gl.bindVertexArray(ctx.quadVAO);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);

  // Debug: read back one force value
  if (ctx.frameCount < 3) {
    const px = new Float32Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    console.log(`Frame ${ctx.frameCount}: Force on P0: [${px[0].toFixed(5)}, ${px[1].toFixed(5)}, ${px[2].toFixed(5)}]`);
  }

  // Unbind textures
  ctx.unbindAllTextures();
  ctx.checkGl('calculateForces');
  // Unbind FBO
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
