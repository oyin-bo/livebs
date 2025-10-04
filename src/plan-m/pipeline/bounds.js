// Approximate min/max XY from current position texture via sparse GPU readback
export function updateWorldBoundsFromTexture(ctx, sampleCount = 256) {
  const gl = ctx.gl;
  if (!gl || !ctx.positionTextures) return;

  const fbos = ctx.positionTextures.framebuffers;
  const idx = ctx.positionTextures.currentIndex;
  const fbo = fbos[idx];
  if (!fbo) return;

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.readBuffer(gl.COLOR_ATTACHMENT0);

  let minX = Infinity, minY = Infinity;
  let maxX = -Infinity, maxY = -Infinity;
  const px = new Float32Array(4);

  // Low-discrepancy traversal over the texture
  const w = ctx.textureWidth;
  const h = ctx.textureHeight;
  const step = Math.max(1, Math.floor(Math.sqrt((w * h) / sampleCount)));
  let count = 0;
  for (let y = 0; y < h && count < sampleCount; y += step) {
    for (let x = 0; x < w && count < sampleCount; x += step) {
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.FLOAT, px);
      if (px[3] > 0.0) { // mass filter
        const X = px[0], Y = px[1];
        if (X < minX) minX = X;
        if (Y < minY) minY = Y;
        if (X > maxX) maxX = X;
        if (Y > maxY) maxY = Y;
        count++;
      }
    }
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  if (count > 0 && isFinite(minX) && isFinite(minY) && isFinite(maxX) && isFinite(maxY)) {
    const padX = Math.max(0.5, 0.1 * Math.max(1e-6, (maxX - minX)));
    const padY = Math.max(0.5, 0.1 * Math.max(1e-6, (maxY - minY)));
    ctx.options.worldBounds = {
      min: [minX - padX, minY - padY, ctx.options.worldBounds.min[2]],
      max: [maxX + padX, maxY + padY, ctx.options.worldBounds.max[2]]
    };
  }
}
