// WebGL debug utilities for Plan M

/**
 * Unbind all textures from the first N units to reduce risk of feedback loops.
 * @param {WebGL2RenderingContext} gl 
 * @param {number} maxUnitsHint Optional hint to limit how many units to clear (default 16)
 */
export function unbindAllTextures(gl, maxUnitsHint = 16) {
  if (!gl) return;
  const maxUnits = Math.min(maxUnitsHint, gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) || maxUnitsHint);
  for (let i = 0; i < maxUnits; i++) {
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
  gl.activeTexture(gl.TEXTURE0);
}

/**
 * Check for GL error and log with a tag; returns the error enum.
 * @param {WebGL2RenderingContext} gl 
 * @param {string} tag 
 */
export function checkGl(gl, tag = '') {
  if (!gl) return 0;
  const err = gl.getError();
  if (err !== gl.NO_ERROR) {
    console.error(`Plan M GL error after ${tag}: 0x${err.toString(16)}`);
  }
  return err;
}

/**
 * Check framebuffer completeness and log if incomplete.
 * @param {WebGL2RenderingContext} gl 
 * @param {string} tag 
 */
export function checkFBO(gl, tag = '') {
  if (!gl) return;
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.error(`Plan M FBO incomplete at ${tag}: 0x${status.toString(16)}`);
  }
}
