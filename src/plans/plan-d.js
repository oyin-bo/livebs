import * as THREE from 'three';

export default class PlanD {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this._objects = [];
  }

  start() {
    // Multi-resolution visualization: concentric rings with different colors
    const rings = new THREE.Group();
    for (let i = 1; i <= 5; i++) {
      const r = i * 0.5;
      const seg = 64;
      const geo = new THREE.RingGeometry(r - 0.02, r + 0.02, seg);
      const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color().setHSL(i / 6, 0.7, 0.5), side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.rotation.x = -Math.PI / 2;
      rings.add(mesh);
    }
    this.scene.add(rings);
    this._objects.push(rings);
  }

  stop() {
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
  }

  update() {}
}
