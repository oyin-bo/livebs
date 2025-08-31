import * as THREE from 'three';

export default class PlanA {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this._objects = [];
  }

  start() {
    // simple points cloud demo
    const geom = new THREE.BufferGeometry();
    const count = 1000;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3 + 0] = (Math.random() - 0.5) * 4;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 4;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 4;
    }
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: 0x66ccff, size: 0.02 });
    const points = new THREE.Points(geom, mat);
    this.scene.add(points);
    this._objects.push(points);
  }

  stop() {
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
  }

  update() {
    // noop
  }
}
