import * as THREE from 'three';

export default class PlanM {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this._objects = [];
  }

  start() {
    // Quadtree pyramid visualization: scaled quads stacked in z
    const group = new THREE.Group();
    const levels = 6;
    for (let i = 0; i < levels; i++) {
      const size = Math.pow(2, levels - i) * 0.1;
      const geo = new THREE.PlaneGeometry(size, size);
      const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color().setHSL(i / levels, 0.6, 0.5), side: THREE.DoubleSide, transparent: true, opacity: 0.25 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(0, 0, -i * 0.02);
      mesh.rotation.x = -Math.PI / 2;
      group.add(mesh);
    }
    this.scene.add(group);
    this._objects.push(group);
  }

  stop() {
    this._objects.forEach(o => this.scene.remove(o));
    this._objects = [];
  }

  update() {}
}
