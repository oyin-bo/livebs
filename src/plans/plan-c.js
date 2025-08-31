import * as THREE from 'three';

export default class PlanC {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this._objects = [];
  }

  start() {
    // visual representation of spatial grid cells
    const group = new THREE.Group();
    const cellSize = 1.0;
    const dim = 6;
    const geo = new THREE.PlaneGeometry(cellSize, cellSize);
    for (let x = -dim; x <= dim; x++) {
      for (let y = -dim; y <= dim; y++) {
        const mat = new THREE.MeshBasicMaterial({ color: 0x223344, side: THREE.DoubleSide, transparent: true, opacity: 0.15 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x * cellSize, y * cellSize, 0);
        mesh.rotation.x = -Math.PI / 2;
        group.add(mesh);
      }
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
