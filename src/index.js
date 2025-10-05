import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import PlanM from './plan-m/index.js';

// Basic renderer + scene + camera
const container = document.getElementById('app');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.domElement.style.width = '100%';
renderer.domElement.style.height = '100%';
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Simple grid + axes
const grid = new THREE.GridHelper(10, 10, 0x222222, 0x222222);
scene.add(grid);

const axes = new THREE.AxesHelper(2);
scene.add(axes);

// Single production plan: Plan M
const planM = new PlanM(scene, renderer);
let activePlan = planM;
activePlan.start();

// Minimal switch function kept for debug only
function switchPlan(key) {
  // noop in production: only Plan M is supported
  console.warn('switchPlan is disabled in production mode. Only Plan M is active.');
}

// Simple keyboard UI: keys 1-4 map to plans A,C,D,M
// Keys + and - to control simulation speed for Plan M
window.addEventListener('keydown', (e) => {
	// Speed controls for Plan M
	if (activePlan) {
		if (e.key === '+' || e.key === '=') {
			activePlan.options.dt *= 2;
			console.log(`Plan M speed up: dt = ${activePlan.options.dt.toFixed(4)}`);
		}
		if (e.key === '-' || e.key === '_') {
			activePlan.options.dt /= 2;
			console.log(`Plan M slow down: dt = ${activePlan.options.dt.toFixed(4)}`);
		}
		if (e.key === '0') {
			activePlan.options.dt = 1 / 60;
			console.log(`Plan M speed reset: dt = ${activePlan.options.dt.toFixed(4)}`);
		}
	}
});

// Basic render loop
function animate() {
	requestAnimationFrame(animate);
	controls.update();
	
	// Render Three.js scene first
	renderer.render(scene, camera);
	
	// Then update and render custom GPU plans (so they draw on top)
	if (activePlan && activePlan.update) activePlan.update();
}

window.addEventListener('resize', () => {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();

// Expose switch function for debug
window.switchPlan = switchPlan;
window.camera = camera; // Expose camera for Plan M