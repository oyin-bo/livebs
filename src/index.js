import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import PlanA from './plan-a/index.js';
import PlanC from './plan-c/index.js';
import PlanD from './plan-d/index.js';
import PlanM from './plan-m/index.js';

// Basic renderer + scene + camera
const container = document.getElementById('app');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 20);
// Add camera to scene so Plan A can find it
scene.add(camera);

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

// Plans registry
const plans = {
	a: new PlanA(scene, renderer),
	c: new PlanC(scene, renderer),
	d: new PlanD(scene, renderer),
	m: new PlanM(scene, renderer),
};

let activePlan = plans.a;
activePlan.start();

function switchPlan(key) {
	if (activePlan && activePlan.stop) activePlan.stop();
	activePlan = plans[key];
	if (activePlan && activePlan.start) activePlan.start();
	// update HUD active button
	try {
		const hud = document.getElementById('hud');
		if (hud) {
			hud.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.getAttribute('data-plan') === key));
		}
	} catch (err) { /* ignore */ }
}

// Simple keyboard UI: keys 1-4 map to plans A,C,D,M
window.addEventListener('keydown', (e) => {
	if (e.key === '1') switchPlan('a');
	if (e.key === '2') switchPlan('c');
	if (e.key === '3') switchPlan('d');
	if (e.key === '4') switchPlan('m');
});

// Basic render loop
function animate() {
	requestAnimationFrame(animate);
	controls.update();
	if (activePlan && activePlan.update) activePlan.update();
	renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();

// Expose switch function for debug
window.switchPlan = switchPlan;