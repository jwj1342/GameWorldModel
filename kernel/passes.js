// passes.js — rgb / depth / id 三个渲染 pass，都直接渲到主画布并用 toDataURL 编码 PNG
import * as THREE from 'three';

const depthMaterial = new THREE.ShaderMaterial({
  uniforms: { far: { value: 100.0 } },
  vertexShader: `varying float vViewZ; void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0); vViewZ = -mv.z; gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `uniform float far; varying float vViewZ; void main(){ float d = clamp(vViewZ / far, 0.0, 1.0); float v = d * 65535.0; float hi = floor(v / 256.0); float lo = v - hi * 256.0; gl_FragColor = vec4(hi / 255.0, lo / 255.0, 0.0, 1.0); }`,
});

export function renderPass(renderer, scene, camera, registry, pass, { far = 100 } = {}) {
  const canvas = renderer.domElement;
  const prevBg = scene.background, prevOverride = scene.overrideMaterial;
  const prevColorSpace = renderer.outputColorSpace, prevToneMapping = renderer.toneMapping;
  if (pass === 'depth' || pass === 'id') {
    // encoded passes must not be gamma-converted: write linear bytes exactly
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace; renderer.toneMapping = THREE.NoToneMapping;
  }
  if (pass === 'depth') {
    depthMaterial.uniforms.far.value = far;
    scene.overrideMaterial = depthMaterial;
    scene.background = new THREE.Color(1, 1, 0); // far plane: hi=255, lo=255
    renderer.render(scene, camera);
  } else if (pass === 'id') {
    const swapped = [];
    scene.traverse(o => {
      if (!o.isMesh) return;
      const entry = o.userData.entry;
      swapped.push([o, o.material]);
      o.material = new THREE.MeshBasicMaterial({ color: entry ? entry.idColor : 0x000000 });
    });
    scene.background = new THREE.Color(0, 0, 0);
    renderer.render(scene, camera);
    for (const [o, m] of swapped) { o.material.dispose(); o.material = m; }
  } else {
    renderer.render(scene, camera);
  }
  scene.overrideMaterial = prevOverride; scene.background = prevBg;
  const url = canvas.toDataURL('image/png');
  renderer.outputColorSpace = prevColorSpace; renderer.toneMapping = prevToneMapping;
  return url.slice(url.indexOf(',') + 1);
}

export function decodeDepthPng() { /* decoding happens on the Python side: depth = (R*256 + G) / 65535 * far */ }
