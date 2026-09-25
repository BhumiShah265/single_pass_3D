// Global Three.js references

const OrbitControls = THREE.OrbitControls;

const GLTFLoader = THREE.GLTFLoader;



document.getElementById('shell').innerHTML = Aero.shell();



const container = document.getElementById('studioViewport');

const canvas = document.getElementById('threeCanvas');

const loadingOverlay = document.getElementById('gisLoadingOverlay');

const loadingText = document.getElementById('gisLoadingText');

const studioJob = document.getElementById('studioJob');

const projectMessage = document.getElementById('projectMessage');



// Sidebar stats

const statSparse = document.getElementById('statSparse');

const statDense = document.getElementById('statDense');

const statVertices = document.getElementById('statVertices');

const statTriangles = document.getElementById('statTriangles');

const statReprojection = document.getElementById('statReprojection');

const statGsd = document.getElementById('statGsd');

const statCrs = document.getElementById('statCrs');



// HUD Elements

const gisCoords = document.getElementById('gisCoords');
const estimateNotice = document.getElementById('estimateNotice');
const pointSizeBox = document.getElementById('pointSizeBox');

const pointSizeSlider = document.getElementById('pointSizeSlider');

const pointSizeVal = document.getElementById('pointSizeVal');



// ── Three.js Scene, Camera, Renderer ─────────────────────────────────────────

const scene = new THREE.Scene();

scene.background = new THREE.Color(0x080c14);



const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);

camera.position.set(30, 26, 32);



const renderer = new THREE.WebGLRenderer({

  canvas,

  antialias: true,

  powerPreference: 'high-performance',

  alpha: false,

});

renderer.setPixelRatio(Math.min(Math.max(window.devicePixelRatio || 1, 1) * 1.5, 2.5));
if (THREE.SRGBColorSpace) {

  renderer.outputColorSpace = THREE.SRGBColorSpace;

} else if (THREE.sRGBEncoding) {

  renderer.outputEncoding = THREE.sRGBEncoding;

}

renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;


// OrbitControls

const controls = new OrbitControls(camera, canvas);

controls.enableDamping = true;

controls.dampingFactor = 0.05;

controls.screenSpacePanning = true;

controls.maxPolarAngle = Math.PI * 0.49; // Keep camera strictly above ground plane for aerial survey

controls.minDistance = 0.5;

controls.maxDistance = 600;

controls.target.set(0, 0, 0);



// Lighting

const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambientLight);



const sunLight = new THREE.DirectionalLight(0xffffff, 0.9);
sunLight.position.set(50, 70, 40);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.left = -100;
sunLight.shadow.camera.right = 100;
sunLight.shadow.camera.top = 100;
sunLight.shadow.camera.bottom = -100;
sunLight.shadow.camera.near = 0.5;
sunLight.shadow.camera.far = 250;
sunLight.shadow.bias = -0.0002;
sunLight.shadow.normalBias = 0.02;
scene.add(sunLight);



const fillLight = new THREE.DirectionalLight(0x94a3b8, 0.3);
fillLight.position.set(-40, 20, -40);

scene.add(fillLight);



// Coordinate Ground Grid

const grid = new THREE.GridHelper(120, 60, 0x38bdf8, 0x1e293b);

grid.position.y = -0.05;

scene.add(grid);



// ── Scene State ──────────────────────────────────────────────────────────────

let currentMeshGroup = null;
let currentEstimatedGroup = null;
let currentPointsObject = null;

let currentCamerasGroup = null;

const originalMaterials = new Map();

let currentDisplayMode = 'mesh';
let observedStats = null;
let estimatedStats = null;
let sceneRadius = 35;

const sceneBoundingBox = new THREE.Box3();
let jobLoadToken = 0;


// Resize handling

function resize() {

  const width = container.clientWidth;

  const height = container.clientHeight || 640;

  camera.aspect = width / height;

  camera.updateProjectionMatrix();

  renderer.setSize(width, height, false);

}

const resizeObserver = new ResizeObserver(resize);

resizeObserver.observe(container);

resize();



// ── Animation Loop ───────────────────────────────────────────────────────────

function animate() {

  requestAnimationFrame(animate);

  controls.update();

  renderer.render(scene, camera);

}

animate();



// ── Mouse Raycasting for Coordinates & Double-Click Pivot ─────────────────────

const raycaster = new THREE.Raycaster();

const mouse = new THREE.Vector2();



canvas.addEventListener('mousemove', (e) => {

  const rect = canvas.getBoundingClientRect();

  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;

  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;



  raycaster.setFromCamera(mouse, camera);

  const targets = [];

  if (currentMeshGroup && currentMeshGroup.visible) targets.push(currentMeshGroup);
  if (currentEstimatedGroup && currentEstimatedGroup.visible) targets.push(currentEstimatedGroup);
  if (currentPointsObject && currentPointsObject.visible) targets.push(currentPointsObject);



  const intersects = raycaster.intersectObjects(targets, true);

  if (intersects.length > 0) {

    const pt = intersects[0].point;

    gisCoords.textContent = `X: ${pt.x.toFixed(2)}m · Y: ${pt.z.toFixed(2)}m · Elev: ${pt.y.toFixed(2)}m`;

  }

});



canvas.addEventListener('dblclick', (e) => {

  const rect = canvas.getBoundingClientRect();

  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;

  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;



  raycaster.setFromCamera(mouse, camera);

  const targets = [];

  if (currentMeshGroup && currentMeshGroup.visible) targets.push(currentMeshGroup);
  if (currentEstimatedGroup && currentEstimatedGroup.visible) targets.push(currentEstimatedGroup);
  if (currentPointsObject && currentPointsObject.visible) targets.push(currentPointsObject);



  const intersects = raycaster.intersectObjects(targets, true);

  if (intersects.length > 0) {

    const target = intersects[0].point;

    controls.target.copy(target);

    Aero.toast(`Orbit pivot centered on surface`);

  }

});



// Prevent page scroll when zooming canvas via mouse wheel
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
}, { passive: false });



// ── View Presets ─────────────────────────────────────────────────────────────

function setView(viewName) {

  document.querySelectorAll('[id^="btnView"]').forEach(b => b.classList.remove('active'));

  const btn = document.getElementById(viewName === 'top' ? 'btnViewTop' : viewName === 'iso' ? 'btnViewIso' : viewName === 'front' ? 'btnViewFront' : viewName === 'back' ? 'btnViewBack' : 'btnView3D');
  if (btn) btn.classList.add('active');



  const r = Math.max(20, sceneRadius);
  const center = new THREE.Vector3();
  if (sceneBoundingBox && !sceneBoundingBox.isEmpty()) {
    sceneBoundingBox.getCenter(center);
  }
  controls.target.copy(center);

  camera.up.set(0, 1, 0);
  if (viewName === 'top') {
    camera.position.set(center.x, center.y + r * 1.5, center.z + 0.001);
  } else if (viewName === 'iso') {
    camera.position.set(center.x + r * 0.9, center.y + r * 0.9, center.z + r * 0.9);
  } else if (viewName === 'front') {
    camera.position.set(center.x, center.y + r * 0.75, center.z + r * 1.2);
  } else if (viewName === 'back') {
    camera.position.set(center.x, center.y + r * 0.75, center.z - r * 1.2);
  } else {
    camera.position.set(center.x + r * 0.8, center.y + r * 0.75, center.z + r * 0.85);
  }

  controls.update();

}



document.getElementById('btnViewTop').onclick = () => setView('top');

document.getElementById('btnView3D').onclick = () => setView('3d');

document.getElementById('btnViewIso').onclick = () => setView('iso');

document.getElementById('btnViewFront').onclick = () => setView('front');
document.getElementById('btnViewBack').onclick = () => {
  if (!currentEstimatedGroup) return;

  // The photographed mesh can have open edges. Use the closed inferred shell
  // for the opposite-side preset so the rear view does not expose those gaps.
  setDisplayMode('estimated');
  sceneBoundingBox.setFromObject(currentEstimatedGroup);
  const sphere = new THREE.Sphere();
  sceneBoundingBox.getBoundingSphere(sphere);
  sceneRadius = sphere.radius || 35;
  grid.position.y = sceneBoundingBox.min.y - 0.05;
  setView('back');
  Aero.toast('Showing the opposite side of the complete model');
};
document.getElementById('btnResetView').onclick = () => {
  setView('3d');

  Aero.toast('Perspective reset to 3D aerial view');

};



// Fullscreen Toggle

document.getElementById('btnFullscreen').onclick = () => {

  if (!document.fullscreenElement) {

    container.requestFullscreen().catch(() => {});

  } else {

    document.exitFullscreen().catch(() => {});

  }

};



// ── Display Mode Switcher ────────────────────────────────────────────────────

function setDisplayMode(mode) {
  currentDisplayMode = mode;
  document.querySelectorAll('[id^="btnMode"]').forEach(b => b.classList.remove('active'));
  pointSizeBox.style.display = (mode === 'points') ? 'flex' : 'none';
  if (currentEstimatedGroup) currentEstimatedGroup.visible = false;
  estimateNotice.style.display = mode === 'estimated' ? 'inline-flex' : 'none';
  if (observedStats) {
    const inferred = mode === 'estimated' && estimatedStats;
    statVertices.textContent = (inferred ? estimatedStats.vertices : observedStats.vertices).toLocaleString();
    statTriangles.textContent = (inferred ? estimatedStats.triangles : observedStats.triangles).toLocaleString();
    statReprojection.textContent = observedStats.reprojection;
    statGsd.textContent = observedStats.gsd;
  }


  if (mode === 'mesh') {

    document.getElementById('btnModeMesh').classList.add('active');

    if (currentMeshGroup) {

      currentMeshGroup.visible = true;

      currentMeshGroup.traverse((child) => {

        if (child.isMesh && originalMaterials.has(child)) {

          child.material = originalMaterials.get(child);

          child.material.wireframe = false;

        }

      });

    }

    if (currentPointsObject) currentPointsObject.visible = false;

  } else if (mode === 'points') {

    document.getElementById('btnModePoints').classList.add('active');

    if (currentMeshGroup) currentMeshGroup.visible = false;

    if (currentPointsObject) currentPointsObject.visible = true;

  } else if (mode === 'wireframe') {

    document.getElementById('btnModeWireframe').classList.add('active');

    if (currentMeshGroup) {

      currentMeshGroup.visible = true;

      currentMeshGroup.traverse((child) => {

        if (child.isMesh) {

          if (!child.userData.wireframeMat) {

            child.userData.wireframeMat = new THREE.MeshBasicMaterial({ color: 0x4ee6d0, wireframe: true });

          }

          child.material = child.userData.wireframeMat;

        }

      });

    }

    if (currentPointsObject) currentPointsObject.visible = false;

  } else if (mode === 'clay') {
    document.getElementById('btnModeClay').classList.add('active');

    if (currentMeshGroup) {

      currentMeshGroup.visible = true;

      currentMeshGroup.traverse((child) => {

        if (child.isMesh) {

          if (!child.userData.clayMat) {

            child.userData.clayMat = new THREE.MeshLambertMaterial({ color: 0xa8b6c8 });

          }

          child.material = child.userData.clayMat;

        }

      });

    }

    if (currentPointsObject) currentPointsObject.visible = false;
  } else if (mode === 'estimated' && currentEstimatedGroup) {
    document.getElementById('btnModeEstimated').classList.add('active');
    if (currentMeshGroup) currentMeshGroup.visible = false;
    if (currentPointsObject) currentPointsObject.visible = false;
    currentEstimatedGroup.visible = true;
  }
}


document.getElementById('btnModeMesh').onclick = () => setDisplayMode('mesh');

document.getElementById('btnModePoints').onclick = () => setDisplayMode('points');

document.getElementById('btnModeWireframe').onclick = () => setDisplayMode('wireframe');

document.getElementById('btnModeClay').onclick = () => setDisplayMode('clay');
document.getElementById('btnModeEstimated').onclick = () => {
  if (currentEstimatedGroup) setDisplayMode('estimated');
};


// Point Size Slider

pointSizeSlider.oninput = (e) => {

  const sz = parseFloat(e.target.value);

  pointSizeVal.textContent = `${sz}px`;

  if (currentPointsObject && currentPointsObject.material) {

    currentPointsObject.material.size = sz * 0.1;

  }

};



// Grid and Camera Frustums Toggles

document.getElementById('btnToggleGrid').onclick = function() {

  grid.visible = !grid.visible;

  this.classList.toggle('active', grid.visible);

};



document.getElementById('btnToggleCameras').onclick = function() {

  if (currentCamerasGroup) {

    currentCamerasGroup.visible = !currentCamerasGroup.visible;

    this.classList.toggle('active', currentCamerasGroup.visible);

  }

};



// ── Build Drone Camera Frustums & Trajectory ─────────────────────────────────

function buildCameraTrajectory(trajectory) {

  if (currentCamerasGroup) {

    scene.remove(currentCamerasGroup);

    currentCamerasGroup = null;

  }

  if (!trajectory || !trajectory.length) return;



  const group = new THREE.Group();

  const pathPoints = [];

  const pyramidGeo = new THREE.ConeGeometry(0.7, 1.2, 4);

  pyramidGeo.rotateX(Math.PI / 2); // Point along forward axis



  const frustumMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true });

  const cameraPointMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });



  trajectory.forEach((cam) => {

    if (typeof cam.x === 'number' && typeof cam.y === 'number' && typeof cam.z === 'number') {

      const pos = new THREE.Vector3(cam.x, cam.y, cam.z);

      pathPoints.push(pos);



      // Camera body marker

      const marker = new THREE.Mesh(new THREE.SphereGeometry(0.25, 8, 8), cameraPointMat);

      marker.position.copy(pos);

      group.add(marker);



      // Viewing frustum pyramid

      const frustum = new THREE.Mesh(pyramidGeo, frustumMat);

      frustum.position.copy(pos);

      if (cam.R && cam.R.length === 3) {

        const mat = new THREE.Matrix4();

        // Camera rotation from COLMAP

        mat.set(

          cam.R[0][0], cam.R[0][1], cam.R[0][2], 0,

          cam.R[1][0], cam.R[1][1], cam.R[1][2], 0,

          cam.R[2][0], cam.R[2][1], cam.R[2][2], 0,

          0, 0, 0, 1

        );

        frustum.rotation.setFromRotationMatrix(mat);

      }

      group.add(frustum);

    }

  });



  // Spline Flight Path Line

  if (pathPoints.length > 1) {

    const curve = new THREE.CatmullRomCurve3(pathPoints);

    const splinePoints = curve.getPoints(pathPoints.length * 4);

    const lineGeo = new THREE.BufferGeometry().setFromPoints(splinePoints);

    const lineMat = new THREE.LineBasicMaterial({ color: 0x4ee6d0, linewidth: 2 });

    const splineLine = new THREE.Line(lineGeo, lineMat);

    group.add(splineLine);

  }



  group.visible = false; // Hidden by default, toggleable via HUD

  scene.add(group);

  currentCamerasGroup = group;

}



// ── Build Dense Point Cloud from points.json ─────────────────────────────────

function buildPointCloud(rawPoints) {

  if (currentPointsObject) {

    scene.remove(currentPointsObject);

    currentPointsObject = null;

  }

  if (!rawPoints || !rawPoints.length) return;



  const count = rawPoints.length;

  const positions = new Float32Array(count * 3);

  const colors = new Float32Array(count * 3);



  for (let i = 0; i < count; i++) {

    const pt = rawPoints[i];

    positions[i * 3] = pt[0];

    positions[i * 3 + 1] = pt[1];

    positions[i * 3 + 2] = pt[2];



    if (pt.length >= 6) {

      colors[i * 3] = (pt[3] || 128) / 255.0;

      colors[i * 3 + 1] = (pt[4] || 128) / 255.0;

      colors[i * 3 + 2] = (pt[5] || 128) / 255.0;

    } else {

      colors[i * 3] = 0.4;

      colors[i * 3 + 1] = 0.8;

      colors[i * 3 + 2] = 0.6;

    }

  }



  const geometry = new THREE.BufferGeometry();

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));



  const material = new THREE.PointsMaterial({

    size: 0.25,

    vertexColors: true,

    sizeAttenuation: true,

  });



  const pointsMesh = new THREE.Points(geometry, material);

  pointsMesh.visible = false;

  scene.add(pointsMesh);

  currentPointsObject = pointsMesh;

}



// ── Load Job & Reconstruction Deliverables ───────────────────────────────────

async function loadJob(id, sample = null) {
  if (!id) return;
  const token = ++jobLoadToken;
  document.getElementById('btnViewBack').disabled = true;
  loadingOverlay.classList.remove('hidden');

  loadingText.textContent = `Loading reconstruction ${id.slice(0, 8)}…`;



  try {

    const [payload, metrics] = sample
      ? [sample.payload, sample.metrics]
      : await Promise.all([
          Aero.api(`/points/${id}`),
          Aero.api(`/measurements/${id}`)
        ]);



    // Update Sidebar metrics

    projectMessage.textContent = `Loaded ${id.slice(0, 8)} · verified project output`;

    statSparse.textContent = (metrics.sparse_points || 0).toLocaleString();

    statDense.textContent = (metrics.dense_points || 0).toLocaleString();

    statVertices.textContent = (metrics.mesh_vertices || 0).toLocaleString();

    statTriangles.textContent = (metrics.mesh_triangles || 0).toLocaleString();

    statReprojection.textContent = metrics.reprojection_error_px != null ? `${metrics.reprojection_error_px} px` : '—';

    statGsd.textContent = typeof metrics.gsd_cm_px === 'number' ? `${metrics.gsd_cm_px.toFixed(2)} cm/px` : 'Unavailable';

    statCrs.textContent = metrics.crs || 'Local metric';
    observedStats = {
      vertices: metrics.mesh_vertices || 0,
      triangles: metrics.mesh_triangles || 0,
      reprojection: statReprojection.textContent,
      gsd: statGsd.textContent,
    };
    estimatedStats = null;


    // 1. Build Point Cloud & Flight Path

    if (payload.points) {

      buildPointCloud(payload.points);

    }

    if (payload.trajectory) {

      buildCameraTrajectory(payload.trajectory);

    }



    // 2. Load GLB Textured Mesh

    if (currentMeshGroup) {
      scene.remove(currentMeshGroup);

      currentMeshGroup = null;

      originalMaterials.clear();

    }
    if (currentEstimatedGroup) {
      scene.remove(currentEstimatedGroup);
      currentEstimatedGroup = null;
    }
    document.getElementById('btnModeEstimated').disabled = true;
    document.getElementById('btnViewBack').disabled = true;
    estimateNotice.style.display = 'none';


    const completeUrl = payload.complete_glb_url || payload.estimated_glb_url;
    if (completeUrl) {
      new GLTFLoader().load(
        `${completeUrl}?t=${Date.now()}`,
        (estimate) => {
          if (token !== jobLoadToken) return;
          currentEstimatedGroup = estimate.scene;
          let vertices = 0;
          let triangles = 0;
          const countedPositionBuffers = new Set();
          currentEstimatedGroup.traverse((child) => {
            if (child.isMesh && child.geometry) {
              const materials = Array.isArray(child.material) ? child.material : [child.material];
              materials.forEach((material) => {
                if (!material) return;
                material.toneMapped = !material.isMeshBasicMaterial;
                if (material.map) {
                  material.map.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 8);
                  material.map.needsUpdate = true;
                }
              });
              child.castShadow = true;
              child.receiveShadow = true;
              const position = child.geometry.attributes.position;
              if (position && !countedPositionBuffers.has(position.array)) {
                countedPositionBuffers.add(position.array);
                vertices += position.count;
              }
              triangles += child.geometry.index
                ? child.geometry.index.count / 3
                : child.geometry.attributes.position.count / 3;
            }
          });
          estimatedStats = { vertices, triangles };
          scene.add(currentEstimatedGroup);
          document.getElementById('btnModeEstimated').disabled = false;
          document.getElementById('btnViewBack').disabled = false;
          if (!currentMeshGroup) {
            sceneBoundingBox.setFromObject(currentEstimatedGroup);
            const sphere = new THREE.Sphere();
            sceneBoundingBox.getBoundingSphere(sphere);
            sceneRadius = sphere.radius || 35;
            grid.position.y = sceneBoundingBox.min.y - 0.05;
            setView('3d');
            loadingOverlay.classList.add('hidden');
          }
          setDisplayMode('estimated');
        },
        undefined,
        (error) => {
          console.warn('Estimated surface unavailable:', error);
          if (!payload.glb_url) {
            loadingOverlay.classList.add('hidden');
            projectMessage.textContent = 'Complete 3D model could not be loaded.';
          }
        }
      );
    }

    if (payload.glb_url) {

      loadingText.textContent = 'Streaming photorealistic 3D mesh…';

      const gltfLoader = new GLTFLoader();

      gltfLoader.load(

        `${payload.glb_url}?t=${Date.now()}`,

        (gltf) => {
          if (token !== jobLoadToken) return;
          currentMeshGroup = gltf.scene;



          currentMeshGroup.traverse((child) => {

            if (child.isMesh) {

              originalMaterials.set(child, child.material);

              if (child.material.map) {

                if (THREE.SRGBColorSpace) {

                  child.material.map.colorSpace = THREE.SRGBColorSpace;

                } else if (THREE.sRGBEncoding) {

                  child.material.map.encoding = THREE.sRGBEncoding;

                }

              }

            }

          });



          scene.add(currentMeshGroup);



          // Compute scene bounding box and fit camera

          sceneBoundingBox.setFromObject(currentMeshGroup);

          const sphere = new THREE.Sphere();

          sceneBoundingBox.getBoundingSphere(sphere);

          sceneRadius = sphere.radius || 35;
          grid.position.y = sceneBoundingBox.min.y - 0.05;

          if (currentDisplayMode !== 'estimated') {
            setDisplayMode('mesh');
          }

          setView('3d');

          loadingOverlay.classList.add('hidden');

          Aero.toast('3D Model loaded successfully');
        },

        (progress) => {

          if (progress.total > 0) {

            const pct = Math.round((progress.loaded / progress.total) * 100);

            loadingText.textContent = `Streaming 3D mesh… ${pct}%`;

          }

        },

        (error) => {

          console.error('GLTF load error:', error);

          // Fallback to point cloud if GLB fails

          if (currentPointsObject) {

            setDisplayMode('points');

            setView('3d');

          }

          loadingOverlay.classList.add('hidden');

          Aero.toast('Using dense point cloud display mode');

        }

      );

    } else {

      if (currentPointsObject) {

        setDisplayMode('points');

        setView('3d');

      }

      if (!completeUrl) loadingOverlay.classList.add('hidden');

    }



  } catch (err) {

    console.error('Error loading reconstruction:', err);

    loadingOverlay.classList.add('hidden');

    projectMessage.textContent = err.message;

    Aero.toast(err.message, true);

  }

}



async function loadJobs() {

  if (new URLSearchParams(window.location.search).get('sample') === 'v5') {
    studioJob.innerHTML = '<option value="sample-v5">V5 restored sample</option>';
    await loadJob('sample-v5', {
      payload: { complete_glb_url: '/frontend/assets/v5_complete_2026-09-20_1811.glb' },
      metrics: { sparse_points: 18038, dense_points: 397330, crs: 'Local (non-georeferenced)' },
    });
    projectMessage.textContent = 'Restored V5 sample · inferred surface';
    return;
  }

  try {

    const r = await Aero.api('/jobs');

    const completed = r.jobs.filter(j => j.status === 'completed');

    studioJob.innerHTML = completed.length

      ? completed.map(j => `<option value="${j.job_id}">${j.filename || j.job_id.slice(0, 8)}</option>`).join('')

      : '<option value="">No completed reconstructions</option>';

    if (completed.length) loadJob(studioJob.value);

    else {

      projectMessage.textContent = 'Complete a reconstruction to inspect it here.';

      loadingOverlay.classList.add('hidden');

    }

  } catch (e) {

    projectMessage.textContent = e.message;

    Aero.toast(e.message, true);

    loadingOverlay.classList.add('hidden');

  }

}



studioJob.onchange = () => loadJob(studioJob.value);

loadJobs();
