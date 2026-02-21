'use client';

import { useRef, useState, useMemo, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { Slider } from '@/components/ui/slider';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { SimulationMain } from '@/components/ui/simulation-main';
import { useTheme } from '@/lib/use-theme';

// ---------------------------------------------------------------------------
// Lorenz ODE integrator
// ---------------------------------------------------------------------------

interface LorenzTrajectory {
  points: THREE.Vector3[];
  velocities: number[];
}

function computeLorenz(
  sigma: number,
  rho: number,
  beta: number,
  x0 = 1,
  y0 = 1,
  z0 = 1,
): LorenzTrajectory {
  const dt = 0.005;
  const steps = 16000;
  let x = x0;
  let y = y0;
  let z = z0;

  const points: THREE.Vector3[] = [];
  const velocities: number[] = [];

  for (let i = 0; i < steps; i++) {
    const dx = sigma * (y - x);
    const dy = x * (rho - z) - y;
    const dz = x * y - beta * z;

    x += dx * dt;
    y += dy * dt;
    z += dz * dt;

    // Swap y/z so the attractor wings sit upright
    points.push(new THREE.Vector3(x, z, y));
    velocities.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }

  return { points, velocities };
}

// ---------------------------------------------------------------------------
// Velocity-to-colour mapping  (blue -> cyan -> green -> yellow -> red)
// ---------------------------------------------------------------------------

const COLOR_STOPS: [number, THREE.Color][] = [
  [0.0, new THREE.Color(0x0033ff)],
  [0.25, new THREE.Color(0x00ccff)],
  [0.5, new THREE.Color(0x00ff66)],
  [0.75, new THREE.Color(0xffcc00)],
  [1.0, new THREE.Color(0xff2200)],
];

function velocityToColor(t: number): THREE.Color {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const [t0, c0] = COLOR_STOPS[i];
    const [t1, c1] = COLOR_STOPS[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const f = (clamped - t0) / (t1 - t0);
      return c0.clone().lerp(c1, f);
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1][1].clone();
}

// ---------------------------------------------------------------------------
// Build a tube mesh with per-vertex colours driven by velocity
// ---------------------------------------------------------------------------

function buildTubeGeometry(
  trajectory: LorenzTrajectory,
  tubeRadius: number,
  radialSegments: number,
): THREE.BufferGeometry {
  const { points, velocities } = trajectory;

  // Subsample every Nth point for performance
  const step = 4;
  const sampledPoints: THREE.Vector3[] = [];
  const sampledVelocities: number[] = [];
  for (let i = 0; i < points.length; i += step) {
    sampledPoints.push(points[i]);
    sampledVelocities.push(velocities[i]);
  }

  if (sampledPoints.length < 4) {
    return new THREE.BufferGeometry();
  }

  const curve = new THREE.CatmullRomCurve3(sampledPoints, false, 'catmullrom', 0.5);
  const tubeGeo = new THREE.TubeGeometry(
    curve,
    sampledPoints.length - 1,
    tubeRadius,
    radialSegments,
    false,
  );

  // Normalise velocities for colour mapping
  let minV = Infinity;
  let maxV = -Infinity;
  for (const v of sampledVelocities) {
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  const range = maxV - minV || 1;

  // Paint vertices -- each ring of (radialSegments+1) verts maps to one curve sample
  const posAttr = tubeGeo.attributes.position;
  const vertCount = posAttr.count;
  const colors = new Float32Array(vertCount * 3);
  const ringSize = radialSegments + 1;
  const tmpColor = new THREE.Color();

  for (let i = 0; i < vertCount; i++) {
    const segmentIdx = Math.floor(i / ringSize);
    const velIdx = Math.min(segmentIdx, sampledVelocities.length - 1);
    const t = (sampledVelocities[velIdx] - minV) / range;
    tmpColor.copy(velocityToColor(t));
    colors[i * 3] = tmpColor.r;
    colors[i * 3 + 1] = tmpColor.g;
    colors[i * 3 + 2] = tmpColor.b;
  }

  tubeGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return tubeGeo;
}

// ---------------------------------------------------------------------------
// Initial conditions for multiple trails
// ---------------------------------------------------------------------------

const INITIAL_CONDITIONS: [number, number, number][] = [
  [1, 1, 1],
  [1.01, 1, 1],
  [0.99, 1.01, 0.99],
];

// ---------------------------------------------------------------------------
// LorenzTrail -- a single tube trail coloured by velocity
// ---------------------------------------------------------------------------

interface LorenzTrailProps {
  sigma: number;
  rho: number;
  beta: number;
  initial: [number, number, number];
  tubeRadius?: number;
}

function LorenzTrail({ sigma, rho, beta, initial, tubeRadius = 0.12 }: LorenzTrailProps) {
  const meshRef = useRef<THREE.Mesh>(null!);

  const geometry = useMemo(() => {
    const traj = computeLorenz(sigma, rho, beta, initial[0], initial[1], initial[2]);
    return buildTubeGeometry(traj, tubeRadius, 6);
  }, [sigma, rho, beta, initial, tubeRadius]);

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        vertexColors
        roughness={0.35}
        metalness={0.15}
        emissive={new THREE.Color(0xffffff)}
        emissiveIntensity={0.45}
        toneMapped={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Faint axis lines for spatial reference
// ---------------------------------------------------------------------------

function AxisLines({ isDark }: { isDark: boolean }) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array([
      -60, 0, 0, 60, 0, 0,
      0, -10, 0, 0, 80, 0,
      0, 0, -60, 0, 0, 60,
    ]);
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    return geo;
  }, []);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={isDark ? '#333344' : '#b0bdd4'} transparent opacity={isDark ? 0.3 : 0.5} />
    </lineSegments>
  );
}

// ---------------------------------------------------------------------------
// Scene wrapper (all 3D objects that use R3F hooks)
// ---------------------------------------------------------------------------

interface SceneProps {
  sigma: number;
  rho: number;
  beta: number;
  trails: number;
  autoRotate: boolean;
  isDark: boolean;
}

function Scene({ sigma, rho, beta, trails, autoRotate, isDark }: SceneProps) {
  const controlsRef = useRef<React.ComponentRef<typeof OrbitControls>>(null!);

  // Scale the attractor so it fits nicely in the viewport
  const scale = 0.04;

  const trailElements = useMemo(() => {
    const elements: React.JSX.Element[] = [];
    for (let i = 0; i < trails; i++) {
      const ic = INITIAL_CONDITIONS[i] ?? INITIAL_CONDITIONS[0];
      elements.push(
        <LorenzTrail
          key={`trail-${i}-${sigma}-${rho}-${beta}`}
          sigma={sigma}
          rho={rho}
          beta={beta}
          initial={ic}
          tubeRadius={trails === 1 ? 0.15 : 0.10}
        />,
      );
    }
    return elements;
  }, [sigma, rho, beta, trails]);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={isDark ? 0.15 : 0.4} color="#aaccff" />
      <directionalLight position={[10, 20, 10]} intensity={isDark ? 0.6 : 1.0} color="#ffffff" />
      <directionalLight position={[-10, -5, -10]} intensity={isDark ? 0.2 : 0.4} color={isDark ? '#6688cc' : '#aabbee'} />
      <pointLight position={[0, 2, 0]} intensity={isDark ? 0.5 : 0.3} color="#ff8844" distance={8} />

      {/* Attractor group */}
      <group scale={[scale, scale, scale]}>
        {trailElements}
        <AxisLines isDark={isDark} />
      </group>

      {/* Post-processing */}
      {isDark && (
        <EffectComposer>
          <Bloom
            luminanceThreshold={0.2}
            luminanceSmoothing={0.9}
            intensity={1.4}
            mipmapBlur
          />
        </EffectComposer>
      )}

      {/* Controls */}
      <OrbitControls
        ref={controlsRef}
        autoRotate={autoRotate}
        autoRotateSpeed={0.6}
        enableDamping
        dampingFactor={0.08}
        minDistance={1}
        maxDistance={12}
        makeDefault
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// Main exported component
// ---------------------------------------------------------------------------

export function LorenzAttractor() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [sigma, setSigma] = useState([10]);
  const [rho, setRho] = useState([28]);
  const [beta, setBeta] = useState([2.67]);
  const [trails, setTrails] = useState([1]);
  const [autoRotate, setAutoRotate] = useState(true);

  const toggleAutoRotate = useCallback(() => setAutoRotate((v) => !v), []);

  return (
    <div className="flex flex-col lg:flex-row gap-4">
      {/* 3D Canvas */}
      <SimulationMain
        className="flex-1 w-full rounded-lg overflow-hidden"
        style={{ minHeight: 500, background: isDark ? '#050510' : '#f0f4ff' }}
        scaleMode="fill"
      >
        <Canvas
          gl={{
            antialias: true,
            toneMapping: THREE.ACESFilmicToneMapping,
            toneMappingExposure: 1.2,
          }}
          camera={{ position: [3.5, 2.5, 3.5], fov: 50, near: 0.1, far: 100 }}
          dpr={[1, 2]}
          style={{ width: '100%', height: '100%' }}
        >
          <color attach="background" args={[isDark ? '#050510' : '#f0f4ff']} />
          <Scene
            sigma={sigma[0]}
            rho={rho[0]}
            beta={beta[0]}
            trails={trails[0]}
            autoRotate={autoRotate}
            isDark={isDark}
          />
        </Canvas>
      </SimulationMain>

      {/* Parameters panel */}
      <Card className="w-full lg:w-80 shrink-0">
        <CardHeader>
          <CardTitle>Parameters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Sigma */}
          <div>
            <label className="text-sm font-medium text-[var(--text-strong)]">
              Sigma ({'\u03C3'}): {sigma[0].toFixed(1)}
            </label>
            <Slider value={sigma} onValueChange={setSigma} min={0} max={20} step={0.1} />
          </div>

          {/* Rho */}
          <div>
            <label className="text-sm font-medium text-[var(--text-strong)]">
              Rho ({'\u03C1'}): {rho[0].toFixed(1)}
            </label>
            <Slider value={rho} onValueChange={setRho} min={0} max={50} step={0.1} />
          </div>

          {/* Beta */}
          <div>
            <label className="text-sm font-medium text-[var(--text-strong)]">
              Beta ({'\u03B2'}): {beta[0].toFixed(2)}
            </label>
            <Slider value={beta} onValueChange={setBeta} min={0} max={5} step={0.01} />
          </div>

          {/* Trails */}
          <div>
            <label className="text-sm font-medium text-[var(--text-strong)]">
              Trails: {trails[0]}
            </label>
            <Slider value={trails} onValueChange={setTrails} min={1} max={3} step={1} />
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Multiple trails show sensitivity to initial conditions
            </p>
          </div>

          {/* Auto-rotate toggle */}
          <button
            onClick={toggleAutoRotate}
            className={`w-full px-4 py-2 rounded text-sm font-medium transition-colors ${
              autoRotate
                ? 'bg-[var(--accent)] text-white hover:bg-[var(--accent-strong)]'
                : 'bg-[var(--surface-2)] text-[var(--text-strong)] hover:bg-[var(--surface-3)]'
            }`}
          >
            Auto-rotate: {autoRotate ? 'ON' : 'OFF'}
          </button>

          {/* Presets */}
          <div className="space-y-2">
            <p className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wide">
              Presets
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => {
                  setSigma([10]);
                  setRho([28]);
                  setBeta([2.67]);
                }}
                className="px-3 py-1.5 text-xs rounded bg-[var(--surface-2)] text-[var(--text-strong)] hover:bg-[var(--surface-3)] transition-colors"
              >
                Classic
              </button>
              <button
                onClick={() => {
                  setSigma([10]);
                  setRho([15]);
                  setBeta([2.67]);
                }}
                className="px-3 py-1.5 text-xs rounded bg-[var(--surface-2)] text-[var(--text-strong)] hover:bg-[var(--surface-3)] transition-colors"
              >
                Transient
              </button>
              <button
                onClick={() => {
                  setSigma([14]);
                  setRho([28]);
                  setBeta([4]);
                }}
                className="px-3 py-1.5 text-xs rounded bg-[var(--surface-2)] text-[var(--text-strong)] hover:bg-[var(--surface-3)] transition-colors"
              >
                Tight Wings
              </button>
              <button
                onClick={() => {
                  setSigma([10]);
                  setRho([45]);
                  setBeta([2.67]);
                }}
                className="px-3 py-1.5 text-xs rounded bg-[var(--surface-2)] text-[var(--text-strong)] hover:bg-[var(--surface-3)] transition-colors"
              >
                High Rho
              </button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
