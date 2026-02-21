'use client';

import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { Slider } from '@/components/ui/slider';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { useTheme, type Theme } from '@/lib/use-theme';

/* ------------------------------------------------------------------ */
/*  Deterministic RNG                                                  */
/* ------------------------------------------------------------------ */

function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------------ */
/*  Topple logic                                                       */
/* ------------------------------------------------------------------ */

function topple(grid: number[][], i: number, j: number, size: number): number {
  let count = 0;
  const stack: [number, number][] = [[i, j]];
  while (stack.length > 0) {
    const [x, y] = stack.pop()!;
    while (grid[x][y] >= 4) {
      grid[x][y] -= 4;
      count++;
      if (x > 0) {
        grid[x - 1][y] += 1;
        if (grid[x - 1][y] >= 4) stack.push([x - 1, y]);
      }
      if (x < size - 1) {
        grid[x + 1][y] += 1;
        if (grid[x + 1][y] >= 4) stack.push([x + 1, y]);
      }
      if (y > 0) {
        grid[x][y - 1] += 1;
        if (grid[x][y - 1] >= 4) stack.push([x, y - 1]);
      }
      if (y < size - 1) {
        grid[x][y + 1] += 1;
        if (grid[x][y + 1] >= 4) stack.push([x, y + 1]);
      }
    }
  }
  return count;
}

/* ------------------------------------------------------------------ */
/*  Turbo-like color palette                                           */
/* ------------------------------------------------------------------ */

function turboColor(t: number): [number, number, number] {
  const r = Math.max(0, Math.min(1,
    0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))));
  const g = Math.max(0, Math.min(1,
    0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * 7.56348409)))));
  const b = Math.max(0, Math.min(1,
    0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))));
  return [r, g, b];
}

/* ------------------------------------------------------------------ */
/*  Simulation state                                                   */
/* ------------------------------------------------------------------ */

interface SimState {
  grid: number[][];
  rng: () => number;
  step: number;
  avalanches: number[];
  avalancheCount: number;
}

function createSimState(size: number, seed: number): SimState {
  return {
    grid: Array.from({ length: size }, () => Array(size).fill(0)),
    rng: mulberry32((size * 97 + seed * 131) >>> 0),
    step: 0,
    avalanches: [],
    avalancheCount: 0,
  };
}

/* ------------------------------------------------------------------ */
/*  Animated 3D Instanced Grid (renders inside Canvas)                 */
/* ------------------------------------------------------------------ */

function AnimatedSandpileGrid({
  simRef,
  size,
  speed,
  playing,
  onStep,
}: {
  simRef: React.MutableRefObject<SimState>;
  size: number;
  speed: number;
  playing: boolean;
  onStep: (step: number, avalancheCount: number) => void;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const colorObj = useMemo(() => new THREE.Color(), []);
  const frameCount = useRef(0);

  // Use refs for values read inside useFrame to avoid stale closures
  const playingRef = useRef(playing);
  const speedRef = useRef(speed);
  useEffect(() => {
    playingRef.current = playing;
    speedRef.current = speed;
  }, [playing, speed]);

  const totalCells = size * size;
  const heightScale = 0.6;
  const baseHeight = 0.08;
  const cellSize = 0.9;

  const updateMesh = useCallback(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const sim = simRef.current;
    const offset = (size - 1) / 2;

    let idx = 0;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const val = sim.grid[i][j];
        const h = baseHeight + val * heightScale;

        dummy.position.set(
          (j - offset) * cellSize,
          h / 2,
          (i - offset) * cellSize,
        );
        dummy.scale.set(cellSize * 0.92, h, cellSize * 0.92);
        dummy.updateMatrix();
        mesh.setMatrixAt(idx, dummy.matrix);

        const t = Math.min(val / 3, 1);
        const [r, g, b] = turboColor(t);
        colorObj.setRGB(r, g, b);
        mesh.setColorAt(idx, colorObj);

        idx++;
      }
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [simRef, size, dummy, colorObj, baseHeight, heightScale, cellSize]);

  useFrame(() => {
    if (!playingRef.current) return;
    const sim = simRef.current;
    const spd = speedRef.current;

    for (let s = 0; s < spd; s++) {
      const i = Math.floor(sim.rng() * size);
      const j = Math.floor(sim.rng() * size);
      sim.grid[i][j] += 1;

      if (sim.grid[i][j] >= 4) {
        const a = topple(sim.grid, i, j, size);
        if (a > 0) {
          sim.avalanches.push(a);
          sim.avalancheCount++;
        }
      }
      sim.step++;
    }

    updateMesh();

    // Throttle React state updates to every 3 frames
    frameCount.current++;
    if (frameCount.current % 3 === 0) {
      onStep(sim.step, sim.avalancheCount);
    }
  });

  // Initial render of empty grid
  useEffect(() => {
    updateMesh();
  }, [updateMesh]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, totalCells]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        roughness={0.45}
        metalness={0.15}
        vertexColors
        toneMapped={false}
      />
    </instancedMesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Dark ground plane beneath the grid                                 */
/* ------------------------------------------------------------------ */

function GroundPlane({ size, theme }: { size: number; theme: Theme }) {
  const extent = size * 0.95;
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
      <planeGeometry args={[extent, extent]} />
      <meshStandardMaterial
        color={theme === 'light' ? '#dfe8fb' : '#111118'}
        roughness={0.9}
        metalness={0.0}
      />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Main exported component                                            */
/* ------------------------------------------------------------------ */

export function SandpileModel() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [size, setSize] = useState(35);
  const [speed, setSpeed] = useState(5);
  const [playing, setPlaying] = useState(true);
  const [stepCount, setStepCount] = useState(0);
  const [avalancheCount, setAvalancheCount] = useState(0);
  const [rerun, setRerun] = useState(0);
  const simRef = useRef<SimState>(createSimState(35, 0));

  // Reset simulation when size or rerun changes
  useEffect(() => {
    simRef.current = createSimState(size, rerun);
    setStepCount(0);
    setAvalancheCount(0);
  }, [size, rerun]);

  const handleStep = useCallback((step: number, avCount: number) => {
    setStepCount(step);
    setAvalancheCount(avCount);
  }, []);

  // Snapshot avalanche data for chart — recompute every 200 steps or when paused
  const chartEpoch = Math.floor(stepCount / 200);
  const chartAvalanches = useMemo(() => {
    return simRef.current.avalanches.filter(v => v > 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartEpoch, playing]);

  const camDist = size * 0.7;

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Grid Size: {size}</label>
          <Slider value={[size]} onValueChange={([v]) => setSize(v)} min={15} max={80} step={5} className="w-48" />
        </div>
        <div>
          <label className="mb-1 block text-sm text-[var(--text-muted)]">Speed: {speed} grains/frame</label>
          <Slider value={[speed]} onValueChange={([v]) => setSpeed(v)} min={1} max={50} step={1} className="w-48" />
        </div>
        <button
          onClick={() => setPlaying(p => !p)}
          className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => setRerun(v => v + 1)}
          className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4"
        >
          Reset
        </button>
      </div>

      <div className="text-sm text-[var(--text-muted)]">
        Step: {stepCount.toLocaleString()} | Avalanches: {avalancheCount.toLocaleString()}
      </div>

      {/* 3D Height-Field Visualization */}
      <div
        className="w-full rounded-lg overflow-hidden"
        style={{ height: 420, background: isDark ? '#08080c' : '#f0f4ff' }}
      >
        <Canvas
          key={`${size}-${rerun}-${theme}`}
          camera={{
            position: [camDist * 0.7, camDist * 0.55, camDist * 0.7],
            fov: 50,
            near: 0.1,
            far: camDist * 5,
          }}
          gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: isDark ? 1.2 : 1.0 }}
        >
          <color attach="background" args={[isDark ? '#08080c' : '#f0f4ff']} />
          <ambientLight intensity={isDark ? 0.35 : 0.6} />
          <directionalLight
            position={[camDist * 0.8, camDist * 1.2, camDist * 0.4]}
            intensity={isDark ? 1.1 : 1.4}
          />
          <directionalLight
            position={[-camDist * 0.5, camDist * 0.6, -camDist * 0.7]}
            intensity={isDark ? 0.4 : 0.6}
            color={isDark ? '#6688cc' : '#aabbee'}
          />

          <GroundPlane size={size} theme={theme} />
          <AnimatedSandpileGrid
            simRef={simRef}
            size={size}
            speed={speed}
            playing={playing}
            onStep={handleStep}
          />

          <OrbitControls
            target={[0, 0.5, 0]}
            enablePan
            enableZoom
            minDistance={size * 0.3}
            maxDistance={size * 2.5}
          />

          {isDark && (
            <EffectComposer>
              <Bloom
                intensity={0.3}
                luminanceThreshold={0.6}
                luminanceSmoothing={0.4}
                mipmapBlur
              />
            </EffectComposer>
          )}
        </Canvas>
      </div>

      {/* Avalanche Histogram */}
      {chartAvalanches.length > 0 && (
        <div className="w-full">
          <CanvasChart
            data={[
              {
                x: chartAvalanches,
                type: 'histogram',
                marker: { color: '#f59e0b' },
                nbinsx: 50,
              },
            ]}
            layout={{
              title: { text: 'Avalanche Size Distribution' },
              xaxis: { title: { text: 'Avalanche size' }, type: 'log' },
              yaxis: { title: { text: 'Count' }, type: 'log' },
              margin: { t: 40, r: 20, b: 50, l: 60 },
            }}
            style={{ width: '100%', height: 360 }}
          />
        </div>
      )}
    </div>
  );
}
