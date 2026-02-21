'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { Slider } from '@/components/ui/slider';
import { useTheme } from '@/lib/use-theme';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

interface IsingResult {
  grid: number[][];
  energyHistory: number[];
  magnetizationHistory: number[];
  snapshots: { step: number; grid: number[][] }[];
}

function runIsing(size: number, nsteps: number, beta: number, nsnapshots: number): IsingResult {
  // Initialize random grid
  const grid: number[][] = [];
  for (let i = 0; i < size; i++) {
    const row: number[] = [];
    for (let j = 0; j < size; j++) {
      row.push(Math.random() > 0.5 ? 1 : -1);
    }
    grid.push(row);
  }

  // Calculate initial energy
  let E = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const sumNeighbors =
        grid[i][(j + 1) % size] +
        grid[i][(j - 1 + size) % size] +
        grid[(i + 1) % size][j] +
        grid[(i - 1 + size) % size][j];
      E += -grid[i][j] * sumNeighbors / 2;
    }
  }

  const energyHistory: number[] = [E];
  const magnetizationHistory: number[] = [grid.flat().reduce((s, v) => s + v, 0)];
  const snapshots: { step: number; grid: number[][] }[] = [];
  const snapshotSteps = Array.from({ length: nsnapshots }, (_, i) => Math.floor(i * nsteps / nsnapshots));

  for (let step = 0; step < nsteps; step++) {
    const i = Math.floor(Math.random() * size);
    const j = Math.floor(Math.random() * size);

    const sumNeighbors =
      grid[i][(j + 1) % size] +
      grid[i][(j - 1 + size) % size] +
      grid[(i + 1) % size][j] +
      grid[(i - 1 + size) % size][j];

    const dE = 2 * grid[i][j] * sumNeighbors;

    if (Math.random() < Math.exp(-beta * dE)) {
      grid[i][j] *= -1;
      E += dE;
    }

    energyHistory.push(E);
    magnetizationHistory.push(grid.flat().reduce((s, v) => s + v, 0));

    if (snapshotSteps.includes(step)) {
      snapshots.push({ step, grid: grid.map(r => [...r]) });
    }
  }

  // Push final snapshot
  snapshots.push({ step: nsteps, grid: grid.map(r => [...r]) });

  return { grid: grid.map(r => [...r]), energyHistory, magnetizationHistory, snapshots };
}

// ---------- R3F instanced spin grid ----------

const colorUp = new THREE.Color('#ff4444');
const colorDown = new THREE.Color('#4444ff');
const dummy = new THREE.Object3D();

function SpinGrid({ grid }: { grid: number[][] }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const size = grid.length;
  const count = size * size;

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const colorArray = new Float32Array(count * 3);
    const halfSize = size / 2;

    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const idx = i * size + j;
        const spin = grid[i][j];
        const yPos = spin === 1 ? 0.3 : 0.1;

        dummy.position.set(i - halfSize, yPos, j - halfSize);
        dummy.updateMatrix();
        mesh.setMatrixAt(idx, dummy.matrix);

        const color = spin === 1 ? colorUp : colorDown;
        colorArray[idx * 3] = color.r;
        colorArray[idx * 3 + 1] = color.g;
        colorArray[idx * 3 + 2] = color.b;
      }
    }

    mesh.instanceMatrix.needsUpdate = true;
    mesh.geometry.setAttribute(
      'color',
      new THREE.InstancedBufferAttribute(colorArray, 3)
    );

    mesh.computeBoundingSphere();
  }, [grid, size, count]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <boxGeometry args={[0.9, 0.5, 0.9]} />
      <meshStandardMaterial
        vertexColors
        toneMapped={false}
        emissive="#ffffff"
        emissiveIntensity={0.15}
      />
    </instancedMesh>
  );
}

// ---------- Main component ----------

export function IsingModel() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [size, setSize] = useState(30);
  const [beta, setBeta] = useState(0.44);
  const [nsteps, setNsteps] = useState(5000);
  const [seed, setSeed] = useState(0);
  const [selectedSnapshot, setSelectedSnapshot] = useState(-1);

  const result = useMemo(() => {
    // seed is used only as a trigger for re-run
    return runIsing(size, nsteps, beta, 4);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, nsteps, beta, seed]);

  // Default to the last snapshot
  const snapshotIndex =
    selectedSnapshot < 0 || selectedSnapshot >= result.snapshots.length
      ? result.snapshots.length - 1
      : selectedSnapshot;

  // Reset selected snapshot when simulation parameters change
  useEffect(() => {
    setSelectedSnapshot(-1);
  }, [size, nsteps, beta, seed]);

  const activeGrid = result.snapshots[snapshotIndex].grid;
  const cameraDistance = size * 0.9;

  return (
    <div className="space-y-6">
      {/* Sliders */}
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Size: {size}</label>
          <Slider
            min={5}
            max={80}
            step={5}
            value={[size]}
            onValueChange={([v]) => setSize(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Beta (1/T): {beta.toFixed(2)}</label>
          <Slider
            min={0.01}
            max={2.0}
            step={0.01}
            value={[beta]}
            onValueChange={([v]) => setBeta(v)}
            className="w-48"
          />
        </div>
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Steps: {nsteps}</label>
          <Slider
            min={100}
            max={50000}
            step={100}
            value={[nsteps]}
            onValueChange={([v]) => setNsteps(v)}
            className="w-48"
          />
        </div>
        <button
          onClick={() => setSeed(s => s + 1)}
          className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4"
        >
          Re-run
        </button>
      </div>

      {/* 3D Spin Grid */}
      <SimulationMain className="w-full rounded-lg overflow-hidden" style={{ height: 400, background: isDark ? '#0a0a0f' : '#f0f4ff' }}>
        <Canvas
          camera={{
            position: [cameraDistance * 0.6, cameraDistance * 0.7, cameraDistance * 0.6],
            fov: 50,
            near: 0.1,
            far: cameraDistance * 5,
          }}
          gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.2 }}
        >
          <color attach="background" args={[isDark ? '#0a0a0f' : '#f0f4ff']} />
          <ambientLight intensity={isDark ? 0.3 : 0.6} />
          <directionalLight position={[10, 20, 10]} intensity={isDark ? 1.0 : 1.4} />
          <directionalLight position={[-10, 15, -10]} intensity={isDark ? 0.4 : 0.6} color={isDark ? '#ffffff' : '#aabbee'} />
          <SpinGrid grid={activeGrid} />
          {isDark && (
            <EffectComposer>
              <Bloom
                intensity={0.5}
                luminanceThreshold={0.6}
                luminanceSmoothing={0.9}
                mipmapBlur
              />
            </EffectComposer>
          )}
          <OrbitControls
            enablePan
            enableZoom
            enableRotate
            minDistance={cameraDistance * 0.3}
            maxDistance={cameraDistance * 3}
          />
        </Canvas>
      </SimulationMain>

      {/* Snapshot selector */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-sm text-[var(--text-muted)] mr-1">Snapshot:</span>
        {result.snapshots.map((snap, idx) => {
          const isActive = idx === snapshotIndex;
          return (
            <button
              key={idx}
              onClick={() => setSelectedSnapshot(idx)}
              className={`
                px-3 py-1.5 rounded text-sm font-medium transition-colors
                ${isActive
                  ? 'bg-[var(--accent-strong)] text-white'
                  : 'bg-[var(--surface-1)] text-[var(--text-muted)] hover:bg-[var(--border-strong)] hover:text-[var(--text-strong)]'
                }
              `}
            >
              {idx + 1}
              <span className="ml-1 text-xs opacity-70">
                (t={snap.step})
              </span>
            </button>
          );
        })}
      </div>

      {/* Energy and magnetization time series */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CanvasChart
          data={[
            {
              y: result.energyHistory,
              type: 'scatter',
              mode: 'lines',
              line: { color: '#ef4444', width: 1 },
              name: 'Energy',
            },
          ]}
          layout={{
            title: { text: 'Energy vs Time', font: { size: 13 } },
            xaxis: { title: { text: 'Timestep' } },
            yaxis: { title: { text: 'Energy' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          }}
          style={{ width: '100%', height: 300 }}
        />
        <CanvasChart
          data={[
            {
              y: result.magnetizationHistory.map(Math.abs),
              type: 'scatter',
              mode: 'lines',
              line: { color: '#f59e0b', width: 1 },
              name: '|Magnetization|',
            },
          ]}
          layout={{
            title: { text: '|Magnetization| vs Time', font: { size: 13 } },
            xaxis: { title: { text: 'Timestep' } },
            yaxis: { title: { text: '|M|' } },
            margin: { t: 40, r: 20, b: 50, l: 60 },
            showlegend: false,
          }}
          style={{ width: '100%', height: 300 }}
        />
      </div>
    </div>
  );
}
