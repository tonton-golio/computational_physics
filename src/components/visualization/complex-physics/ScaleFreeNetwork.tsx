'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { CanvasChart } from '@/components/ui/canvas-chart';
import { SimulationMain } from '@/components/ui/simulation-main';
import { Slider } from '@/components/ui/slider';
import { useTheme } from '@/lib/use-theme';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

interface NetworkNode {
  id: number;
  degree: number;
  group: number;
  x: number;
  y: number;
  z: number;
}

interface NetworkLink {
  source: number;
  target: number;
}

function generateScaleFreeNetwork(n: number): { nodes: NetworkNode[]; links: NetworkLink[] } {
  const nodeList: NetworkNode[] = [];
  const linkList: NetworkLink[] = [];

  // Start with a small complete graph (triangle)
  for (let i = 0; i < Math.min(3, n); i++) {
    nodeList.push({
      id: i,
      degree: Math.min(3, n) - 1,
      group: 1,
      x: Math.cos(2 * Math.PI * i / Math.min(3, n)),
      y: Math.sin(2 * Math.PI * i / Math.min(3, n)),
      z: 0,
    });
  }

  if (n >= 3) {
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) {
        linkList.push({ source: i, target: j });
      }
    }
  }

  // Add remaining nodes with preferential attachment (Barabasi-Albert)
  for (let i = 3; i < n; i++) {
    const degrees = nodeList.map(node => node.degree);
    const totalDegree = degrees.reduce((sum, d) => sum + d, 0);
    const probabilities = degrees.map(d => d / totalDegree);

    const m = 2;
    const targets = new Set<number>();

    for (let k = 0; k < m && targets.size < nodeList.length; k++) {
      const rand = Math.random();
      let cumulative = 0;
      for (let j = 0; j < nodeList.length; j++) {
        cumulative += probabilities[j];
        if (rand <= cumulative && !targets.has(j)) {
          targets.add(j);
          linkList.push({ source: i, target: j });
          nodeList[j].degree++;
          break;
        }
      }
    }

    nodeList.push({
      id: i,
      degree: targets.size,
      group: Math.floor(Math.random() * 3) + 1,
      x: (Math.random() - 0.5) * 2,
      y: (Math.random() - 0.5) * 2,
      z: (Math.random() - 0.5) * 2,
    });
  }

  return { nodes: nodeList, links: linkList };
}

// 3D force-directed layout (Fruchterman-Reingold extended to 3D)
function layoutNetwork(nodes: NetworkNode[], links: NetworkLink[], iterations: number = 50): NetworkNode[] {
  const positioned = nodes.map(n => ({ ...n }));
  const k = Math.sqrt(4 / Math.max(1, nodes.length));

  for (let iter = 0; iter < iterations; iter++) {
    const temp = 1 - iter / iterations;
    const dx = new Array(positioned.length).fill(0);
    const dy = new Array(positioned.length).fill(0);
    const dz = new Array(positioned.length).fill(0);

    // Repulsion
    for (let i = 0; i < positioned.length; i++) {
      for (let j = i + 1; j < positioned.length; j++) {
        const ddx = positioned[i].x - positioned[j].x;
        const ddy = positioned[i].y - positioned[j].y;
        const ddz = positioned[i].z - positioned[j].z;
        const dist = Math.max(0.01, Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz));
        const force = (k * k) / dist;
        dx[i] += (ddx / dist) * force;
        dy[i] += (ddy / dist) * force;
        dz[i] += (ddz / dist) * force;
        dx[j] -= (ddx / dist) * force;
        dy[j] -= (ddy / dist) * force;
        dz[j] -= (ddz / dist) * force;
      }
    }

    // Attraction along edges
    for (const link of links) {
      const si = link.source;
      const ti = link.target;
      if (si >= positioned.length || ti >= positioned.length) continue;
      const ddx = positioned[si].x - positioned[ti].x;
      const ddy = positioned[si].y - positioned[ti].y;
      const ddz = positioned[si].z - positioned[ti].z;
      const dist = Math.max(0.01, Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz));
      const force = (dist * dist) / k;
      dx[si] -= (ddx / dist) * force;
      dy[si] -= (ddy / dist) * force;
      dz[si] -= (ddz / dist) * force;
      dx[ti] += (ddx / dist) * force;
      dy[ti] += (ddy / dist) * force;
      dz[ti] += (ddz / dist) * force;
    }

    // Apply with temperature
    for (let i = 0; i < positioned.length; i++) {
      const disp = Math.max(0.01, Math.sqrt(dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i]));
      const maxDisp = temp * 0.1;
      positioned[i].x += (dx[i] / disp) * Math.min(disp, maxDisp);
      positioned[i].y += (dy[i] / disp) * Math.min(disp, maxDisp);
      positioned[i].z += (dz[i] / disp) * Math.min(disp, maxDisp);
    }
  }

  return positioned;
}

const GROUP_COLORS = ['#3b82f6', '#ec4899', '#10b981'];
function NetworkNodes({ nodes }: { nodes: NetworkNode[] }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tempObject = useMemo(() => new THREE.Object3D(), []);
  const tempColor = useMemo(() => new THREE.Color(), []);

  const maxDegree = useMemo(() => Math.max(1, ...nodes.map(n => n.degree)), [nodes]);

  useEffect(() => {
    if (!meshRef.current) return;

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const scale = 0.03 + (node.degree / maxDegree) * 0.12;
      tempObject.position.set(node.x, node.y, node.z);
      tempObject.scale.set(scale, scale, scale);
      tempObject.updateMatrix();
      meshRef.current.setMatrixAt(i, tempObject.matrix);

      const colorIndex = (node.group - 1) % GROUP_COLORS.length;
      tempColor.set(GROUP_COLORS[colorIndex]);
      meshRef.current.setColorAt(i, tempColor);
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [nodes, maxDegree, tempObject, tempColor]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, nodes.length]}>
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial
        emissive="#ffffff"
        emissiveIntensity={0.8}
        toneMapped={false}
        vertexColors
      />
    </instancedMesh>
  );
}

function NetworkEdges({ nodes, links, isDark }: { nodes: NetworkNode[]; links: NetworkLink[]; isDark: boolean }) {
  const geometryRef = useRef<THREE.BufferGeometry>(null);

  const positions = useMemo(() => {
    const arr: number[] = [];
    for (const link of links) {
      const s = nodes[link.source];
      const t = nodes[link.target];
      if (s && t) {
        arr.push(s.x, s.y, s.z);
        arr.push(t.x, t.y, t.z);
      }
    }
    return new Float32Array(arr);
  }, [nodes, links]);

  useEffect(() => {
    if (geometryRef.current) {
      geometryRef.current.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometryRef.current.computeBoundingSphere();
    }
  }, [positions]);

  return (
    <lineSegments>
      <bufferGeometry ref={geometryRef} />
      <lineBasicMaterial color={isDark ? '#334455' : '#8a9bbf'} transparent opacity={isDark ? 0.35 : 0.5} />
    </lineSegments>
  );
}

function NetworkScene({ nodes, links, isDark }: { nodes: NetworkNode[]; links: NetworkLink[]; isDark: boolean }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.05;
    }
  });

  return (
    <>
      <color attach="background" args={[isDark ? '#0a0a0f' : '#f0f4ff']} />

      <ambientLight intensity={isDark ? 0.25 : 0.6} />
      <directionalLight position={[10, 20, 10]} intensity={isDark ? 0.9 : 1.3} />
      <directionalLight position={[-10, 15, -10]} intensity={isDark ? 0.35 : 0.5} color={isDark ? '#6688cc' : '#aabbee'} />

      <group ref={groupRef}>
        <NetworkEdges nodes={nodes} links={links} isDark={isDark} />
        <NetworkNodes nodes={nodes} />
      </group>

      <OrbitControls
        enableDamping
        dampingFactor={0.12}
        minDistance={0.5}
        maxDistance={5}
      />

      {isDark && (
        <EffectComposer>
          <Bloom
            intensity={0.6}
            luminanceThreshold={0.4}
            luminanceSmoothing={0.9}
            mipmapBlur
          />
        </EffectComposer>
      )}
    </>
  );
}

export function ScaleFreeNetwork() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [numNodes, setNumNodes] = useState(50);
  const [seed, setSeed] = useState(0);

  const network = useMemo(() => {
    const raw = generateScaleFreeNetwork(numNodes);
    const positioned = layoutNetwork(raw.nodes, raw.links);
    return { nodes: positioned, links: raw.links };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numNodes, seed]);

  const degrees = useMemo(() => network.nodes.map(n => n.degree), [network]);

  const cameraDistance = useMemo(() => {
    let maxR = 0;
    for (const node of network.nodes) {
      const r = Math.sqrt(node.x * node.x + node.y * node.y + node.z * node.z);
      if (r > maxR) maxR = r;
    }
    return Math.max(1.5, maxR * 2.5);
  }, [network]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap gap-6 items-center">
        <div>
          <label className="text-sm text-[var(--text-muted)] block mb-1">Nodes: {numNodes}</label>
          <Slider
            min={10}
            max={100}
            step={5}
            value={[numNodes]}
            onValueChange={([v]) => setNumNodes(v)}
            className="w-48"
          />
        </div>
        <button
          onClick={() => setSeed(s => s + 1)}
          className="px-4 py-2 bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white rounded text-sm mt-4"
        >
          Regenerate
        </button>
      </div>

      {/* 3D Network Graph */}
      <SimulationMain
        className="w-full rounded-lg overflow-hidden"
        style={{ height: 500, background: isDark ? '#0a0a0f' : '#f0f4ff' }}
      >
        <Canvas
          camera={{ position: [cameraDistance * 0.6, cameraDistance * 0.4, cameraDistance * 0.8], fov: 50 }}
          gl={{
            antialias: true,
            toneMapping: THREE.ACESFilmicToneMapping,
            toneMappingExposure: 1.2,
          }}
        >
          <NetworkScene nodes={network.nodes} links={network.links} isDark={isDark} />
        </Canvas>
      </SimulationMain>

      {/* Degree Distribution Histogram */}
      <CanvasChart
        data={[{
          x: degrees,
          type: 'histogram' as const,
          marker: { color: '#3b82f6' },
        }]}
        layout={{
          title: { text: 'Degree Distribution', font: { size: 13 } },
          xaxis: { title: { text: 'Degree' } },
          yaxis: { title: { text: 'Frequency' } },
          margin: { t: 40, r: 20, b: 50, l: 60 },
        }}
        style={{ width: '100%', height: 300 }}
      />
    </div>
  );
}
