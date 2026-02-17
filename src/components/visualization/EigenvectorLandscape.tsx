'use client';

import { useRef, useState, useMemo, useEffect, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

type Matrix2x2 = number[][];

function computeEigen2x2(matrix: Matrix2x2): { evals: [number, number]; evecs: [[number, number], [number, number]] } {
  const a = matrix[0][0], b = matrix[0][1];
  const c = matrix[1][0], d = matrix[1][1];
  const disc = Math.sqrt((a - d) ** 2 + 4 * b * c);
  const lambda1 = (a + d + disc) / 2;
  const lambda2 = (a + d - disc) / 2;
  let v1: [number, number], v2: [number, number];
  if (Math.abs(b) > 1e-10) {
    v1 = [1, (lambda1 - a) / b];
    v2 = [1, (lambda2 - a) / b];
  } else if (Math.abs(c) > 1e-10) {
    v1 = [(lambda1 - d) / c, 1];
    v2 = [(lambda2 - d) / c, 1];
  } else {
    v1 = [1, 0];
    v2 = [0, 1];
  }
  const norm1 = Math.sqrt(v1[0] ** 2 + v1[1] ** 2);
  const norm2 = Math.sqrt(v2[0] ** 2 + v2[1] ** 2);
  v1 = [v1[0] / norm1, v1[1] / norm1];
  v2 = [v2[0] / norm2, v2[1] / norm2];
  const scale = 3;
  v1 = [v1[0] * scale, v1[1] * scale];
  v2 = [v2[0] * scale, v2[1] * scale];
  return { evals: [lambda1, lambda2], evecs: [v1, v2] };
}

function computeQuadratic(a: number, b: number, c: number, x: number, y: number): number {
  return 0.5 * (a * x * x + 2 * b * x * y + c * y * y);
}

function computeGradient(a: number, b: number, c: number, x: number, y: number): [number, number] {
  return [a * x + b * y, b * x + c * y];
}

export function EigenvectorLandscape() {
  const [a, setA] = useState(3);
  const [b, setB] = useState(1);
  const [c, setC] = useState(2);
  const [resetKey, setResetKey] = useState(0);

  const handleReset = useCallback(() => setResetKey(prev => prev + 1), []);

  const surfaceRef = useRef<THREE.Mesh>(null!);
  const ballRef = useRef<THREE.Group>(null!);
  const velocityRef = useRef([0, 0] as [number, number]);

  const geometry = useMemo(
    () => new THREE.PlaneGeometry(8, 8, 64, 64).rotateX(-Math.PI / 2),
    []
  );

  const { evals, evecs } = useMemo(() => {
    const res = computeEigen2x2([[a, b], [b, c]]);
    return res;
  }, [a, b, c]);

  const updateSurface = useCallback(() => {
    if (!surfaceRef.current) return;
    const geo = surfaceRef.current.geometry as THREE.BufferGeometry;
    const pos = geo.attributes.position as THREE.Float32BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = computeQuadratic(a, b, c, x, y);
      pos.setZ(i, z);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
  }, [a, b, c]);

  useEffect(() => {
    updateSurface();
  }, [updateSurface]);

  useEffect(() => {
    if (ballRef.current) {
      ballRef.current.position.set(2.5, 1.2, 0);
      velocityRef.current = [0, 0];
    }
  }, [resetKey]);

  useFrame((state, delta) => {
    if (!ballRef.current) return;
    const p = ballRef.current.position;
    const [gx, gy] = computeGradient(a, b, c, p.x, p.y);
    velocityRef.current[0] -= gx * 5 * delta;
    velocityRef.current[1] -= gy * 5 * delta;
    velocityRef.current[0] *= 0.95;
    velocityRef.current[1] *= 0.95;
    p.x += velocityRef.current[0] * delta * 2;
    p.y += velocityRef.current[1] * delta * 2;
    // bounds
    p.x = Math.max(-3.5, Math.min(3.5, p.x));
    p.y = Math.max(-3.5, Math.min(3.5, p.y));
    p.z = computeQuadratic(a, b, c, p.x, p.y);
  });

  return (
    <div className="space-y-6">
      <div className="flex gap-4 items-center flex-wrap">
        <span className="text-white">Quadratic Form:</span>
        <label>
          a11: <input type="number" step="0.1" value={a} onChange={(e) => setA(parseFloat(e.target.value) || 0)} className="w-20 px-2 py-1 bg-[#151525] rounded text-white text-center" />
        </label>
        <label>
          a12=a21: <input type="number" step="0.1" value={b} onChange={(e) => setB(parseFloat(e.target.value) || 0)} className="w-20 px-2 py-1 bg-[#151525] rounded text-white text-center" />
        </label>
        <label>
          a22: <input type="number" step="0.1" value={c} onChange={(e) => setC(parseFloat(e.target.value) || 0)} className="w-20 px-2 py-1 bg-[#151525] rounded text-white text-center" />
        </label>
        <span className="text-green-400">λ1={evals[0].toFixed(2)}, λ2={evals[1].toFixed(2)}</span>
        <button onClick={handleReset} className="px-4 py-1 bg-blue-600 rounded hover:bg-blue-700 ml-auto">
          Reset Ball
        </button>
        <button
          onClick={() => {
            setA(3);
            setB(1);
            setC(2);
          }}
          className="px-4 py-1 bg-gray-600 rounded hover:bg-gray-700"
        >
          Reset Matrix
        </button>
      </div>
      <div className="w-full h-96 bg-black rounded-lg">
        <Canvas camera={{ position: [5, 5, 5] }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <mesh ref={surfaceRef} geometry={geometry}>
            <meshPhongMaterial color="#ffaa00" shininess={100} />
          </mesh>
          <group ref={ballRef}>
            <mesh>
              <sphereGeometry args={[0.15]} />
              <meshPhongMaterial color="#00aaff" shininess={100} />
            </mesh>
          </group>
          {/* Eigenvectors */}
          {evecs.map((vec, i) => (
            <arrowHelper
              key={i}
              args={[
                        new THREE.Vector3(vec[0], vec[1], 0).normalize(),
                new THREE.Vector3(0, 0, 0),
                2,
                i === 0 ? "#00ff00" : "#ff0000",
              ]}
            />
          ))}
          <OrbitControls />
        </Canvas>
      </div>
      <p className="text-sm text-gray-400">
        The landscape is z = ½ xᵀ A x. Ball rolls downhill following gradient descent. Eigenvectors (arrows) are directions of no curvature change (steepest ascent/descent paths).
      </p>
    </div>
  );
}