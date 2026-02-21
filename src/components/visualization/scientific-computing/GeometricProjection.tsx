'use client';

import React, { useState, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import { Slider } from '@/components/ui/slider';
import { SimulationPanel, SimulationLabel } from '@/components/ui/simulation-panel';
import { SimulationMain } from '@/components/ui/simulation-main';
import { useTheme } from '@/lib/use-theme';
import type { SimulationComponentProps } from '@/shared/types/simulation';

type Vec3 = [number, number, number];

/* ------------------------------------------------------------------ */
/*  Arrow3D – shaft (Line) + cone arrowhead + Html label              */
/* ------------------------------------------------------------------ */

interface ArrowProps {
  from: Vec3;
  to: Vec3;
  color: string;
  label: string;
  lineWidth?: number;
  dashed?: boolean;
}

function Arrow3D({ from, to, color, label, lineWidth = 2.5, dashed = false }: ArrowProps) {
  const arrow = useMemo(() => {
    const fromV = new THREE.Vector3(...from);
    const toV = new THREE.Vector3(...to);
    const dir = new THREE.Vector3().subVectors(toV, fromV);
    const len = dir.length();
    if (len < 1e-6) return null;
    dir.normalize();

    const headLength = Math.min(0.15, len * 0.2);
    const headRadius = headLength * 0.35;
    const conePos = toV.clone().sub(dir.clone().multiplyScalar(headLength / 2));
    const quaternion = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      dir,
    );
    const labelPos: Vec3 = [
      toV.x + dir.x * 0.15,
      toV.y + dir.y * 0.15,
      toV.z + dir.z * 0.15,
    ];

    return { conePos, quaternion, headLength, headRadius, labelPos };
  }, [from, to]);

  if (!arrow) return null;

  return (
    <group>
      <Line
        points={[from, to]}
        color={color}
        lineWidth={lineWidth}
        dashed={dashed}
        dashSize={0.12}
        gapSize={0.06}
      />
      <mesh position={arrow.conePos} quaternion={arrow.quaternion}>
        <coneGeometry args={[arrow.headRadius, arrow.headLength, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <Html position={arrow.labelPos} center style={{ pointerEvents: 'none' }}>
        <span
          style={{
            color,
            fontSize: 13,
            fontWeight: 'bold',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            whiteSpace: 'nowrap',
            textShadow: '0 1px 4px rgba(0,0,0,0.4)',
          }}
        >
          {label}
        </span>
      </Html>
    </group>
  );
}

/* ------------------------------------------------------------------ */
/*  Projection result type                                            */
/* ------------------------------------------------------------------ */

interface ProjectionResult {
  a1: number[];
  a2: number[];
  b: number[];
  bHat: number[];
  r: number[];
  rNorm: number;
  rDotA1: number;
  rDotA2: number;
  x0: number;
  x1: number;
}

/* ------------------------------------------------------------------ */
/*  Faint axis lines for spatial orientation                          */
/* ------------------------------------------------------------------ */

function AxisLines({ isDark }: { isDark: boolean }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const positions = new Float32Array([
      -3, 0, 0, 3, 0, 0,
      0, -3, 0, 0, 3, 0,
      0, 0, -3, 0, 0, 3,
    ]);
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    return g;
  }, []);

  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial
        color={isDark ? '#334' : '#b0bdd4'}
        transparent
        opacity={isDark ? 0.25 : 0.4}
      />
    </lineSegments>
  );
}

/* ------------------------------------------------------------------ */
/*  Scene – all R3F objects                                           */
/* ------------------------------------------------------------------ */

function Scene({ result, isDark }: { result: ProjectionResult; isDark: boolean }) {
  const { a1, a2, b, bHat } = result;

  const origin: Vec3 = [0, 0, 0];
  const a1Tip: Vec3 = [a1[0] * 1.5, a1[1] * 1.5, a1[2] * 1.5];
  const a2Tip: Vec3 = [a2[0] * 1.5, a2[1] * 1.5, a2[2] * 1.5];
  const bVec: Vec3 = [b[0], b[1], b[2]];
  const bHatVec: Vec3 = [bHat[0], bHat[1], bHat[2]];

  // Column space plane (parallelogram spanned by a1, a2)
  const planeGeo = useMemo(() => {
    const s = 2.5;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array([
      -s * a1[0] - s * a2[0], -s * a1[1] - s * a2[1], -s * a1[2] - s * a2[2],
       s * a1[0] - s * a2[0],  s * a1[1] - s * a2[1],  s * a1[2] - s * a2[2],
       s * a1[0] + s * a2[0],  s * a1[1] + s * a2[1],  s * a1[2] + s * a2[2],
      -s * a1[0] + s * a2[0], -s * a1[1] + s * a2[1], -s * a1[2] + s * a2[2],
    ]);
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.computeVertexNormals();
    return geo;
  }, [a1, a2]);

  // Plane border (closed loop)
  const planeBorder: Vec3[] = useMemo(() => {
    const s = 2.5;
    return [
      [-s * a1[0] - s * a2[0], -s * a1[1] - s * a2[1], -s * a1[2] - s * a2[2]],
      [ s * a1[0] - s * a2[0],  s * a1[1] - s * a2[1],  s * a1[2] - s * a2[2]],
      [ s * a1[0] + s * a2[0],  s * a1[1] + s * a2[1],  s * a1[2] + s * a2[2]],
      [-s * a1[0] + s * a2[0], -s * a1[1] + s * a2[1], -s * a1[2] + s * a2[2]],
      [-s * a1[0] - s * a2[0], -s * a1[1] - s * a2[1], -s * a1[2] - s * a2[2]],
    ];
  }, [a1, a2]);

  const planeLabelPos: Vec3 = [
    2.5 * 0.7 * a1[0] + 2.5 * 0.7 * a2[0],
    2.5 * 0.7 * a1[1] + 2.5 * 0.7 * a2[1],
    2.5 * 0.7 * a1[2] + 2.5 * 0.7 * a2[2],
  ];

  // Right-angle marker at projection point
  const rightAngle = useMemo<Vec3[] | null>(() => {
    const r = result.r;
    const rLen = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (rLen < 0.01) return null;

    const m = 0.1;
    const rDir = [r[0] / rLen, r[1] / rLen, r[2] / rLen];

    const bHL = Math.sqrt(bHat[0] ** 2 + bHat[1] ** 2 + bHat[2] ** 2);
    const tDir = bHL > 0.01
      ? [bHat[0] / bHL, bHat[1] / bHL, bHat[2] / bHL]
      : [1, 0, 0];

    return [
      [bHat[0] + m * tDir[0], bHat[1] + m * tDir[1], bHat[2] + m * tDir[2]],
      [bHat[0] + m * tDir[0] + m * rDir[0], bHat[1] + m * tDir[1] + m * rDir[1], bHat[2] + m * tDir[2] + m * rDir[2]],
      [bHat[0] + m * rDir[0], bHat[1] + m * rDir[1], bHat[2] + m * rDir[2]],
    ];
  }, [result.r, bHat]);

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={isDark ? 0.3 : 0.6} />
      <directionalLight position={[5, 8, 5]} intensity={isDark ? 0.5 : 0.8} />
      <directionalLight position={[-3, -2, -5]} intensity={isDark ? 0.15 : 0.3} color="#8888ff" />

      <AxisLines isDark={isDark} />

      {/* Column space plane fill */}
      <mesh geometry={planeGeo}>
        <meshBasicMaterial
          color="#3b82f6"
          transparent
          opacity={isDark ? 0.06 : 0.1}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Column space plane border */}
      <Line points={planeBorder} color="#3b82f6" lineWidth={1} transparent opacity={0.25} />

      {/* Col(A) label */}
      <Html position={planeLabelPos} center style={{ pointerEvents: 'none' }}>
        <span
          style={{
            color: 'rgba(59,130,246,0.55)',
            fontSize: 12,
            fontStyle: 'italic',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          }}
        >
          Col(A)
        </span>
      </Html>

      {/* Column vectors */}
      <Arrow3D from={origin} to={a1Tip} color="#8b5cf6" label="a\u2081" />
      <Arrow3D from={origin} to={a2Tip} color="#ec4899" label="a\u2082" />

      {/* Data vector b */}
      <Arrow3D from={origin} to={bVec} color="#ef4444" label="b" lineWidth={3} />

      {/* Projection b_hat */}
      <Arrow3D from={origin} to={bHatVec} color="#10b981" label="b\u0302 = Ax\u0302" lineWidth={3} />

      {/* Residual (dashed) */}
      <Arrow3D from={bHatVec} to={bVec} color="#f59e0b" label="r = b \u2212 Ax\u0302" lineWidth={2} dashed />

      {/* Right-angle mark */}
      {rightAngle && <Line points={rightAngle} color="#f59e0b" lineWidth={1.5} />}

      {/* Camera controls */}
      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        minDistance={2}
        maxDistance={10}
        makeDefault
      />
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                    */
/* ------------------------------------------------------------------ */

export default function GeometricProjection({}: SimulationComponentProps) {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [theta, setTheta] = useState(30);
  const [bAngle, setBAngle] = useState(50);

  const result = useMemo<ProjectionResult>(() => {
    const thetaRad = (theta * Math.PI) / 180;
    const bRad = (bAngle * Math.PI) / 180;

    const a1 = [1, 0, 0];
    const a2 = [Math.cos(thetaRad), Math.sin(thetaRad), 0];

    const bMag = 2.0;
    const inPlane = bMag * Math.cos(bRad * 0.5);
    const outPlane = bMag * Math.sin(bRad * 0.5);
    const b = [
      inPlane * Math.cos(thetaRad / 2),
      inPlane * Math.sin(thetaRad / 2),
      outPlane,
    ];

    const ata00 = a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2];
    const ata01 = a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2];
    const ata11 = a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2];

    const atb0 = a1[0] * b[0] + a1[1] * b[1] + a1[2] * b[2];
    const atb1 = a2[0] * b[0] + a2[1] * b[1] + a2[2] * b[2];

    const det = ata00 * ata11 - ata01 * ata01;
    const x0 = (ata11 * atb0 - ata01 * atb1) / det;
    const x1 = (-ata01 * atb0 + ata00 * atb1) / det;

    const bHat = [
      x0 * a1[0] + x1 * a2[0],
      x0 * a1[1] + x1 * a2[1],
      x0 * a1[2] + x1 * a2[2],
    ];

    const r = [b[0] - bHat[0], b[1] - bHat[1], b[2] - bHat[2]];
    const rNorm = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

    const rDotA1 = r[0] * a1[0] + r[1] * a1[1] + r[2] * a1[2];
    const rDotA2 = r[0] * a2[0] + r[1] * a2[1] + r[2] * a2[2];

    return { a1, a2, b, bHat, r, rNorm, rDotA1, rDotA2, x0, x1 };
  }, [theta, bAngle]);

  return (
    <SimulationPanel>
      <h3 className="text-lg font-semibold text-[var(--text-strong)]">
        Least Squares as Geometric Projection
      </h3>
      <p className="text-sm text-[var(--text-soft)] mb-4">
        The least-squares solution projects <strong>b</strong> onto the column space of A.
        The residual <strong>r</strong> = b &minus; Ax&#770; is perpendicular to Col(A).
        Drag to rotate the 3D view.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div>
          <SimulationLabel>Angle between columns: {theta}&deg;</SimulationLabel>
          <Slider
            value={[theta]}
            onValueChange={([v]) => setTheta(v)}
            min={10}
            max={90}
            step={1}
            className="w-full"
          />
        </div>
        <div>
          <SimulationLabel>b out-of-plane: {bAngle}&deg;</SimulationLabel>
          <Slider
            value={[bAngle]}
            onValueChange={([v]) => setBAngle(v)}
            min={5}
            max={85}
            step={1}
            className="w-full"
          />
        </div>
      </div>

      <SimulationMain
        className="w-full rounded-md border border-[var(--border)] overflow-hidden"
        style={{ height: 420, background: isDark ? '#0a0a14' : '#f8fafc' }}
        scaleMode="fill"
      >
        <Canvas
          camera={{ position: [3, 2.5, 3], fov: 50, near: 0.1, far: 100 }}
          dpr={[1, 2]}
          style={{ width: '100%', height: '100%' }}
        >
          <color attach="background" args={[isDark ? '#0a0a14' : '#f8fafc']} />
          <Scene result={result} isDark={isDark} />
        </Canvas>
      </SimulationMain>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
          <div className="text-xs text-[var(--text-muted)]">||r||</div>
          <div className="text-sm font-mono font-semibold text-[var(--accent)]">
            {result.rNorm.toFixed(4)}
          </div>
        </div>
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
          <div className="text-xs text-[var(--text-muted)]">r &middot; a1</div>
          <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
            {result.rDotA1.toExponential(2)}
          </div>
        </div>
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
          <div className="text-xs text-[var(--text-muted)]">r &middot; a2</div>
          <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
            {result.rDotA2.toExponential(2)}
          </div>
        </div>
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface-2)]/50 p-2.5 text-center">
          <div className="text-xs text-[var(--text-muted)]">
            x&#770; = ({result.x0.toFixed(2)}, {result.x1.toFixed(2)})
          </div>
          <div className="text-sm font-mono font-semibold text-[var(--text-strong)]">
            Coefficients
          </div>
        </div>
      </div>
    </SimulationPanel>
  );
}
