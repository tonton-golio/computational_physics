'use client';

import React, { useRef, useState, useCallback, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { Slider } from '@/components/ui/slider';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useTheme } from '@/lib/use-theme';

// ---------------------------------------------------------------------------
// GLSL Shaders
// ---------------------------------------------------------------------------

const mandelbrotVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

const mandelbrotFragmentShader = /* glsl */ `
  precision highp float;

  uniform vec2 uCenter;
  uniform float uZoom;
  uniform int uMaxIter;
  uniform float uTime;
  uniform vec2 uResolution;

  varying vec2 vUv;

  vec3 palette(float t) {
    // Multi-stop neon / psychedelic palette
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.00, 0.10, 0.20);
    return a + b * cos(6.28318 * (c * t + d));
  }

  void main() {
    // Map UV to complex plane, correcting for aspect ratio
    float aspect = uResolution.x / uResolution.y;
    vec2 uv = vUv * 2.0 - 1.0;
    uv.x *= aspect;

    // Scale by zoom and translate to center
    vec2 c = uv / uZoom + uCenter;

    // Mandelbrot iteration
    vec2 z = vec2(0.0);
    int n = 0;
    for (int i = 0; i < 1000; i++) {
      if (i >= uMaxIter) break;
      float x2 = z.x * z.x;
      float y2 = z.y * z.y;
      if (x2 + y2 > 256.0) break;
      z = vec2(x2 - y2 + c.x, 2.0 * z.x * z.y + c.y);
      n++;
    }

    if (n >= uMaxIter) {
      // Interior of the set: deep black
      gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
      // Smooth iteration count for continuous coloring
      float smoothVal = float(n) - log2(log2(dot(z, z))) + 4.0;

      // Apply palette with time-based color cycling
      float t = smoothVal * 0.02 + uTime * 0.08;
      vec3 col = palette(t);

      gl_FragColor = vec4(col, 1.0);
    }
  }
`;

// ---------------------------------------------------------------------------
// Inner R3F scene component: fullscreen quad with the Mandelbrot shader
// ---------------------------------------------------------------------------

interface MandelbrotQuadProps {
  centerRef: React.RefObject<[number, number]>;
  zoomRef: React.RefObject<number>;
  maxIter: number;
}

function MandelbrotQuad({ centerRef, zoomRef, maxIter }: MandelbrotQuadProps) {
  const materialRef = useRef<THREE.ShaderMaterial>(null!);
  const { size } = useThree();

  const uniforms = useMemo(
    () => ({
      uCenter: { value: new THREE.Vector2(centerRef.current![0], centerRef.current![1]) },
      uZoom: { value: zoomRef.current! },
      uMaxIter: { value: maxIter },
      uTime: { value: 0 },
      uResolution: { value: new THREE.Vector2(size.width, size.height) },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.uResolution.value.set(size.width, size.height);
    }
  }, [size]);

  useFrame(({ clock }) => {
    const mat = materialRef.current;
    if (!mat) return;
    mat.uniforms.uCenter.value.set(centerRef.current![0], centerRef.current![1]);
    mat.uniforms.uZoom.value = zoomRef.current!;
    mat.uniforms.uMaxIter.value = maxIter;
    mat.uniforms.uTime.value = clock.getElapsedTime();
  });

  return (
    <mesh frustumCulled={false}>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={mandelbrotVertexShader}
        fragmentShader={mandelbrotFragmentShader}
        uniforms={uniforms}
      />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Default view parameters
// ---------------------------------------------------------------------------

const DEFAULT_CENTER: [number, number] = [-0.5, 0.0];
const DEFAULT_ZOOM = 1.0;
const ZOOM_SPEED = 1.1;

// ---------------------------------------------------------------------------
// Exported component
// ---------------------------------------------------------------------------

export function MandelbrotFractal() {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [maxIter, setMaxIter] = useState<number[]>([200]);
  const [displayCenter, setDisplayCenter] = useState<[number, number]>(DEFAULT_CENTER);
  const [displayZoom, setDisplayZoom] = useState<number>(DEFAULT_ZOOM);

  const centerRef = useRef<[number, number]>([...DEFAULT_CENTER]);
  const zoomRef = useRef<number>(DEFAULT_ZOOM);

  const isDragging = useRef(false);
  const lastMouse = useRef<[number, number]>([0, 0]);
  const containerRef = useRef<HTMLDivElement>(null);

  const syncDisplay = useCallback(() => {
    setDisplayCenter([centerRef.current[0], centerRef.current[1]]);
    setDisplayZoom(zoomRef.current);
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    isDragging.current = true;
    lastMouse.current = [e.clientX, e.clientY];
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const dx = e.clientX - lastMouse.current[0];
    const dy = e.clientY - lastMouse.current[1];
    lastMouse.current = [e.clientX, e.clientY];

    const aspect = rect.width / rect.height;
    const scale = 2.0 / zoomRef.current;
    centerRef.current = [
      centerRef.current[0] - (dx / rect.width) * scale * aspect,
      centerRef.current[1] + (dy / rect.height) * scale,
    ];
    syncDisplay();
  }, [syncDisplay]);

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
  }, []);

  const handleMouseLeave = useCallback(() => {
    isDragging.current = false;
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const aspect = rect.width / rect.height;

    const mx = ((e.clientX - rect.left) / rect.width) * 2.0 - 1.0;
    const my = -(((e.clientY - rect.top) / rect.height) * 2.0 - 1.0);

    const beforeX = centerRef.current[0] + (mx * aspect) / zoomRef.current;
    const beforeY = centerRef.current[1] + my / zoomRef.current;

    const factor = e.deltaY < 0 ? ZOOM_SPEED : 1.0 / ZOOM_SPEED;
    zoomRef.current *= factor;

    const afterX = centerRef.current[0] + (mx * aspect) / zoomRef.current;
    const afterY = centerRef.current[1] + my / zoomRef.current;

    centerRef.current = [
      centerRef.current[0] + (beforeX - afterX),
      centerRef.current[1] + (beforeY - afterY),
    ];

    syncDisplay();
  }, [syncDisplay]);

  const handleReset = useCallback(() => {
    centerRef.current = [...DEFAULT_CENTER];
    zoomRef.current = DEFAULT_ZOOM;
    syncDisplay();
  }, [syncDisplay]);

  const fmt = (n: number) => {
    if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(4);
    return n.toFixed(6);
  };

  return (
    <div className="space-y-4">
      {/* Canvas container */}
      <div
        ref={containerRef}
        className="relative w-full rounded-lg overflow-hidden border border-[var(--border-strong)]"
        style={{ height: 600, background: isDark ? '#000' : '#f0f4ff', cursor: 'grab' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
      >
        <Canvas
          orthographic
          camera={{ left: -1, right: 1, top: 1, bottom: -1, near: 0, far: 1 }}
          gl={{ antialias: false, preserveDrawingBuffer: true }}
          style={{ width: '100%', height: '100%' }}
          resize={{ scroll: false }}
        >
          <MandelbrotQuad
            centerRef={centerRef}
            zoomRef={zoomRef}
            maxIter={maxIter[0]}
          />
        </Canvas>
      </div>

      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Mandelbrot Explorer</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-sm font-medium text-[var(--text-strong)]">
              Max Iterations: {maxIter[0]}
            </label>
            <Slider
              value={maxIter}
              onValueChange={setMaxIter}
              min={100}
              max={1000}
              step={10}
            />
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <div className="text-sm text-[var(--text-muted)] font-mono space-y-1">
              <div>Center: ({fmt(displayCenter[0])}, {fmt(displayCenter[1])}i)</div>
              <div>Zoom: {displayZoom.toFixed(2)}x</div>
            </div>
            <button
              onClick={handleReset}
              className="ml-auto px-4 py-2 rounded-md text-sm font-medium bg-[var(--accent)] hover:bg-[var(--accent-strong)] text-white transition-colors"
            >
              Reset View
            </button>
          </div>

          <p className="text-xs text-[var(--text-muted)]">
            Scroll to zoom. Click and drag to pan. Increase iterations for more detail at deep zoom levels.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
