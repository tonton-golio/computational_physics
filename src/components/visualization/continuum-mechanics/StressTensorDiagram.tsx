'use client';

import React from 'react';

function Arrow({ x1, y1, x2, y2, color }: { x1: number; y1: number; x2: number; y2: number; color: string }) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 8;
  const ha1 = angle + Math.PI * 0.82;
  const ha2 = angle - Math.PI * 0.82;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={2} />
      <line x1={x2} y1={y2} x2={x2 + headLen * Math.cos(ha1)} y2={y2 + headLen * Math.sin(ha1)} stroke={color} strokeWidth={2} />
      <line x1={x2} y1={y2} x2={x2 + headLen * Math.cos(ha2)} y2={y2 + headLen * Math.sin(ha2)} stroke={color} strokeWidth={2} />
    </g>
  );
}

/**
 * SVG diagram of a stress tensor cube showing all 9 Cauchy stress components.
 * Renders a 3D-looking cube with labeled arrows for normal and shear stresses.
 */
export default function StressTensorDiagram() {
  // Cube face vertices in isometric-ish projection
  const cx = 260;
  const cy = 200;
  const dx = 80; // half-width along x-axis
  const dy = 80; // half-height along y-axis
  const dz = 60; // shift for z-axis

  // Corner helpers (front face, back face)
  const fl = { x: cx - dx, y: cy + dy };       // front-left
  const fr = { x: cx + dx, y: cy + dy };       // front-right
  const ftl = { x: cx - dx, y: cy - dy };      // front-top-left
  const ftr = { x: cx + dx, y: cy - dy };      // front-top-right

  const bl = { x: fl.x + dz, y: fl.y - dz };  // back-left
  const br = { x: fr.x + dz, y: fr.y - dz };  // back-right
  const btl = { x: ftl.x + dz, y: ftl.y - dz };
  const btr = { x: ftr.x + dz, y: ftr.y - dz };

  const faceStyle = (opacity: number) => ({
    fill: `rgba(59, 130, 246, ${opacity})`,
    stroke: 'rgba(147, 197, 253, 0.7)',
    strokeWidth: 1.5,
  });

  const arrowColor = '#f59e0b';
  const shearColor = '#34d399';
  const labelColor = '#e5e7eb';
  const subLabelColor = '#9ca3af';

  // Mid-points of visible faces for arrow origins
  const rightFaceMid = {
    x: (fr.x + br.x + btr.x + ftr.x) / 4,
    y: (fr.y + br.y + btr.y + ftr.y) / 4,
  };
  const topFaceMid = {
    x: (ftl.x + ftr.x + btr.x + btl.x) / 4,
    y: (ftl.y + ftr.y + btr.y + btl.y) / 4,
  };
  const frontFaceMid = {
    x: (fl.x + fr.x + ftr.x + ftl.x) / 4,
    y: (fl.y + fr.y + ftr.y + ftl.y) / 4,
  };

  const aLen = 50;

  return (
    <div className="w-full rounded-lg p-6 mb-8">
      <h3 className="text-lg font-semibold mb-2 text-[var(--text-strong)]">
        Cauchy Stress Tensor
      </h3>
      <p className="text-sm text-[var(--text-muted)] mb-4">
        The nine components of the stress tensor acting on an infinitesimal cube element.
        Normal stresses (yellow) act perpendicular to each face; shear stresses (green)
        act tangent to them.
      </p>
      <svg viewBox="0 0 520 400" className="w-full max-w-lg mx-auto" style={{ maxHeight: 380 }}>
        {/* Back faces (partially visible) */}
        <polygon
          points={`${btl.x},${btl.y} ${btr.x},${btr.y} ${br.x},${br.y} ${bl.x},${bl.y}`}
          {...faceStyle(0.06)}
        />
        {/* Top face */}
        <polygon
          points={`${ftl.x},${ftl.y} ${ftr.x},${ftr.y} ${btr.x},${btr.y} ${btl.x},${btl.y}`}
          {...faceStyle(0.15)}
        />
        {/* Right face */}
        <polygon
          points={`${ftr.x},${ftr.y} ${fr.x},${fr.y} ${br.x},${br.y} ${btr.x},${btr.y}`}
          {...faceStyle(0.10)}
        />
        {/* Front face */}
        <polygon
          points={`${fl.x},${fl.y} ${fr.x},${fr.y} ${ftr.x},${ftr.y} ${ftl.x},${ftl.y}`}
          {...faceStyle(0.20)}
        />

        {/* Coordinate axes */}
        <Arrow x1={80} y1={350} x2={140} y2={350} color="#6b7280" />
        <Arrow x1={80} y1={350} x2={80} y2={295} color="#6b7280" />
        <Arrow x1={80} y1={350} x2={115} y2={320} color="#6b7280" />
        <text x={148} y={354} fill={subLabelColor} fontSize={14} fontStyle="italic">x</text>
        <text x={72} y={290} fill={subLabelColor} fontSize={14} fontStyle="italic">y</text>
        <text x={120} y={316} fill={subLabelColor} fontSize={14} fontStyle="italic">z</text>

        {/* === RIGHT FACE (x-face): normal = +x === */}
        {/* σ_xx: normal stress pointing right */}
        <Arrow x1={rightFaceMid.x} y1={rightFaceMid.y} x2={rightFaceMid.x + aLen} y2={rightFaceMid.y} color={arrowColor} />
        <text x={rightFaceMid.x + aLen + 4} y={rightFaceMid.y + 5} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>xx</tspan>
        </text>
        {/* σ_xy: shear on x-face in y-direction */}
        <Arrow x1={rightFaceMid.x} y1={rightFaceMid.y} x2={rightFaceMid.x} y2={rightFaceMid.y - aLen} color={shearColor} />
        <text x={rightFaceMid.x + 4} y={rightFaceMid.y - aLen - 4} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>xy</tspan>
        </text>
        {/* σ_xz: shear on x-face in z-direction */}
        <Arrow x1={rightFaceMid.x} y1={rightFaceMid.y} x2={rightFaceMid.x + dz * 0.65} y2={rightFaceMid.y - dz * 0.65} color={shearColor} />
        <text x={rightFaceMid.x + dz * 0.65 + 4} y={rightFaceMid.y - dz * 0.65 - 2} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>xz</tspan>
        </text>

        {/* === TOP FACE (y-face): normal = +y === */}
        {/* σ_yy: normal stress pointing up */}
        <Arrow x1={topFaceMid.x} y1={topFaceMid.y} x2={topFaceMid.x} y2={topFaceMid.y - aLen} color={arrowColor} />
        <text x={topFaceMid.x + 4} y={topFaceMid.y - aLen - 4} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>yy</tspan>
        </text>
        {/* σ_yx: shear on y-face in x-direction */}
        <Arrow x1={topFaceMid.x} y1={topFaceMid.y} x2={topFaceMid.x + aLen} y2={topFaceMid.y} color={shearColor} />
        <text x={topFaceMid.x + aLen + 4} y={topFaceMid.y + 5} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>yx</tspan>
        </text>
        {/* σ_yz: shear on y-face in z-direction */}
        <Arrow x1={topFaceMid.x} y1={topFaceMid.y} x2={topFaceMid.x + dz * 0.65} y2={topFaceMid.y - dz * 0.65} color={shearColor} />
        <text x={topFaceMid.x + dz * 0.65 + 4} y={topFaceMid.y - dz * 0.65 - 2} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>yz</tspan>
        </text>

        {/* === FRONT FACE (z-face): normal = -z (towards viewer) === */}
        {/* σ_zz: normal stress pointing towards viewer (approximated as diagonal) */}
        <Arrow x1={frontFaceMid.x} y1={frontFaceMid.y} x2={frontFaceMid.x - dz * 0.65} y2={frontFaceMid.y + dz * 0.65} color={arrowColor} />
        <text x={frontFaceMid.x - dz * 0.65 - 36} y={frontFaceMid.y + dz * 0.65 + 14} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>zz</tspan>
        </text>
        {/* σ_zx: shear on z-face in x-direction */}
        <Arrow x1={frontFaceMid.x} y1={frontFaceMid.y} x2={frontFaceMid.x + aLen} y2={frontFaceMid.y} color={shearColor} />
        <text x={frontFaceMid.x + aLen + 4} y={frontFaceMid.y + 5} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>zx</tspan>
        </text>
        {/* σ_zy: shear on z-face in y-direction */}
        <Arrow x1={frontFaceMid.x} y1={frontFaceMid.y} x2={frontFaceMid.x} y2={frontFaceMid.y - aLen} color={shearColor} />
        <text x={frontFaceMid.x - 30} y={frontFaceMid.y - aLen - 4} fill={labelColor} fontSize={13} fontFamily="serif" fontStyle="italic">
          &sigma;<tspan baselineShift="sub" fontSize={10}>zy</tspan>
        </text>
      </svg>
      <div className="mt-2 text-xs text-[var(--text-soft)] text-center">
        <span className="inline-block w-3 h-0.5 mr-1 align-middle" style={{ background: arrowColor }} /> Normal stresses
        <span className="mx-3">|</span>
        <span className="inline-block w-3 h-0.5 mr-1 align-middle" style={{ background: shearColor }} /> Shear stresses
      </div>
    </div>
  );
}
