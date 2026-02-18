'use client';

import { useState } from 'react';

interface UNetArchitectureProps {
  id?: string;
}

interface LayerInfo {
  label: string;
  dims: string;
  description: string;
  type: 'encoder' | 'bottleneck' | 'decoder' | 'skip' | 'output';
}

const LAYERS: LayerInfo[] = [
  { label: 'Input', dims: '128x128x1', description: 'Input image (grayscale)', type: 'encoder' },
  { label: 'Enc 1', dims: '128x128x64', description: 'Two 3x3 conv + ReLU', type: 'encoder' },
  { label: 'Enc 2', dims: '64x64x128', description: 'MaxPool 2x2, then two 3x3 conv + ReLU', type: 'encoder' },
  { label: 'Enc 3', dims: '32x32x256', description: 'MaxPool 2x2, then two 3x3 conv + ReLU', type: 'encoder' },
  { label: 'Bottleneck', dims: '16x16x512', description: 'MaxPool 2x2, two 3x3 conv + ReLU. Deepest feature representation.', type: 'bottleneck' },
  { label: 'Dec 3', dims: '32x32x256', description: 'Upsample 2x, concatenate with Enc 3 (skip), two 3x3 conv', type: 'decoder' },
  { label: 'Dec 2', dims: '64x64x128', description: 'Upsample 2x, concatenate with Enc 2 (skip), two 3x3 conv', type: 'decoder' },
  { label: 'Dec 1', dims: '128x128x64', description: 'Upsample 2x, concatenate with Enc 1 (skip), two 3x3 conv', type: 'decoder' },
  { label: 'Output', dims: '128x128xC', description: '1x1 convolution to C class probabilities per pixel', type: 'output' },
];

export default function UNetArchitecture({ id: _id }: UNetArchitectureProps) {
  const [hoveredLayer, setHoveredLayer] = useState<number | null>(null);

  const layerColor = (type: string, isHovered: boolean) => {
    const alpha = isHovered ? 1 : 0.7;
    switch (type) {
      case 'encoder': return `rgba(59, 130, 246, ${alpha})`;
      case 'bottleneck': return `rgba(139, 92, 246, ${alpha})`;
      case 'decoder': return `rgba(16, 185, 129, ${alpha})`;
      case 'output': return `rgba(239, 68, 68, ${alpha})`;
      default: return `rgba(100, 100, 100, ${alpha})`;
    }
  };

  // U-shape layout positions
  const positions = [
    { x: 60, y: 30, w: 90, h: 40 },   // Input
    { x: 60, y: 80, w: 90, h: 40 },   // Enc 1
    { x: 170, y: 100, w: 80, h: 35 },  // Enc 2
    { x: 270, y: 120, w: 70, h: 30 },  // Enc 3
    { x: 355, y: 145, w: 70, h: 30 },  // Bottleneck
    { x: 440, y: 120, w: 70, h: 30 },  // Dec 3
    { x: 530, y: 100, w: 80, h: 35 },  // Dec 2
    { x: 630, y: 80, w: 90, h: 40 },   // Dec 1
    { x: 630, y: 30, w: 90, h: 40 },   // Output
  ];

  // Arrows between sequential layers
  const arrows = [
    [0, 1], [1, 2], [2, 3], [3, 4], // encoder path
    [4, 5], [5, 6], [6, 7], [7, 8], // decoder path
  ];

  // Skip connections
  const skips = [
    { from: 1, to: 7 }, // Enc 1 -> Dec 1
    { from: 2, to: 6 }, // Enc 2 -> Dec 2
    { from: 3, to: 5 }, // Enc 3 -> Dec 3
  ];

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        U-Net Architecture
      </h3>

      <svg viewBox="0 0 780 220" className="w-full h-auto" style={{ maxHeight: 350 }}>
        {/* Skip connections (draw first, behind) */}
        {skips.map((skip, i) => {
          const from = positions[skip.from];
          const to = positions[skip.to];
          const isHighlighted = hoveredLayer === skip.from || hoveredLayer === skip.to;
          const fromCx = from.x + from.w / 2;
          const fromCy = from.y + from.h / 2;
          const toCx = to.x + to.w / 2;
          const toCy = to.y + to.h / 2;
          // Curved path above the blocks
          const midY = Math.min(fromCy, toCy) - 30 - i * 15;
          return (
            <path
              key={`skip-${i}`}
              d={`M ${fromCx + from.w / 2} ${fromCy} C ${fromCx + 60} ${midY}, ${toCx - 60} ${midY}, ${toCx - to.w / 2} ${toCy}`}
              fill="none"
              stroke={isHighlighted ? '#f59e0b' : '#f59e0b'}
              strokeWidth={isHighlighted ? 2.5 : 1.5}
              strokeDasharray="6,3"
              opacity={isHighlighted ? 1 : 0.4}
            />
          );
        })}

        {/* Sequential arrows */}
        {arrows.map(([fromIdx, toIdx], i) => {
          const from = positions[fromIdx];
          const to = positions[toIdx];
          const x1 = from.x + from.w / 2;
          const y1 = from.y + from.h;
          const x2 = to.x + to.w / 2;
          const y2 = to.y;

          // For horizontal connections
          const isDown = toIdx <= 4;
          const sx = isDown ? from.x + from.w : from.x + from.w;
          const sy = from.y + from.h / 2;
          const ex = isDown ? to.x : to.x;
          const ey = to.y + to.h / 2;

          if (Math.abs(fromIdx - toIdx) === 1 && (fromIdx === 0 || toIdx === 8 || fromIdx === 7)) {
            // Vertical arrows (input->enc1, dec1->output)
            return (
              <line
                key={`arrow-${i}`}
                x1={x1} y1={y1}
                x2={x2} y2={y2}
                stroke="#666"
                strokeWidth={1.5}
                markerEnd="url(#unet-arrow)"
              />
            );
          }

          return (
            <line
              key={`arrow-${i}`}
              x1={sx} y1={sy}
              x2={ex} y2={ey}
              stroke="#666"
              strokeWidth={1.5}
              markerEnd="url(#unet-arrow)"
            />
          );
        })}

        <defs>
          <marker id="unet-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill="#666" />
          </marker>
        </defs>

        {/* Layer blocks */}
        {positions.map((pos, i) => {
          const layer = LAYERS[i];
          const isHovered = hoveredLayer === i;
          return (
            <g
              key={i}
              onMouseEnter={() => setHoveredLayer(i)}
              onMouseLeave={() => setHoveredLayer(null)}
              className="cursor-pointer"
            >
              <rect
                x={pos.x}
                y={pos.y}
                width={pos.w}
                height={pos.h}
                rx={6}
                fill={layerColor(layer.type, isHovered)}
                stroke={isHovered ? '#fff' : 'transparent'}
                strokeWidth={isHovered ? 2 : 0}
              />
              <text
                x={pos.x + pos.w / 2}
                y={pos.y + pos.h / 2 - 4}
                textAnchor="middle"
                fill="#fff"
                fontSize={10}
                fontWeight={600}
              >
                {layer.label}
              </text>
              <text
                x={pos.x + pos.w / 2}
                y={pos.y + pos.h / 2 + 10}
                textAnchor="middle"
                fill="rgba(255,255,255,0.7)"
                fontSize={8}
              >
                {layer.dims}
              </text>
            </g>
          );
        })}

        {/* Labels */}
        <text x="100" y="195" textAnchor="middle" fill="#3b82f6" fontSize={11} fontWeight={600}>Encoder</text>
        <text x="390" y="195" textAnchor="middle" fill="#8b5cf6" fontSize={11} fontWeight={600}>Bottleneck</text>
        <text x="630" y="195" textAnchor="middle" fill="#10b981" fontSize={11} fontWeight={600}>Decoder</text>
        <text x="390" y="12" textAnchor="middle" fill="#f59e0b" fontSize={10}>Skip Connections (dashed)</text>
      </svg>

      {/* Info panel */}
      <div className="mt-4 p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        {hoveredLayer !== null ? (
          <div>
            <span className="font-semibold text-[var(--text-strong)]">{LAYERS[hoveredLayer].label}</span>
            <span className="ml-2 text-blue-300">[{LAYERS[hoveredLayer].dims}]</span>
            <p className="mt-1">{LAYERS[hoveredLayer].description}</p>
          </div>
        ) : (
          <p>Hover over a layer to see its dimensions and role. The <strong className="text-blue-400">encoder</strong> downsamples to extract features, the <strong className="text-purple-400">bottleneck</strong> captures high-level context, and the <strong className="text-green-400">decoder</strong> upsamples with <strong className="text-yellow-400">skip connections</strong> preserving fine spatial detail.</p>
        )}
      </div>
    </div>
  );
}
