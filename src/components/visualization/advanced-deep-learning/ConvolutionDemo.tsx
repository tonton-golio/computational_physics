'use client';

import { useState, useMemo } from 'react';
import { CanvasHeatmap } from '@/components/ui/canvas-heatmap';
import { Slider } from '@/components/ui/slider';
import { useTheme } from '@/lib/use-theme';

interface ConvolutionDemoProps {
  id?: string;
}

type KernelType = 'identity' | 'edge-detect' | 'sharpen' | 'blur' | 'emboss' | 'custom';

const PRESET_KERNELS: Record<Exclude<KernelType, 'custom'>, { name: string; kernel: number[][]; description: string }> = {
  identity: {
    name: 'Identity',
    kernel: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
    description: 'Passes through the input unchanged. Only the center weight is 1.',
  },
  'edge-detect': {
    name: 'Edge Detection',
    kernel: [
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1],
    ],
    description: 'Detects edges by highlighting pixels that differ from their neighbors. Laplacian operator.',
  },
  sharpen: {
    name: 'Sharpen',
    kernel: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
    description: 'Enhances edges by amplifying the center pixel relative to neighbors.',
  },
  blur: {
    name: 'Box Blur',
    kernel: [
      [1 / 9, 1 / 9, 1 / 9],
      [1 / 9, 1 / 9, 1 / 9],
      [1 / 9, 1 / 9, 1 / 9],
    ],
    description: 'Averages each pixel with its neighbors, smoothing the image.',
  },
  emboss: {
    name: 'Emboss',
    kernel: [
      [-2, -1, 0],
      [-1, 1, 1],
      [0, 1, 2],
    ],
    description: 'Creates a 3D raised effect by emphasizing directional intensity differences.',
  },
};

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateImage(size: number, pattern: string): number[][] {
  const rng = mulberry32(123);
  const img: number[][] = [];
  for (let i = 0; i < size; i++) {
    img.push([]);
    for (let j = 0; j < size; j++) {
      switch (pattern) {
        case 'checkerboard':
          img[i].push((i + j) % 2 === 0 ? 1.0 : 0.0);
          break;
        case 'gradient':
          img[i].push(j / (size - 1));
          break;
        case 'cross':
          img[i].push(i === Math.floor(size / 2) || j === Math.floor(size / 2) ? 1.0 : 0.0);
          break;
        case 'diagonal':
          img[i].push(Math.abs(i - j) <= 1 ? 1.0 : 0.0);
          break;
        case 'random':
          img[i].push(rng());
          break;
        default:
          img[i].push(0);
      }
    }
  }
  return img;
}

function convolve2d(image: number[][], kernel: number[][]): number[][] {
  const imgH = image.length;
  const imgW = image[0].length;
  const kH = kernel.length;
  const kW = kernel[0].length;
  const padH = Math.floor(kH / 2);
  const padW = Math.floor(kW / 2);

  const result: number[][] = [];
  for (let i = 0; i < imgH; i++) {
    result.push([]);
    for (let j = 0; j < imgW; j++) {
      let sum = 0;
      for (let ki = 0; ki < kH; ki++) {
        for (let kj = 0; kj < kW; kj++) {
          const ii = i + ki - padH;
          const jj = j + kj - padW;
          if (ii >= 0 && ii < imgH && jj >= 0 && jj < imgW) {
            sum += image[ii][jj] * kernel[ki][kj];
          }
        }
      }
      result[i].push(sum);
    }
  }
  return result;
}

export default function ConvolutionDemo({ id: _id }: ConvolutionDemoProps) {
  const theme = useTheme();
  const isDark = theme === 'dark';
  const [kernelType, setKernelType] = useState<KernelType>('edge-detect');
  const [pattern, setPattern] = useState('cross');
  const [imageSize, setImageSize] = useState(10);
  const [stepMode, setStepMode] = useState(false);
  const [highlightRow, setHighlightRow] = useState(0);
  const [highlightCol, setHighlightCol] = useState(0);
  const [customKernel, setCustomKernel] = useState<number[][]>([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ]);

  const kernel = useMemo(() => {
    if (kernelType === 'custom') return customKernel;
    return PRESET_KERNELS[kernelType].kernel;
  }, [kernelType, customKernel]);

  const inputImage = useMemo(() => generateImage(imageSize, pattern), [imageSize, pattern]);
  const outputImage = useMemo(() => convolve2d(inputImage, kernel), [inputImage, kernel]);

  // Compute breakdown for highlighted pixel
  const breakdown = useMemo(() => {
    if (!stepMode) return null;
    const r = highlightRow;
    const c = highlightCol;
    const kH = kernel.length;
    const kW = kernel[0].length;
    const padH = Math.floor(kH / 2);
    const padW = Math.floor(kW / 2);
    const terms: { inputVal: number; kernelVal: number; product: number; ir: number; ic: number }[] = [];
    let sum = 0;
    for (let ki = 0; ki < kH; ki++) {
      for (let kj = 0; kj < kW; kj++) {
        const ii = r + ki - padH;
        const jj = c + kj - padW;
        const inVal = (ii >= 0 && ii < imageSize && jj >= 0 && jj < imageSize) ? inputImage[ii][jj] : 0;
        const kVal = kernel[ki][kj];
        const prod = inVal * kVal;
        terms.push({ inputVal: inVal, kernelVal: kVal, product: prod, ir: ii, ic: jj });
        sum += prod;
      }
    }
    return { terms, sum, outputVal: outputImage[r]?.[c] ?? 0 };
  }, [stepMode, highlightRow, highlightCol, kernel, inputImage, outputImage, imageSize]);

  // Build traces
  const inputTrace: any = {
    type: 'heatmap' as const,
    z: [...inputImage].reverse(),
    colorscale: 'Greys',
    showscale: false,
    hovertemplate: 'row: %{y}<br>col: %{x}<br>value: %{z:.2f}<extra>Input</extra>',
  };

  const kernelTrace: any = {
    type: 'heatmap' as const,
    z: [...kernel].reverse(),
    colorscale: isDark
      ? [
          [0, '#1a1a2e'],
          [0.5, '#16213e'],
          [1, '#e94560'],
        ]
      : [
          [0, '#dfe8fb'],
          [0.5, '#b0bdd4'],
          [1, '#e94560'],
        ],
    showscale: false,
    hovertemplate: 'row: %{y}<br>col: %{x}<br>weight: %{z:.3f}<extra>Kernel</extra>',
  };

  const outputTrace: any = {
    type: 'heatmap' as const,
    z: [...outputImage].reverse(),
    colorscale: 'Viridis',
    showscale: true,
    hovertemplate: 'row: %{y}<br>col: %{x}<br>value: %{z:.2f}<extra>Output</extra>',
  };

  // Add highlight shapes for step mode
  const inputShapes: any[] = [];
  const outputShapes: any[] = [];
  if (stepMode) {
    const kH = kernel.length;
    const padH = Math.floor(kH / 2);
    // Highlight receptive field on input
    const rStart = highlightRow - padH;
    const rEnd = highlightRow + padH;
    const cStart = highlightCol - padH;
    const cEnd = highlightCol + padH;
    inputShapes.push({
      type: 'rect',
      x0: cStart - 0.5,
      x1: cEnd + 0.5,
      y0: (imageSize - 1 - rEnd) - 0.5,
      y1: (imageSize - 1 - rStart) + 0.5,
      line: { color: '#f39c12', width: 3 },
    });
    // Highlight output pixel
    outputShapes.push({
      type: 'rect',
      x0: highlightCol - 0.5,
      x1: highlightCol + 0.5,
      y0: (imageSize - 1 - highlightRow) - 0.5,
      y1: (imageSize - 1 - highlightRow) + 0.5,
      line: { color: '#e74c3c', width: 3 },
    });
  }

  const commonLayout = {
    margin: { t: 35, b: 30, l: 30, r: 30 },
    xaxis: { showticklabels: false, showgrid: false },
    yaxis: { showticklabels: false, showgrid: false, scaleanchor: 'x' as const },
  };

  const updateCustomKernel = (row: number, col: number, value: string) => {
    const newKernel = customKernel.map((r) => [...r]);
    newKernel[row][col] = parseFloat(value) || 0;
    setCustomKernel(newKernel);
  };

  const kernelDesc = kernelType !== 'custom' ? PRESET_KERNELS[kernelType as Exclude<KernelType, 'custom'>].description : 'Your custom kernel.';

  return (
    <div className="w-full bg-[var(--surface-1)] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-[var(--text-strong)]">
        2D Convolution Visualization
      </h3>

      {/* Controls Row */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">Kernel</label>
          <select
            value={kernelType}
            onChange={(e) => setKernelType(e.target.value as KernelType)}
            className="bg-[var(--surface-2)] text-[var(--text-strong)] border border-[var(--border-strong)] rounded px-3 py-1.5 text-sm"
          >
            {Object.entries(PRESET_KERNELS).map(([key, val]) => (
              <option key={key} value={key}>
                {val.name}
              </option>
            ))}
            <option value="custom">Custom</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">Input Pattern</label>
          <select
            value={pattern}
            onChange={(e) => setPattern(e.target.value)}
            className="bg-[var(--surface-2)] text-[var(--text-strong)] border border-[var(--border-strong)] rounded px-3 py-1.5 text-sm"
          >
            <option value="cross">Cross</option>
            <option value="checkerboard">Checkerboard</option>
            <option value="gradient">Gradient</option>
            <option value="diagonal">Diagonal</option>
            <option value="random">Random</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-[var(--text-muted)] mb-1">
            Grid Size: {imageSize}x{imageSize}
          </label>
          <Slider
            min={5}
            max={16}
            value={[imageSize]}
            onValueChange={([v]) => setImageSize(v)}
            className="w-32"
          />
        </div>
        <div className="flex items-end">
          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer">
            <input
              type="checkbox"
              checked={stepMode}
              onChange={(e) => setStepMode(e.target.checked)}
              className="accent-blue-500"
            />
            Step-by-step mode
          </label>
        </div>
      </div>

      {/* Step-by-step controls */}
      {stepMode && (
        <div className="flex flex-wrap gap-4 mb-4 p-3 bg-[var(--surface-2)] rounded">
          <div>
            <label className="block text-sm text-[var(--text-muted)] mb-1">
              Output row: {highlightRow}
            </label>
            <Slider
              min={0}
              max={imageSize - 1}
              step={1}
              value={[highlightRow]}
              onValueChange={([v]) => setHighlightRow(v)}
              className="w-32"
            />
          </div>
          <div>
            <label className="block text-sm text-[var(--text-muted)] mb-1">
              Output col: {highlightCol}
            </label>
            <Slider
              min={0}
              max={imageSize - 1}
              step={1}
              value={[highlightCol]}
              onValueChange={([v]) => setHighlightCol(v)}
              className="w-32"
            />
          </div>
          {breakdown && (
            <div className="flex-1 min-w-[200px] text-xs text-[var(--text-muted)]">
              <p className="font-semibold text-[var(--text-strong)] mb-1">
                Output[{highlightRow},{highlightCol}] = {breakdown.sum.toFixed(3)}
              </p>
              <div className="flex flex-wrap gap-1">
                {breakdown.terms.map((t, i) => (
                  <span key={i} className={t.product !== 0 ? 'text-blue-300' : 'opacity-50'}>
                    {i > 0 && '+ '}
                    {t.inputVal.toFixed(1)}*{t.kernelVal.toFixed(1)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Custom kernel editor */}
      {kernelType === 'custom' && (
        <div className="mb-4 p-3 bg-[var(--surface-2)] rounded">
          <p className="text-sm text-[var(--text-muted)] mb-2">Edit 3x3 kernel values:</p>
          <div className="grid grid-cols-3 gap-1 max-w-xs">
            {customKernel.map((row, i) =>
              row.map((val, j) => (
                <input
                  key={`${i}-${j}`}
                  type="number"
                  step="0.1"
                  value={val}
                  onChange={(e) => updateCustomKernel(i, j, e.target.value)}
                  className="bg-[var(--surface-1)] text-[var(--text-strong)] border border-[var(--border-strong)] rounded px-2 py-1 text-sm text-center w-20"
                />
              ))
            )}
          </div>
        </div>
      )}

      {/* Three-panel visualization */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
        <div>
          <p className="text-center text-sm text-[var(--text-muted)] mb-2 font-semibold">Input Image</p>
          <CanvasHeatmap
            data={[inputTrace]}
            layout={{
              ...commonLayout,
              title: { text: '' },
              shapes: inputShapes,
            }}
            style={{ width: '100%', height: '300px' }}
          />
        </div>

        <div>
          <p className="text-center text-sm text-[var(--text-muted)] mb-2 font-semibold">
            Kernel ({kernelType === 'custom' ? 'Custom' : PRESET_KERNELS[kernelType as Exclude<KernelType, 'custom'>].name})
          </p>
          <CanvasHeatmap
            data={[kernelTrace]}
            layout={{
              ...commonLayout,
              title: { text: '' },
            }}
            style={{ width: '100%', height: '300px' }}
          />
          <div className="mt-2 text-center">
            <table className="mx-auto text-xs text-[var(--text-muted)]">
              <tbody>
                {kernel.map((row, i) => (
                  <tr key={i}>
                    {row.map((val, j) => (
                      <td
                        key={j}
                        className="px-2 py-1 border border-[var(--border-strong)]"
                        style={{
                          backgroundColor:
                            val > 0
                              ? `rgba(233, 69, 96, ${Math.min(Math.abs(val) / 8, 0.6)})`
                              : val < 0
                              ? `rgba(69, 183, 209, ${Math.min(Math.abs(val) / 8, 0.6)})`
                              : 'transparent',
                        }}
                      >
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <p className="text-center text-sm text-[var(--text-muted)] mb-2 font-semibold">Output (Convolved)</p>
          <CanvasHeatmap
            data={[outputTrace]}
            layout={{
              ...commonLayout,
              title: { text: '' },
              shapes: outputShapes,
            }}
            style={{ width: '100%', height: '300px' }}
          />
        </div>
      </div>

      {/* Explanation */}
      <div className="mt-4 p-3 bg-[var(--surface-2)] rounded text-sm text-[var(--text-muted)]">
        <p>{kernelDesc}</p>
        <p className="mt-2">
          The convolution operation slides the kernel across the input image. At each position, the kernel
          values are multiplied element-wise with the overlapping input values, and the results are summed
          to produce one output pixel. Zero-padding is used at the borders to maintain the output size.
          {stepMode && ' Use the row/col sliders to see exactly which input pixels contribute to each output pixel.'}
        </p>
      </div>
    </div>
  );
}
