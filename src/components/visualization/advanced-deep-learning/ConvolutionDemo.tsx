'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ConvolutionDemoProps {
  id?: string;
}

type KernelType = 'identity' | 'edge-detect' | 'sharpen' | 'blur' | 'emboss' | 'custom';

const PRESET_KERNELS: Record<Exclude<KernelType, 'custom'>, { name: string; kernel: number[][] }> = {
  identity: {
    name: 'Identity',
    kernel: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
  },
  'edge-detect': {
    name: 'Edge Detection',
    kernel: [
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1],
    ],
  },
  sharpen: {
    name: 'Sharpen',
    kernel: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
  blur: {
    name: 'Box Blur',
    kernel: [
      [1 / 9, 1 / 9, 1 / 9],
      [1 / 9, 1 / 9, 1 / 9],
      [1 / 9, 1 / 9, 1 / 9],
    ],
  },
  emboss: {
    name: 'Emboss',
    kernel: [
      [-2, -1, 0],
      [-1, 1, 1],
      [0, 1, 2],
    ],
  },
};

// Seeded PRNG
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Generate a simple test image (8x8 grid with patterns)
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

// Apply 2D convolution with zero-padding
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

export default function ConvolutionDemo({ id }: ConvolutionDemoProps) {
  const [kernelType, setKernelType] = useState<KernelType>('edge-detect');
  const [pattern, setPattern] = useState('cross');
  const [imageSize, setImageSize] = useState(10);
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

  // Heatmap trace for input
  const inputTrace: any = {
    type: 'heatmap' as const,
    z: [...inputImage].reverse(),
    colorscale: 'Greys',
    showscale: false,
    hovertemplate: 'row: %{y}<br>col: %{x}<br>value: %{z:.2f}<extra>Input</extra>',
  };

  // Heatmap trace for kernel
  const kernelTrace: any = {
    type: 'heatmap' as const,
    z: [...kernel].reverse(),
    colorscale: [
      [0, '#1a1a2e'],
      [0.5, '#16213e'],
      [1, '#e94560'],
    ],
    showscale: false,
    hovertemplate: 'row: %{y}<br>col: %{x}<br>weight: %{z:.3f}<extra>Kernel</extra>',
  };

  // Heatmap trace for output
  const outputTrace: any = {
    type: 'heatmap' as const,
    z: [...outputImage].reverse(),
    colorscale: 'Viridis',
    showscale: true,
    colorbar: { tickfont: { color: '#ccc' } },
    hovertemplate: 'row: %{y}<br>col: %{x}<br>value: %{z:.2f}<extra>Output</extra>',
  };

  const commonLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(15,15,25,1)',
    font: { color: '#ccc', size: 11 },
    margin: { t: 35, b: 30, l: 30, r: 30 },
    xaxis: { showticklabels: false, showgrid: false },
    yaxis: { showticklabels: false, showgrid: false, scaleanchor: 'x' as const },
  };

  const updateCustomKernel = (row: number, col: number, value: string) => {
    const newKernel = customKernel.map((r) => [...r]);
    newKernel[row][col] = parseFloat(value) || 0;
    setCustomKernel(newKernel);
  };

  return (
    <div className="w-full bg-[#151525] rounded-lg p-6 mb-8">
      <h3 className="text-xl font-semibold mb-4 text-white">
        2D Convolution Visualization
      </h3>

      {/* Controls Row */}
      <div className="flex flex-wrap gap-4 mb-6">
        <div>
          <label className="block text-sm text-gray-400 mb-1">Kernel</label>
          <select
            value={kernelType}
            onChange={(e) => setKernelType(e.target.value as KernelType)}
            className="bg-[#0a0a15] text-gray-200 border border-gray-700 rounded px-3 py-1.5 text-sm"
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
          <label className="block text-sm text-gray-400 mb-1">Input Pattern</label>
          <select
            value={pattern}
            onChange={(e) => setPattern(e.target.value)}
            className="bg-[#0a0a15] text-gray-200 border border-gray-700 rounded px-3 py-1.5 text-sm"
          >
            <option value="cross">Cross</option>
            <option value="checkerboard">Checkerboard</option>
            <option value="gradient">Gradient</option>
            <option value="diagonal">Diagonal</option>
            <option value="random">Random</option>
          </select>
        </div>
        <div>
          <label className="block text-sm text-gray-400 mb-1">
            Grid Size: {imageSize}x{imageSize}
          </label>
          <input
            type="range"
            min={5}
            max={16}
            value={imageSize}
            onChange={(e) => setImageSize(parseInt(e.target.value))}
            className="w-32 accent-blue-500"
          />
        </div>
      </div>

      {/* Custom kernel editor */}
      {kernelType === 'custom' && (
        <div className="mb-4 p-3 bg-[#0a0a15] rounded">
          <p className="text-sm text-gray-400 mb-2">Edit 3x3 kernel values:</p>
          <div className="grid grid-cols-3 gap-1 max-w-xs">
            {customKernel.map((row, i) =>
              row.map((val, j) => (
                <input
                  key={`${i}-${j}`}
                  type="number"
                  step="0.1"
                  value={val}
                  onChange={(e) => updateCustomKernel(i, j, e.target.value)}
                  className="bg-[#151525] text-gray-200 border border-gray-700 rounded px-2 py-1 text-sm text-center w-20"
                />
              ))
            )}
          </div>
        </div>
      )}

      {/* Three-panel visualization */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
        <div>
          <p className="text-center text-sm text-gray-400 mb-2 font-semibold">Input Image</p>
          <Plot
            data={[inputTrace]}
            layout={{
              ...commonLayout,
              title: { text: '' },
            }}
            useResizeHandler
            style={{ width: '100%', height: '300px' }}
            config={{ displayModeBar: false }}
          />
        </div>

        <div>
          <p className="text-center text-sm text-gray-400 mb-2 font-semibold">
            Kernel ({kernelType === 'custom' ? 'Custom' : PRESET_KERNELS[kernelType as Exclude<KernelType, 'custom'>].name})
          </p>
          <Plot
            data={[kernelTrace]}
            layout={{
              ...commonLayout,
              title: { text: '' },
            }}
            useResizeHandler
            style={{ width: '100%', height: '300px' }}
            config={{ displayModeBar: false }}
          />
          {/* Kernel values table */}
          <div className="mt-2 text-center">
            <table className="mx-auto text-xs text-gray-300">
              <tbody>
                {kernel.map((row, i) => (
                  <tr key={i}>
                    {row.map((val, j) => (
                      <td
                        key={j}
                        className="px-2 py-1 border border-gray-700"
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
          <p className="text-center text-sm text-gray-400 mb-2 font-semibold">Output (Convolved)</p>
          <Plot
            data={[outputTrace]}
            layout={{
              ...commonLayout,
              title: { text: '' },
            }}
            useResizeHandler
            style={{ width: '100%', height: '300px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>

      {/* Explanation */}
      <div className="mt-4 p-3 bg-[#0a0a15] rounded text-sm text-gray-400">
        The convolution operation slides the kernel across the input image. At each position, the kernel
        values are multiplied element-wise with the overlapping input values, and the results are summed
        to produce one output pixel. Zero-padding is used at the borders to maintain the output size.
      </div>
    </div>
  );
}
