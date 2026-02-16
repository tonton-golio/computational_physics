'use client';

import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist';

interface GraphProps {
  type: 'harmonic' | 'wave' | 'gaussian' | 'custom';
  params?: Record<string, number>;
  title?: string;
}

export function InteractiveGraph({ type, params = {}, title }: GraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!containerRef.current) return;
    
    let data: Plotly.Data[] = [];
    const layout: Partial<Plotly.Layout> = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(15,15,25,1)',
      font: { color: '#9ca3af', family: 'system-ui' },
      margin: { t: 40, r: 20, b: 40, l: 50 },
      xaxis: { 
        gridcolor: '#1e1e2e',
        zerolinecolor: '#2d2d44',
      },
      yaxis: { 
        gridcolor: '#1e1e2e',
        zerolinecolor: '#2d2d44',
      },
    };
    
    switch (type) {
      case 'harmonic': {
        const A = params.amplitude || 1;
        const omega = params.frequency || 1;
        const phi = params.phase || 0;
        const x = Array.from({ length: 200 }, (_, i) => (i / 200) * 4 * Math.PI);
        const y = x.map(xi => A * Math.cos(omega * xi + phi));
        
        data = [{
          x,
          y,
          type: 'scatter',
          mode: 'lines',
          line: { color: '#3b82f6', width: 2 },
          name: 'x(t)',
        }];
        layout.xaxis = { ...layout.xaxis, title: { text: 'Time' } };
        layout.yaxis = { ...layout.yaxis, title: { text: 'Displacement' } };
        layout.title = { text: title || 'Simple Harmonic Motion' };
        break;
      }
      
      case 'wave': {
        const k = params.wavenumber || 1;
        const omega = params.frequency || 1;
        const x = Array.from({ length: 200 }, (_, i) => (i / 200) * 4 * Math.PI);
        
        // Two time snapshots
        const y1 = x.map(xi => Math.sin(k * xi - omega * 0));
        const y2 = x.map(xi => Math.sin(k * xi - omega * 1));
        
        data = [
          { x, y: y1, type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 2 }, name: 't = 0' },
          { x, y: y2, type: 'scatter', mode: 'lines', line: { color: '#ec4899', width: 2 }, name: 't = 1' },
        ];
        layout.xaxis = { ...layout.xaxis, title: { text: 'Position x' } };
        layout.yaxis = { ...layout.yaxis, title: { text: 'Amplitude' } };
        layout.title = { text: title || 'Wave Propagation' };
        break;
      }
      
      case 'gaussian': {
        const mu = params.mean || 0;
        const sigma = params.stddev || 1;
        const x = Array.from({ length: 200 }, (_, i) => mu - 4 * sigma + (i / 200) * 8 * sigma);
        const y = x.map(xi => (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((xi - mu) / sigma, 2)));
        
        data = [{
          x,
          y,
          type: 'scatter',
          mode: 'lines',
          fill: 'tozeroy',
          fillcolor: 'rgba(59, 130, 246, 0.2)',
          line: { color: '#3b82f6', width: 2 },
        }];
        layout.xaxis = { ...layout.xaxis, title: { text: 'x' } };
        layout.yaxis = { ...layout.yaxis, title: { text: 'P(x)' } };
        layout.title = { text: title || 'Gaussian Distribution' };
        break;
      }
    }
    
    Plotly.newPlot(containerRef.current, data, layout, {
      responsive: true,
      displayModeBar: false,
    });
    
    return () => {
      if (containerRef.current) {
        Plotly.purge(containerRef.current);
      }
    };
  }, [type, params, title]);
  
  return (
    <div 
      ref={containerRef} 
      className="w-full h-64 bg-[#151525] rounded-lg overflow-hidden"
    />
  );
}

// List of available graphs for content
export const GRAPH_DEFS: Record<string, GraphProps> = {
  'harmonic-motion': {
    type: 'harmonic',
    params: { amplitude: 1, frequency: 2, phase: 0 },
  },
  'wave-propagation': {
    type: 'wave',
    params: { wavenumber: 1, frequency: 1 },
  },
  'gaussian': {
    type: 'gaussian',
    params: { mean: 0, stddev: 1 },
  },
};
