'use client';

import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

import { useState, useEffect } from 'react';

export default function ScientificComputing() {
  const [a, setA] = useState(-2);
  const [b, setB] = useState(2);
  const [c, setC] = useState(1);
  const [iterations, setIterations] = useState([]);
  const [running, setRunning] = useState(false);

  const f = (x) => x * x - c;

  const bisection = (a, b, tol = 1e-6, maxIter = 20) => {
    let steps = [];
    let fa = f(a), fb = f(b);
    if (fa * fb > 0) return steps; // no root
    for (let i = 0; i < maxIter; i++) {
      let c = (a + b) / 2;
      let fc = f(c);
      steps.push({ step: i+1, a, b, c, fc });
      if (Math.abs(fc) < tol || Math.abs(b - a) < tol) break;
      if (fa * fc < 0) {
        b = c;
        fb = fc;
      } else {
        a = c;
        fa = fc;
      }
    }
    return steps;
  };

  useEffect(() => {
    setIterations(bisection(a, b));
  }, [a, b, c]);

  const xMin = Math.min(a, b) - 1;
  const xMax = Math.max(a, b) + 1;
  const xVals = Array.from({length: 200}, (_, i) => xMin + (xMax - xMin) * i / 199);
  const yVals = xVals.map(f);

  const data = [
    {
      x: xVals,
      y: yVals,
      type: 'scatter',
      mode: 'lines',
      name: 'f(x) = x² - c'
    }
  ];

  // Add vertical lines for current a and b
  if (iterations.length > 0) {
    const last = iterations[iterations.length - 1];
    data.push({
      x: [last.a, last.a],
      y: [f(last.a), 0],
      type: 'scatter',
      mode: 'lines',
      line: {color: 'red', width: 2},
      name: 'a'
    });
    data.push({
      x: [last.b, last.b],
      y: [f(last.b), 0],
      type: 'scatter',
      mode: 'lines',
      line: {color: 'red', width: 2},
      name: 'b'
    });
  }

  // Add points for all c's
  iterations.forEach((it, idx) => {
    data.push({
      x: [it.c],
      y: [it.fc],
      type: 'scatter',
      mode: 'markers',
      marker: {color: idx % 2 === 0 ? 'green' : 'blue', size: 6},
      name: `c${idx+1}`
    });
  });

  const layout = {
    title: 'Bisection Method: f(x) = x² - c',
    xaxis: {title: 'x'},
    yaxis: {title: 'f(x)'},
    showlegend: false
  };

  return (
    <div style={{padding: '20px'}}>
      <h1>Numerical Root Finding: Bisection Method</h1>
      <p>The bisection method finds a root of f(x) = 0 by repeatedly bisecting an interval [a, b] where f(a) and f(b) have opposite signs.</p>
      <div style={{marginBottom: '20px'}}>
        <label>Parameter c: <input type="range" min="0.1" max="10" step="0.1" value={c} onChange={e => setC(parseFloat(e.target.value))} /> {c.toFixed(1)}</label>
      </div>
      <div style={{marginBottom: '20px'}}>
        <label>Interval a: <input type="range" min="-5" max="0" step="0.1" value={a} onChange={e => setA(parseFloat(e.target.value))} /> {a.toFixed(1)}</label>
      </div>
      <div style={{marginBottom: '20px'}}>
        <label>Interval b: <input type="range" min="0" max="5" step="0.1" value={b} onChange={e => setB(parseFloat(e.target.value))} /> {b.toFixed(1)}</label>
      </div>
      <Plot data={data} layout={layout} style={{width: '100%', height: '500px'}} />
      <h2>Iterations</h2>
      <table style={{borderCollapse: 'collapse', width: '100%'}}>
        <thead>
          <tr style={{border: '1px solid black'}}>
            <th>Step</th><th>a</th><th>b</th><th>c</th><th>f(c)</th><th>Error</th>
          </tr>
        </thead>
        <tbody>
          {iterations.map((it, idx) => {
            const error = Math.abs(it.b - it.a);
            return (
              <tr key={it.step} style={{border: '1px solid black'}}>
                <td>{it.step}</td><td>{it.a.toFixed(6)}</td><td>{it.b.toFixed(6)}</td><td>{it.c.toFixed(6)}</td><td>{it.fc.toFixed(6)}</td><td>{error.toFixed(6)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p>The method converges linearly, with the error halving each iteration.</p>
    </div>
  );
}