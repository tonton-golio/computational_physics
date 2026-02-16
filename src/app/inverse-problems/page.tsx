import React, { useState } from 'react';
import Plotly from 'react-plotly.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// Lotka-Volterra equations: dx/dt = alpha*x - beta*x*y, dy/dt = -gamma*y + delta*x*y

// True parameters for simulated data
const TRUE_ALPHA = 1.5;
const TRUE_BETA = 0.1;
const TRUE_GAMMA = 1.5;
const TRUE_DELTA = 0.075;

// Initial conditions
const X0 = 10;
const Y0 = 5;
const T0 = 0;
const TF = 50;
const DT = 0.1;

function solveLotkaVolterra(alpha: number, beta: number, gamma: number, delta: number, x0: number, y0: number, t0: number, tf: number, dt: number) {
  let t = t0;
  let x = x0;
  let y = y0;
  const times: number[] = [];
  const xs: number[] = [];
  const ys: number[] = [];
  while (t <= tf + dt) { // +dt to include tf
    times.push(t);
    xs.push(x);
    ys.push(y);
    const dxdt = alpha * x - beta * x * y;
    const dydt = -gamma * y + delta * x * y;
    x += dxdt * dt;
    y += dydt * dt;
    t += dt;
  }
  return { times, xs, ys };
}

// Simulate data with true parameters
const trueData = solveLotkaVolterra(TRUE_ALPHA, TRUE_BETA, TRUE_GAMMA, TRUE_DELTA, X0, Y0, T0, TF, DT);

export default function InverseProblemsPage() {
  const [alpha, setAlpha] = useState(1.0);
  const [beta, setBeta] = useState(0.1);
  const [gamma, setGamma] = useState(1.0);
  const [delta, setDelta] = useState(0.05);

  // Compute fit with current parameters
  const fitData = solveLotkaVolterra(alpha, beta, gamma, delta, X0, Y0, T0, TF, DT);

  // Compute residuals
  const resX = trueData.xs.map((d, i) => d - fitData.xs[i]);
  const resY = trueData.ys.map((d, i) => d - fitData.ys[i]);

  // Sum of squares for residuals
  const sumSqX = resX.reduce((sum, r) => sum + r * r, 0);
  const sumSqY = resY.reduce((sum, r) => sum + r * r, 0);
  const totalSumSq = sumSqX + sumSqY;

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <h1 className="text-3xl font-bold text-center mb-8">Inverse Problems: Parameter Estimation in ODEs</h1>
      <p className="text-center mb-8">
        This demonstration shows parameter estimation for the Lotka-Volterra (predator-prey) model using least squares.
        Simulated data is generated with true parameters (α={TRUE_ALPHA}, β={TRUE_BETA}, γ={TRUE_GAMMA}, δ={TRUE_DELTA}).
        Use the sliders to set initial guesses for the parameters, and observe how the fit and residuals change.
        In practice, nonlinear least squares optimization (e.g., Levenberg-Marquardt) would minimize the sum of squared residuals starting from these guesses.
      </p>

      <Card>
        <CardHeader>
          <CardTitle>Parameter Sliders (Initial Guesses)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label>α (prey growth): {alpha.toFixed(2)}</label>
              <input
                type="range"
                min={0.5}
                max={3}
                step={0.01}
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label>β (predation): {beta.toFixed(3)}</label>
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.001}
                value={beta}
                onChange={(e) => setBeta(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label>γ (predator death): {gamma.toFixed(2)}</label>
              <input
                type="range"
                min={0.5}
                max={3}
                step={0.01}
                value={gamma}
                onChange={(e) => setGamma(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label>δ (predator efficiency): {delta.toFixed(3)}</label>
              <input
                type="range"
                min={0.01}
                max={0.2}
                step={0.001}
                value={delta}
                onChange={(e) => setDelta(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
          <div className="mt-4">
            <p>Total Sum of Squared Residuals: {totalSumSq.toFixed(2)}</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Population Dynamics: Data vs Fit</CardTitle>
        </CardHeader>
        <CardContent>
          <Plotly
            data={[
              {
                x: trueData.times,
                y: trueData.xs,
                type: 'scatter',
                mode: 'markers',
                name: 'Data Prey (x)',
                marker: { color: 'blue' }
              },
              {
                x: trueData.times,
                y: trueData.ys,
                type: 'scatter',
                mode: 'markers',
                name: 'Data Predator (y)',
                marker: { color: 'red' }
              },
              {
                x: fitData.times,
                y: fitData.xs,
                type: 'scatter',
                mode: 'lines',
                name: 'Fit Prey (x)',
                line: { color: 'blue', dash: 'dash' }
              },
              {
                x: fitData.times,
                y: fitData.ys,
                type: 'scatter',
                mode: 'lines',
                name: 'Fit Predator (y)',
                line: { color: 'red', dash: 'dash' }
              }
            ]}
            layout={{
              title: 'Lotka-Volterra Populations',
              xaxis: { title: 'Time' },
              yaxis: { title: 'Population' },
              height: 500
            }}
            config={{ displayModeBar: false }}
          />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Residuals: Data - Fit</CardTitle>
        </CardHeader>
        <CardContent>
          <Plotly
            data={[
              {
                x: trueData.times,
                y: resX,
                type: 'scatter',
                mode: 'lines',
                name: 'Residual Prey (x)',
                line: { color: 'blue' }
              },
              {
                x: trueData.times,
                y: resY,
                type: 'scatter',
                mode: 'lines',
                name: 'Residual Predator (y)',
                line: { color: 'red' }
              }
            ]}
            layout={{
              title: 'Residuals over Time',
              xaxis: { title: 'Time' },
              yaxis: { title: 'Residual' },
              height: 400
            }}
            config={{ displayModeBar: false }}
          />
        </CardContent>
      </Card>
    </div>
  );
}