import React, { useState, useEffect, useMemo } from 'react';
import * as math from 'mathjs';
import Plotly from 'react-plotly.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

type Matrix2 = number[][];

type Props = Record<string, never>;

function Label({ children }: { children: React.ReactNode }) {
  return <label className="block text-sm text-gray-300">{children}</label>;
}

const ConditionNumberDemo: React.FC<Props> = () => {
  const [a11, setA11] = useState(1);
  const [a12, setA12] = useState(0);
  const [a21, setA21] = useState(0);
  const [a22, setA22] = useState(1);
  const [b1, setB1] = useState(1);
  const [b2, setB2] = useState(0);
  const [eps, setEps] = useState(0);
  const [perturbType, setPerturbType] = useState<'b' | 'A'>('b');

  const A = useMemo(() => [[a11, a12], [a21, a22]] as Matrix2, [a11, a12, a21, a22]);
  const b = useMemo(() => math.matrix([b1, b2]), [b1, b2]);

  const [x, setX] = useState([0, 0]);
  const [cond, setCond] = useState(1);
  const [xPert, setXPert] = useState([0, 0]);
  const [relError, setRelError] = useState(0);

  useEffect(() => {
    try {
      const Am = math.matrix(A);
      const invA = math.inv(Am);
      const normA = Number(math.norm(Am, 'inf'));
      const normInvA = Number(math.norm(invA, 'inf'));
      setCond(normA * normInvA);

      const xSol = math.multiply(invA, b);
      setX([xSol.get([0,0]), xSol.get([1,0])]);

      // Perturb
      let bPert: math.Matrix = b;
      let APert: math.Matrix = Am;
      if (perturbType === 'b') {
        const db = math.random([2,1], -eps, eps);
        bPert = math.add(b, db) as math.Matrix;
      } else {
        const dA = math.random([2,2], -eps, eps);
        APert = math.add(Am, dA) as math.Matrix;
      }
      const xPertSol = math.lusolve(APert, bPert);
      setXPert([xPertSol.get([0,0]), xPertSol.get([1,0])]);

      const dx = math.subtract(xPertSol, xSol);
      const relDx = Number(math.divide(math.norm(dx, 'inf'), math.norm(xSol || math.matrix([1,0]), 'inf')));
      const relDb = eps; // approx
      setRelError(relDx / relDb);
    } catch (e) {
      console.error(e);
    }
  }, [A, b, eps, perturbType]);

  const line1 = {
    x: [-10, 10],
    y: [(b1 - a11 * -10) / a12 || 0, (b1 - a11 * 10) / a12 || 0],
    mode: 'lines',
    name: 'Eq 1',
    line: { color: 'blue', width: 3 },
  };

  const line2 = {
    x: [-10, 10],
    y: [(b2 - a21 * -10) / a22 || 0, (b2 - a21 * 10) / a22 || 0],
    mode: 'lines',
    name: 'Eq 2',
    line: { color: 'red', width: 3 },
  };

  // Simplified plot, add perturb lines similarly

  return (
    <Card className="flex flex-col h-[600px] w-full">
      <CardHeader>
        <CardTitle>Condition Number Demo (2x2)</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-4 p-4">
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <Label>a11</Label>
            <input type="range" min={-2} max={2} step={0.1} value={a11} onChange={(e) => setA11(parseFloat(e.target.value))} />
            <span>{a11.toFixed(2)}</span>
          </div>
          {/* similar for a12, a21, a22, b1, b2 */}
          <div>
            <Label>a12</Label>
            <input type="range" min={-2} max={2} step={0.1} value={a12} onChange={(e) => setA12(parseFloat(e.target.value))} />
            <span>{a12.toFixed(2)}</span>
          </div>
          <div>
            <Label>a21</Label>
            <input type="range" min={-2} max={2} step={0.1} value={a21} onChange={(e) => setA21(parseFloat(e.target.value))} />
            <span>{a21.toFixed(2)}</span>
          </div>
          <div>
            <Label>a22</Label>
            <input type="range" min={-2} max={2} step={0.1} value={a22} onChange={(e) => setA22(parseFloat(e.target.value))} />
            <span>{a22.toFixed(2)}</span>
          </div>
          <div>
            <Label>b1</Label>
            <input type="range" min={-2} max={2} step={0.1} value={b1} onChange={(e) => setB1(parseFloat(e.target.value))} />
            <span>{b1.toFixed(2)}</span>
          </div>
          <div>
            <Label>b2</Label>
            <input type="range" min={-2} max={2} step={0.1} value={b2} onChange={(e) => setB2(parseFloat(e.target.value))} />
            <span>{b2.toFixed(2)}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant={perturbType === 'b' ? 'primary' : 'outline'} onClick={() => setPerturbType('b')}>Perturb b</Button>
          <Button variant={perturbType === 'A' ? 'primary' : 'outline'} onClick={() => setPerturbType('A')}>Perturb A</Button>
        </div>
        <div>
          <Label>Eps</Label>
          <input type="range" min={0} max={0.2} step={0.01} value={eps} onChange={(e) => setEps(parseFloat(e.target.value))} />
          <span>{eps.toFixed(3)}</span>
        </div>
        <div className="grid grid-cols-2 gap-4 text-lg font-mono">
          <div>x: [{x[0].toFixed(2)}, {x[1].toFixed(2)}] | cond ≈ {cond.toFixed(1)}</div>
          <div>x&#39;: [{xPert[0].toFixed(2)}, {xPert[1].toFixed(2)}] | rel err ≈ {relError.toFixed(1)}</div>
        </div>
        <div className="flex-1 min-h-0">
          <Plotly
            data={[line1, line2]}
            layout={{ width: 400, height: 400, title: { text: 'Equation Lines (x,y plane)' }, xaxis: {title: { text: 'x1' }}, yaxis: {title: { text: 'x2' }} }}
            config={{ staticPlot: false, displayModeBar: false }}
          />
        </div>
      </CardContent>
    </Card>
  );
};

export default ConditionNumberDemo;
