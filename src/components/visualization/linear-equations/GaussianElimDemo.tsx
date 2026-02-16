import React, { useState, useEffect } from 'react';
import * as math from 'mathjs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Table, TableBody, TableCell, TableHead, TableRow } from '@/components/ui/table';

const GaussianElimDemo: React.FC = () => {
  // 2x2 + b
  const [a11, setA11] = useState(3);
  const [a12, setA12] = useState(2);
  const [a21, setA21] = useState(1);
  const [a22, setA22] = useState(4);
  const [b1, setB1] = useState(8);
  const [b2, setB2] = useState(7);

  const [step, setStep] = useState(0); // 0: initial, 1: after pivot1, 2: backsub
  const [currentAug, setCurrentAug] = useState<number[][]>([]);
  const [solution, setSolution] = useState([0, 0]);

  const reset = () => {
    setStep(0);
    updateAug();
  };

  const updateAug = () => {
    setCurrentAug([
      [a11, a12, b1],
      [a21, a22, b2]
    ]);
  };

  const nextStep = () => {
    const aug = [...currentAug];
    if (step === 0) {
      // Eliminate below pivot 0
      const pivot = aug[0][0];
      if (Math.abs(pivot) < 1e-10) return;
      const factor = aug[1][0] / pivot;
      for (let j = 0; j < 3; j++) {
        aug[1][j] -= factor * aug[0][j];
      }
      setCurrentAug(aug);
      setStep(1);
    } else if (step === 1) {
      // Back substitution
      let x2 = aug[1][2] / aug[1][1];
      let x1 = (aug[0][2] - aug[0][1] * x2) / aug[0][0];
      setSolution([x1, x2]);
      setStep(2);
    }
  };

  useEffect(() => {
    updateAug();
  }, [a11, a12, a21, a22, b1, b2]);

  useEffect(() => {
    if (step === 0) updateAug();
  }, [step]);

  const formatCell = (val: number, col: number) => {
    return col < 2 ? val.toFixed(2) : '= ' + val.toFixed(2);
  };

  return (
    <Card className="flex flex-col h-[500px] w-full">
      <CardHeader>
        <CardTitle>Interactive Gaussian Elimination (2x2)</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-4 p-4 overflow-auto">
        <div className="grid grid-cols-6 gap-2 text-sm">
          <div><Label>a11</Label><Slider min={0} max={5} step={0.1} value={[a11]} onValueChange={([v])=>setA11(v)}/><span>{a11.toFixed(1)}</span></div>
          <div><Label>a12</Label><Slider min={0} max={5} step={0.1} value={[a12]} onValueChange={([v])=>setA12(v)}/><span>{a12.toFixed(1)}</span></div>
          <div><Label>b1</Label><Slider min={0} max={10} step={0.1} value={[b1]} onValueChange={([v])=>setB1(v)}/><span>{b1.toFixed(1)}</span></div>
          <div><Label>a21</Label><Slider min={0} max={5} step={0.1} value={[a21]} onValueChange={([v])=>setA21(v)}/><span>{a21.toFixed(1)}</span></div>
          <div><Label>a22</Label><Slider min={0} max={5} step={0.1} value={[a22]} onValueChange={([v])=>setA22(v)}/><span>{a22.toFixed(1)}</span></div>
          <div><Label>b2</Label><Slider min={0} max={10} step={0.1} value={[b2]} onValueChange={([v])=>setB2(v)}/><span>{b2.toFixed(1)}</span></div>
        </div>
        <div className="flex gap-2">
          <Button onClick={reset}>Reset</Button>
          <Button onClick={nextStep} disabled={step >= 2}>Next Step</Button>
        </div>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell></TableCell>
              <TableCell>x1</TableCell>
              <TableCell>x2</TableCell>
              <TableCell>| b</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {currentAug.map((row, i) => (
              <TableRow key={i}>
                <TableCell>R{i+1}</TableCell>
                {row.map((val, j) => (
                  <TableCell key={j} className={j < 2 ? 'font-mono' : 'font-bold'}>
                    {formatCell(val, j)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {step === 2 && (
          <div className="text-lg font-bold">
            Solution: x1 = {solution[0].toFixed(2)}, x2 = {solution[1].toFixed(2)}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default GaussianElimDemo;
