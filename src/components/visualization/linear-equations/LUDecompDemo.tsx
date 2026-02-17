import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Table, TableBody, TableCell, TableRow } from '@/components/ui/table';

const LUDecompDemo: React.FC = () => {
  const [a11, setA11] = useState(2);
  const [a12, setA12] = useState(1);
  const [a21, setA21] = useState(1);
  const [a22, setA22] = useState(3);

  const A = useMemo(() => [[a11, a12], [a21, a22]], [a11, a12, a21, a22]);

  const { L, U, verify } = useMemo(() => {
    const tol = 1e-10;
    if (Math.abs(a11) < tol) {
      return { L: [[1, 0], [0, 1]], U: [[0, 0], [0, 0]], verify: 0 };
    }

    const u11 = a11;
    const u12 = a12;
    const l21 = a21 / u11;
    const u22 = a22 - l21 * u12;

    const L = [[1, 0], [l21, 1]];
    const U = [[u11, u12], [0, u22]];

    const lu00 = L[0][0] * U[0][0] + L[0][1] * U[1][0];
    const lu01 = L[0][0] * U[0][1] + L[0][1] * U[1][1];
    const lu10 = L[1][0] * U[0][0] + L[1][1] * U[1][0];
    const lu11 = L[1][0] * U[0][1] + L[1][1] * U[1][1];

    const d00 = lu00 - A[0][0];
    const d01 = lu01 - A[0][1];
    const d10 = lu10 - A[1][0];
    const d11 = lu11 - A[1][1];
    const verify = Math.sqrt(d00 * d00 + d01 * d01 + d10 * d10 + d11 * d11);

    return { L, U, verify };
  }, [A, a11, a12, a21, a22]);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>LU Decomposition</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-3 gap-4 p-4">
        <div>
          <h3>A</h3>
          <Table>
            <TableBody>
              <TableRow><TableCell>{a11.toFixed(1)}</TableCell><TableCell>{a12.toFixed(1)}</TableCell></TableRow>
              <TableRow><TableCell>{a21.toFixed(1)}</TableCell><TableCell>{a22.toFixed(1)}</TableCell></TableRow>
            </TableBody>
          </Table>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <div><Label>a11</Label><Slider min={1} max={5} value={[a11]} onValueChange={([v])=>setA11(v)}/></div>
            <div><Label>a12</Label><Slider min={0} max={3} value={[a12]} onValueChange={([v])=>setA12(v)}/></div>
            <div><Label>a21</Label><Slider min={0} max={3} value={[a21]} onValueChange={([v])=>setA21(v)}/></div>
            <div><Label>a22</Label><Slider min={1} max={5} value={[a22]} onValueChange={([v])=>setA22(v)}/></div>
          </div>
        </div>
        <div>
          <h3>L</h3>
          <Table>
            <TableBody>
              <TableRow><TableCell>1</TableCell><TableCell>0</TableCell></TableRow>
              <TableRow><TableCell>{L[1]?.[0]?.toFixed(2) || 0}</TableCell><TableCell>1</TableCell></TableRow>
            </TableBody>
          </Table>
        </div>
        <div>
          <h3>U</h3>
          <Table>
            <TableBody>
              <TableRow><TableCell>{U[0]?.[0]?.toFixed(2) || 0}</TableCell><TableCell>{U[0]?.[1]?.toFixed(2) || 0}</TableCell></TableRow>
              <TableRow><TableCell>0</TableCell><TableCell>{U[1]?.[1]?.toFixed(2) || 0}</TableCell></TableRow>
            </TableBody>
          </Table>
        </div>
        <div className="text-sm">
          <p>||A - LU|| = {verify.toFixed(4)}</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default LUDecompDemo;
