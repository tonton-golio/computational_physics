import React, { useState, useEffect, useMemo } from 'react';
import * as math from 'mathjs';
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
    try {
      const Am = math.matrix(A);
      const lu = math.lu(Am);
      const Lp = math.subset(lu.L, math.index(math.range(0,2), math.range(0,2)));
      const Up = math.subset(lu.U, math.index(math.range(0,2), math.range(0,2)));
      const LU = math.multiply(math.matrix(Lp), math.matrix(Up));
      const verify = math.norm(math.subtract(LU, Am), 'frobenius').toNumber();
      return { L: Lp.toArray(), U: Up.toArray(), verify };
    } catch (e) {
      console.error(e);
      return { L: [[0,0],[0,0]], U: [[0,0],[0,0]], verify: 0 };
    }
  }, [A]);

  useEffect(() => {
    try {
      const Am = math.matrix(A);
      const lu = math.lu(Am);
      const Lp = math.subset(lu.L, math.index(math.range(0,2), math.range(0,2)));
      const Up = math.subset(lu.U, math.index(math.range(0,2), math.range(0,2)));
      setL(Lp.toArray());
      setU(Up.toArray());

      const LU = math.multiply(math.matrix(Lp), math.matrix(Up));
      setVerify(math.norm(math.subtract(LU, Am), 'frobenius').toNumber());
    } catch (e) {
      console.error(e);
    }
  }, [A]);

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
