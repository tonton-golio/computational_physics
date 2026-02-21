export type Matrix2x2 = [[number, number], [number, number]];

export function computeUnitEllipse(matrix: Matrix2x2): { circleX: number[]; circleY: number[]; ellipseX: number[]; ellipseY: number[] } {
  const nPoints = 50;
  const circleX: number[] = [];
  const circleY: number[] = [];
  const ellipseX: number[] = [];
  const ellipseY: number[] = [];

  for (let i = 0; i <= nPoints; i++) {
    const theta = (2 * Math.PI * i) / nPoints;
    const x = Math.cos(theta);
    const y = Math.sin(theta);
    circleX.push(x);
    circleY.push(y);
    ellipseX.push(matrix[0][0] * x + matrix[0][1] * y);
    ellipseY.push(matrix[1][0] * x + matrix[1][1] * y);
  }

  return { circleX, circleY, ellipseX, ellipseY };
}

export function matrixMultiply(A: Matrix2x2, B: Matrix2x2): Matrix2x2 {
  return [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
  ];
}

export function matrixAdd(A: Matrix2x2, B: Matrix2x2): Matrix2x2 {
  return [
    [A[0][0] + B[0][0], A[0][1] + B[0][1]],
    [A[1][0] + B[1][0], A[1][1] + B[1][1]]
  ];
}

export function matrixScale(A: Matrix2x2, s: number): Matrix2x2 {
  return [
    [A[0][0] * s, A[0][1] * s],
    [A[1][0] * s, A[1][1] * s]
  ];
}

export function matrixExp(A: Matrix2x2): Matrix2x2 {
  let result: Matrix2x2 = [[1, 0], [0, 1]]; // I
  let term: Matrix2x2 = [[1, 0], [0, 1]]; // I
  for (let k = 1; k <= 15; k++) { // More terms for accuracy
    term = matrixScale(matrixMultiply(term, A), 1 / k);
    result = matrixAdd(result, term);
  }
  return result;
}
