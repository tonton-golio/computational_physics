import type { LotkaVolterraParams, LotkaVolterraResult } from "@/shared/types/simulation";

function solveLotkaVolterra({
  alpha,
  beta,
  gamma,
  delta,
  x0,
  y0,
  dt,
  steps,
}: LotkaVolterraParams): LotkaVolterraResult {
  const t: number[] = [];
  const x: number[] = [];
  const y: number[] = [];

  let currentX = x0;
  let currentY = y0;
  let currentT = 0;

  for (let i = 0; i < steps; i += 1) {
    t.push(currentT);
    x.push(currentX);
    y.push(currentY);

    const dx = (alpha * currentX - beta * currentX * currentY) * dt;
    const dy = (-gamma * currentY + delta * currentX * currentY) * dt;

    currentX += dx;
    currentY += dy;
    currentT += dt;
  }

  return { t, x, y };
}

self.onmessage = (event: MessageEvent<LotkaVolterraParams>) => {
  const result = solveLotkaVolterra(event.data);
  self.postMessage(result);
};
