type MDPSimulationParams = {
  costPerUnfilled: number;
  setupCost: number;
  maxUnfilled: number;
  alpha: number;
  timeHorizon: number;
};

type MDPSimulationResult = {
  probFills: number[];
  avgCosts: number[];
  minCosts: number[];
  maxCosts: number[];
  optIdx: number;
};

function mulberry32(a: number) {
  return function rng() {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function runMDPSimulation({
  costPerUnfilled,
  setupCost,
  maxUnfilled,
  alpha,
  timeHorizon,
}: MDPSimulationParams): MDPSimulationResult {
  const nProbFill = 30;
  const nTrials = 20;
  const probFills: number[] = [];
  const avgCosts: number[] = [];
  const minCosts: number[] = [];
  const maxCosts: number[] = [];

  for (let i = 1; i <= nProbFill; i += 1) {
    probFills.push(Number((i * 0.03).toFixed(3)));
  }

  for (const p of probFills) {
    let totalCostSum = 0;
    let trialMin = Number.POSITIVE_INFINITY;
    let trialMax = Number.NEGATIVE_INFINITY;

    for (let trial = 0; trial < nTrials; trial += 1) {
      const rng = mulberry32(trial * 1000 + Math.round(p * 10000));
      let totalCost = 0;
      let unfilled = 0;

      for (let t = 0; t < timeHorizon; t += 1) {
        if (rng() < alpha) {
          unfilled += 1;
          if (unfilled > maxUnfilled) {
            unfilled = maxUnfilled;
          }
        }

        if (unfilled > 0 && rng() < p) {
          unfilled = 0;
          totalCost += setupCost;
        } else {
          totalCost += unfilled * costPerUnfilled;
        }
      }

      totalCostSum += totalCost;
      trialMin = Math.min(trialMin, totalCost);
      trialMax = Math.max(trialMax, totalCost);
    }

    avgCosts.push(totalCostSum / nTrials);
    minCosts.push(trialMin);
    maxCosts.push(trialMax);
  }

  let optIdx = 0;
  for (let i = 1; i < avgCosts.length; i += 1) {
    if (avgCosts[i] < avgCosts[optIdx]) {
      optIdx = i;
    }
  }

  return { probFills, avgCosts, minCosts, maxCosts, optIdx };
}

self.onmessage = (event: MessageEvent<MDPSimulationParams>) => {
  const result = runMDPSimulation(event.data);
  self.postMessage(result);
};
