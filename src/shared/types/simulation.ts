import type { ComponentType } from "react";

export type SimulationComponentProps = {
  id?: string;
};

export type SimulationComponent = ComponentType<SimulationComponentProps>;

export interface SimulationDefinition {
  id: string;
  topic: string;
  title: string;
  load: () => Promise<SimulationComponent | null>;
  fallbackToGraph?: boolean;
  usesWorker?: boolean;
}

// ── Worker param/result types ───────────────────────────────────────────

export type LotkaVolterraParams = {
  alpha: number;
  beta: number;
  gamma: number;
  delta: number;
  x0: number;
  y0: number;
  dt: number;
  steps: number;
};

export type LotkaVolterraResult = {
  t: number[];
  x: number[];
  y: number[];
};

export type MDPSimulationParams = {
  costPerUnfilled: number;
  setupCost: number;
  maxUnfilled: number;
  alpha: number;
  timeHorizon: number;
};

export type MDPSimulationResult = {
  probFills: number[];
  avgCosts: number[];
  minCosts: number[];
  maxCosts: number[];
  optIdx: number;
};
