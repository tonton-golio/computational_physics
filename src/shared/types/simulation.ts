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
