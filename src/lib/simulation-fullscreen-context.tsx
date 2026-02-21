"use client";

import { createContext, useContext } from "react";

const SimulationFullscreenContext = createContext(false);

export const SimulationFullscreenProvider = SimulationFullscreenContext.Provider;

export function useSimulationFullscreen() {
  return useContext(SimulationFullscreenContext);
}
