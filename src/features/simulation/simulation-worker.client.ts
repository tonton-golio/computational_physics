"use client";

import type {
  LotkaVolterraParams,
  LotkaVolterraResult,
  MDPSimulationParams,
  MDPSimulationResult,
} from "@/shared/types/simulation";

export type { LotkaVolterraParams, LotkaVolterraResult, MDPSimulationParams, MDPSimulationResult };

type CacheEntry<T> = {
  key: string;
  value: T;
};

const CACHE_VERSION = "simulation-worker-cache-v1";
const MAX_CACHE_ENTRIES = 120;
const lotkaResultCache = new Map<string, CacheEntry<LotkaVolterraResult>>();
const mdpResultCache = new Map<string, CacheEntry<MDPSimulationResult>>();

function stableStringifyRecord(record: Record<string, number>): string {
  const sortedKeys = Object.keys(record).sort();
  const stable: Record<string, number> = {};
  for (const key of sortedKeys) {
    stable[key] = record[key];
  }
  return JSON.stringify(stable);
}

function makeCacheKey(prefix: string, payload: Record<string, number>): string {
  return `${CACHE_VERSION}:${prefix}:${stableStringifyRecord(payload)}`;
}

function getCached<T>(cache: Map<string, CacheEntry<T>>, key: string): T | null {
  const cached = cache.get(key);
  if (!cached) return null;
  cache.delete(key);
  cache.set(key, cached);
  return cached.value;
}

function setCached<T>(cache: Map<string, CacheEntry<T>>, key: string, value: T): void {
  if (cache.has(key)) {
    cache.delete(key);
  }
  cache.set(key, { key, value });
  if (cache.size <= MAX_CACHE_ENTRIES) return;
  const oldestKey = cache.keys().next().value;
  if (!oldestKey) return;
  cache.delete(oldestKey);
}

export function runLotkaVolterraWorker(params: LotkaVolterraParams): Promise<LotkaVolterraResult> {
  const cacheKey = makeCacheKey("lotka", params);
  const cached = getCached(lotkaResultCache, cacheKey);
  if (cached) {
    return Promise.resolve(cached);
  }

  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("../../workers/simulation/lotka-volterra.worker.ts", import.meta.url));

    worker.onmessage = (event: MessageEvent<LotkaVolterraResult>) => {
      const result = event.data;
      setCached(lotkaResultCache, cacheKey, result);
      resolve(result);
      worker.terminate();
    };

    worker.onerror = (error) => {
      reject(error);
      worker.terminate();
    };

    worker.postMessage(params);
  });
}

export function runMDPSimulationWorker(params: MDPSimulationParams): Promise<MDPSimulationResult> {
  const cacheKey = makeCacheKey("mdp", params);
  const cached = getCached(mdpResultCache, cacheKey);
  if (cached) {
    return Promise.resolve(cached);
  }

  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("../../workers/simulation/mdp-simulation.worker.ts", import.meta.url));

    worker.onmessage = (event: MessageEvent<MDPSimulationResult>) => {
      const result = event.data;
      setCached(mdpResultCache, cacheKey, result);
      resolve(result);
      worker.terminate();
    };

    worker.onerror = (error) => {
      reject(error);
      worker.terminate();
    };

    worker.postMessage(params);
  });
}
