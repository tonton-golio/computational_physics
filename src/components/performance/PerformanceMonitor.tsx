'use client';

import { useEffect, useState } from 'react';

interface PerformanceMetrics {
  FCP?: number;
  LCP?: number;
  FID?: number;
  CLS?: number;
  TBT?: number;
  TTFB?: number;
  simulationFirstRenderMs?: number;
}

interface PerformanceMonitorProps {
  onMetricsUpdate?: (metrics: PerformanceMetrics) => void;
  showDebugInfo?: boolean;
}

export function PerformanceMonitor({ onMetricsUpdate, showDebugInfo = false }: PerformanceMonitorProps) {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({});
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Only run on client
    if (typeof window === 'undefined') return;

    let observer: PerformanceObserver | null = null;

    const updateMetrics = (newMetrics: Partial<PerformanceMetrics>) => {
      setMetrics(prev => {
        const updated = { ...prev, ...newMetrics };
        onMetricsUpdate?.(updated);
        return updated;
      });
    };

    // Web Vitals tracking
    const loadWebVitals = async () => {
      try {
        const { onFCP, onLCP, onINP, onCLS, onTTFB } = await import('web-vitals');

        onFCP((metric) => {
          updateMetrics({ FCP: metric.value });
        });

        onLCP((metric) => {
          updateMetrics({ LCP: metric.value });
        });

        // Keep `FID` field name for backwards compatibility in this monitor.
        onINP((metric) => {
          updateMetrics({ FID: metric.value });
        });

        onCLS((metric) => {
          updateMetrics({ CLS: metric.value });
        });

        onTTFB((metric) => {
          updateMetrics({ TTFB: metric.value });
        });

        setIsLoaded(true);
      } catch (error) {
        console.warn('Web Vitals library not available:', error);
      }
    };

    // Total Blocking Time calculation
    const calculateTBT = () => {
      if (!performance.timing) return null;

      const observer = new PerformanceObserver((list) => {
        let totalBlockingTime = 0;
        for (const entry of list.getEntries()) {
          if ((entry as any).duration > 50) {
            totalBlockingTime += (entry as any).duration - 50;
          }
        }
        updateMetrics({ TBT: totalBlockingTime });
      });

      observer.observe({ entryTypes: ['longtask'] });

      // Fallback TBT calculation after page load
      setTimeout(() => {
        if (!metrics.TBT) {
          const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
          const loadTime = navigation.loadEventEnd - navigation.fetchStart;
          // Rough TBT estimation
          const estimatedTBT = Math.max(0, loadTime - 100);
          updateMetrics({ TBT: estimatedTBT });
        }
      }, 1000);

      return observer;
    };

    // Initialize performance monitoring
    loadWebVitals();
    observer = calculateTBT();
    const onSimulationFirstRender = (event: Event) => {
      const customEvent = event as CustomEvent<{ durationMs?: number }>;
      const durationMs = customEvent.detail?.durationMs;
      if (typeof durationMs !== 'number') return;
      updateMetrics({ simulationFirstRenderMs: durationMs });
    };
    window.addEventListener('simulation-first-render', onSimulationFirstRender as EventListener);

    return () => {
      observer?.disconnect();
      window.removeEventListener('simulation-first-render', onSimulationFirstRender as EventListener);
    };
  }, [onMetricsUpdate, metrics.TBT]);

  // Performance status indicators
  const getStatusColor = (value: number | undefined, thresholds: { good: number; poor: number }) => {
    if (value === undefined) return 'text-[var(--text-soft)]';
    if (value <= thresholds.good) return 'text-green-400';
    if (value <= thresholds.poor) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStatusText = (value: number | undefined, thresholds: { good: number; poor: number }) => {
    if (value === undefined) return 'Loading...';
    if (value <= thresholds.good) return 'Good';
    if (value <= thresholds.poor) return 'Needs Work';
    return 'Poor';
  };

  if (!showDebugInfo) return null;

  return (
    <div className="fixed bottom-4 right-4 bg-[var(--surface-1)] border border-[var(--border-strong)] rounded-lg p-4 max-w-sm z-50 text-xs font-mono">
      <h3 className="text-[var(--text-strong)] font-semibold mb-2">Performance Monitor</h3>

      <div className="space-y-1">
        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">FCP:</span>
          <span className={getStatusColor(metrics.FCP, { good: 1800, poor: 3000 })}>
            {metrics.FCP ? `${metrics.FCP.toFixed(0)}ms` : '---'} ({getStatusText(metrics.FCP, { good: 1800, poor: 3000 })})
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">LCP:</span>
          <span className={getStatusColor(metrics.LCP, { good: 2500, poor: 4000 })}>
            {metrics.LCP ? `${metrics.LCP.toFixed(0)}ms` : '---'} ({getStatusText(metrics.LCP, { good: 2500, poor: 4000 })})
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">FID:</span>
          <span className={getStatusColor(metrics.FID, { good: 100, poor: 300 })}>
            {metrics.FID ? `${metrics.FID.toFixed(0)}ms` : '---'} ({getStatusText(metrics.FID, { good: 100, poor: 300 })})
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">CLS:</span>
          <span className={getStatusColor(metrics.CLS, { good: 0.1, poor: 0.25 })}>
            {metrics.CLS ? metrics.CLS.toFixed(3) : '---'} ({getStatusText(metrics.CLS, { good: 0.1, poor: 0.25 })})
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">TBT:</span>
          <span className={getStatusColor(metrics.TBT, { good: 200, poor: 600 })}>
            {metrics.TBT ? `${metrics.TBT.toFixed(0)}ms` : '---'} ({getStatusText(metrics.TBT, { good: 200, poor: 600 })})
          </span>
        </div>

        <div className="flex justify-between">
          <span className="text-[var(--text-muted)]">Sim First Render:</span>
          <span className={getStatusColor(metrics.simulationFirstRenderMs, { good: 350, poor: 800 })}>
            {metrics.simulationFirstRenderMs ? `${metrics.simulationFirstRenderMs.toFixed(0)}ms` : '---'} ({getStatusText(metrics.simulationFirstRenderMs, { good: 350, poor: 800 })})
          </span>
        </div>
      </div>

      {isLoaded && (
        <div className="mt-2 pt-2 border-t border-[var(--border-strong)]">
          <div className="text-green-400 text-xs">✅ Web Vitals loaded</div>
        </div>
      )}
    </div>
  );
}

// Hook for programmatic access to performance metrics
export function usePerformanceMetrics() {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({});

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const loadWebVitals = async () => {
      try {
        const { onFCP, onLCP, onINP, onCLS, onTTFB } = await import('web-vitals');

        onFCP((metric) => setMetrics(prev => ({ ...prev, FCP: metric.value })));
        onLCP((metric) => setMetrics(prev => ({ ...prev, LCP: metric.value })));
        onINP((metric) => setMetrics(prev => ({ ...prev, FID: metric.value })));
        onCLS((metric) => setMetrics(prev => ({ ...prev, CLS: metric.value })));
        onTTFB((metric) => setMetrics(prev => ({ ...prev, TTFB: metric.value })));
      } catch (error) {
        console.warn('Performance monitoring not available:', error);
      }
    };

    loadWebVitals();
    const onSimulationFirstRender = (event: Event) => {
      const customEvent = event as CustomEvent<{ durationMs?: number }>;
      const durationMs = customEvent.detail?.durationMs;
      if (typeof durationMs !== 'number') return;
      setMetrics((prev) => ({ ...prev, simulationFirstRenderMs: durationMs }));
    };
    window.addEventListener('simulation-first-render', onSimulationFirstRender as EventListener);

    return () => {
      window.removeEventListener('simulation-first-render', onSimulationFirstRender as EventListener);
    };
  }, []);

  return metrics;
}