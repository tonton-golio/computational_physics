"use client";

import { useReportWebVitals } from "next/web-vitals";

type WebVitalsMetric = {
  id: string;
  name: string;
  label: string;
  value: number;
  rating?: string;
  navigationType?: string;
  attribution?: unknown;
};

const REPORT_ENDPOINT = "/api/analytics/web-vitals";

function shouldReport(): boolean {
  if (typeof window === "undefined") return false;

  // Always allow explicit opt-in.
  if (process.env.NEXT_PUBLIC_ENABLE_WEB_VITALS === "1") return true;

  // Default: only report in production.
  return process.env.NODE_ENV === "production";
}

function sendMetric(metric: WebVitalsMetric): void {
  const body = JSON.stringify({
    ...metric,
    path: window.location.pathname,
    timestamp: Date.now(),
  });

  if (navigator.sendBeacon) {
    const blob = new Blob([body], { type: "application/json" });
    navigator.sendBeacon(REPORT_ENDPOINT, blob);
    return;
  }

  void fetch(REPORT_ENDPOINT, {
    method: "POST",
    body,
    keepalive: true,
    headers: {
      "content-type": "application/json",
    },
  });
}

export function WebVitalsReporter() {
  useReportWebVitals((metric) => {
    if (!shouldReport()) return;
    sendMetric(metric as WebVitalsMetric);
  });

  return null;
}
