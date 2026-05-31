import type { NextConfig } from "next";
import { BundleAnalyzerPlugin } from "webpack-bundle-analyzer";

const nextConfig: NextConfig = {
  images: {
    // No remote images are consumed via next/image (only local /figures/* paths,
    // and the profile/leaderboard avatars use plain <img>). Avoid an open optimizer
    // proxy — add a specific hostname here if a remote next/image source is introduced.
    formats: ["image/webp", "image/avif"],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },
  compress: true,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          // X-XSS-Protection is deprecated and can introduce vulnerabilities in
          // legacy browsers; modern guidance is to rely on CSP/sniff protection.
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=()",
          },
          {
            key: "Strict-Transport-Security",
            value: "max-age=63072000; includeSubDomains; preload",
          },
        ],
      },
      {
        source: "/_next/static/:path*",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        source: "/figures/:path*",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=86400, s-maxage=86400, stale-while-revalidate=604800",
          },
        ],
      },
    ];
  },
  experimental: {
    optimizePackageImports: [
      "katex",
      "@react-three/fiber",
      "@react-three/drei",
    ],
    webVitalsAttribution: ["CLS", "LCP", "INP", "TTFB"],
  },
  turbopack: {},
  webpack: (config) => {
    if (process.env.ANALYZE === "true" && process.env.NODE_ENV === "production") {
      config.plugins.push(
        new BundleAnalyzerPlugin({
          analyzerMode: "static",
          reportFilename: "./analyze/client.html",
          openAnalyzer: false,
        })
      );
    }
    return config;
  },
  output: "standalone",
};

export default nextConfig;
