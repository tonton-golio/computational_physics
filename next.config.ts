import type { NextConfig } from "next";
import createMDX from "@next/mdx";
const remarkMath = require("remark-math");
const rehypeKatex = require("rehype-katex");

const withMDX = createMDX({
  // Add markdown plugins here as needed
  extension: /\.mdx?$/,
});

const nextConfig: NextConfig = {
  // Enable MDX file support
  pageExtensions: ["js", "jsx", "ts", "tsx", "md", "mdx"],
  
  // Image optimization
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },
  
  // Experimental features
  experimental: {
    optimizePackageImports: ["katex", "@mdx-js/react"],
  },
};

export default withMDX(nextConfig);
