"use client";

import dynamic from "next/dynamic";
import type { AntigravityProps } from "./Antigravity";

const Antigravity = dynamic(() => import("./Antigravity"), { ssr: false });

export default function AntigravityClient(props: AntigravityProps) {
  return <Antigravity {...props} />;
}
