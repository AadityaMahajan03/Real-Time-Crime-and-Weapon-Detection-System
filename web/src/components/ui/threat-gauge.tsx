"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface ThreatGaugeProps {
  score: number; // 0.0 to 1.0
  level: "CRITICAL" | "HIGH" | "ELEVATED" | "NORMAL" | "SAFE";
  className?: string;
}

export function ThreatGauge({ score, level, className }: ThreatGaugeProps) {
  // Semi-circle arc calculations:
  // Radius = 70, circumference for half circle = Math.PI * 70 = 219.9
  const radius = 70;
  const circumference = Math.PI * radius;
  const clamped = Math.max(0, Math.min(1, score));
  const strokeDashoffset = circumference - clamped * circumference;

  const colorConfig = {
    CRITICAL: {
      color: "#F43F5E",
      textColor: "text-neon-red",
      glow: "drop-shadow-[0_0_12px_rgba(244,63,94,0.6)]",
      bgBadge: "bg-neon-red/10 border-neon-red/30",
    },
    HIGH: {
      color: "#FB923C",
      textColor: "text-orange-400",
      glow: "drop-shadow-[0_0_12px_rgba(251,146,60,0.5)]",
      bgBadge: "bg-orange-500/10 border-orange-500/30",
    },
    ELEVATED: {
      color: "#FBBF24",
      textColor: "text-neon-amber",
      glow: "drop-shadow-[0_0_12px_rgba(251,191,36,0.5)]",
      bgBadge: "bg-neon-amber/10 border-neon-amber/30",
    },
    NORMAL: {
      color: "#22D3EE",
      textColor: "text-neon-cyan",
      glow: "drop-shadow-[0_0_12px_rgba(34,211,238,0.5)]",
      bgBadge: "bg-neon-cyan/10 border-neon-cyan/30",
    },
    SAFE: {
      color: "#4ADE80",
      textColor: "text-neon-green",
      glow: "drop-shadow-[0_0_12px_rgba(74,222,128,0.5)]",
      bgBadge: "bg-neon-green/10 border-neon-green/30",
    },
  };

  const current = colorConfig[level] || colorConfig.SAFE;

  return (
    <div className={cn("flex flex-col items-center justify-center", className)}>
      <div className="relative flex items-center justify-center">
        <svg
          viewBox="0 0 180 105"
          className="w-48 overflow-visible"
        >
          {/* Background Arc */}
          <path
            d="M 20 95 A 70 70 0 0 1 160 95"
            fill="none"
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth="12"
            strokeLinecap="round"
          />

          {/* Active Gradient/Glow Arc */}
          <path
            d="M 20 95 A 70 70 0 0 1 160 95"
            fill="none"
            stroke={current.color}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className={cn("transition-all duration-700 ease-out", current.glow)}
          />
        </svg>

        {/* Center Readout */}
        <div className="absolute bottom-1 text-center">
          <div className="font-mono text-3xl font-bold tracking-tight text-white">
            {Math.round(clamped * 100)}%
          </div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400">
            Threat Index
          </div>
        </div>
      </div>

      {/* Level Badge */}
      <div
        className={cn(
          "mt-2 inline-flex items-center rounded-full border px-3 py-0.5 text-xs font-bold tracking-widest uppercase",
          current.bgBadge,
          current.textColor
        )}
      >
        {level}
      </div>
    </div>
  );
}
