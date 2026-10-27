"use client";

import React, { useEffect, useState } from "react";
import { GlassCard } from "./glass-card";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface KpiCardProps {
  label: string;
  value: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  trend?: {
    value: string;
    isPositive?: boolean;
    isNeutral?: boolean;
  };
  icon: LucideIcon;
  glow?: "cyan" | "green" | "amber" | "red" | "violet";
  className?: string;
}

export function KpiCard({
  label,
  value,
  prefix = "",
  suffix = "",
  decimals = 0,
  trend,
  icon: Icon,
  glow = "cyan",
  className,
}: KpiCardProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    // Count up animation
    const duration = 800; // ms
    const startTime = performance.now();
    const startValue = 0;

    function step(now: number) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = startValue + (value - startValue) * easeOut;
      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        setDisplayValue(value);
      }
    }

    const animId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animId);
  }, [value]);

  const iconGlowStyles = {
    cyan: "text-neon-cyan bg-neon-cyan/10 border-neon-cyan/20",
    green: "text-neon-green bg-neon-green/10 border-neon-green/20",
    amber: "text-neon-amber bg-neon-amber/10 border-neon-amber/20",
    red: "text-neon-red bg-neon-red/10 border-neon-red/20",
    violet: "text-neon-violet bg-neon-violet/10 border-neon-violet/20",
  };

  return (
    <GlassCard glow={glow} className={cn("relative overflow-hidden", className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {label}
        </span>
        <div
          className={cn(
            "flex h-9 w-9 items-center justify-center rounded-xl border backdrop-blur-md",
            iconGlowStyles[glow]
          )}
        >
          <Icon className="h-4 w-4" />
        </div>
      </div>

      <div className="mt-4 flex items-baseline gap-1">
        <span className="font-mono text-3xl font-bold tracking-tight text-white md:text-4xl">
          {prefix}
          {decimals > 0 ? displayValue.toFixed(decimals) : Math.round(displayValue)}
          {suffix}
        </span>
      </div>

      {trend && (
        <div className="mt-2.5 flex items-center gap-1.5 text-xs">
          <span
            className={cn(
              "font-mono font-medium",
              trend.isNeutral
                ? "text-slate-400"
                : trend.isPositive
                ? "text-neon-green"
                : "text-neon-red"
            )}
          >
            {trend.value}
          </span>
          <span className="text-slate-500">vs baseline</span>
        </div>
      )}
    </GlassCard>
  );
}
