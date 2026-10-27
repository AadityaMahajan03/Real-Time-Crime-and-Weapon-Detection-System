import React from "react";
import { cn } from "@/lib/utils";

interface LiveBadgeProps {
  className?: string;
  label?: string;
  active?: boolean;
}

export function LiveBadge({ className, label = "LIVE", active = true }: LiveBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-bold tracking-wider uppercase backdrop-blur-md",
        active
          ? "border-neon-cyan/40 bg-neon-cyan/10 text-neon-cyan shadow-[0_0_12px_rgba(34,211,238,0.35)]"
          : "border-slate-700 bg-slate-800/40 text-slate-400",
        className
      )}
    >
      <span className="relative flex h-2 w-2">
        {active && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-neon-cyan opacity-75" />
        )}
        <span
          className={cn(
            "relative inline-flex h-2 w-2 rounded-full",
            active ? "bg-neon-cyan" : "bg-slate-500"
          )}
        />
      </span>
      <span>{label}</span>
    </span>
  );
}
