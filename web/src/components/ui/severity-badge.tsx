import React from "react";
import { Severity } from "@/lib/types";
import { cn } from "@/lib/utils";
import { ShieldAlert, AlertTriangle, AlertCircle, Info, ShieldCheck } from "lucide-react";

interface SeverityBadgeProps {
  severity: Severity;
  showIcon?: boolean;
  className?: string;
}

export function SeverityBadge({ severity, showIcon = true, className }: SeverityBadgeProps) {
  const configs = {
    critical: {
      label: "Critical",
      icon: ShieldAlert,
      classes: "bg-neon-red/10 text-neon-red border-neon-red/30 shadow-[0_0_12px_-2px_rgba(244,63,94,0.3)]",
    },
    high: {
      label: "High",
      icon: AlertTriangle,
      classes: "bg-amber-500/10 text-amber-400 border-amber-500/30 shadow-[0_0_12px_-2px_rgba(245,158,11,0.25)]",
    },
    medium: {
      label: "Medium",
      icon: AlertCircle,
      classes: "bg-neon-amber/10 text-neon-amber border-neon-amber/30 shadow-[0_0_12px_-2px_rgba(251,191,36,0.25)]",
    },
    low: {
      label: "Low",
      icon: Info,
      classes: "bg-neon-cyan/10 text-neon-cyan border-neon-cyan/30 shadow-[0_0_12px_-2px_rgba(34,211,238,0.2)]",
    },
    safe: {
      label: "Safe",
      icon: ShieldCheck,
      classes: "bg-neon-green/10 text-neon-green border-neon-green/30 shadow-[0_0_12px_-2px_rgba(74,222,128,0.2)]",
    },
  };

  const current = configs[severity] || configs.low;
  const Icon = current.icon;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider backdrop-blur-md",
        current.classes,
        className
      )}
    >
      {showIcon && <Icon className="h-3 w-3 shrink-0" />}
      <span>{current.label}</span>
    </span>
  );
}
