import React from "react";
import { IncidentStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

interface StatusChipProps {
  status: IncidentStatus;
  className?: string;
}

export function StatusChip({ status, className }: StatusChipProps) {
  const configs: Record<IncidentStatus, { label: string; dot: string; text: string; bg: string }> = {
    open: {
      label: "Open",
      dot: "bg-neon-red animate-pulse",
      text: "text-rose-300",
      bg: "bg-rose-500/10 border-rose-500/20",
    },
    in_review: {
      label: "In Review",
      dot: "bg-neon-amber animate-pulse",
      text: "text-amber-300",
      bg: "bg-amber-500/10 border-amber-500/20",
    },
    confirmed: {
      label: "Confirmed",
      dot: "bg-neon-cyan",
      text: "text-cyan-300",
      bg: "bg-cyan-500/10 border-cyan-500/20",
    },
    false_alarm: {
      label: "False Alarm",
      dot: "bg-slate-400",
      text: "text-slate-400",
      bg: "bg-slate-500/10 border-slate-500/20",
    },
    resolved: {
      label: "Resolved",
      dot: "bg-neon-green",
      text: "text-emerald-300",
      bg: "bg-emerald-500/10 border-emerald-500/20",
    },
    escalated: {
      label: "Escalated",
      dot: "bg-neon-red animate-ping",
      text: "text-rose-300 font-bold",
      bg: "bg-rose-500/20 border-rose-500/40 shadow-[0_0_10px_rgba(244,63,94,0.3)]",
    },
  };

  const current = configs[status] || configs.open;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium backdrop-blur-sm",
        current.bg,
        current.text,
        className
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full shrink-0", current.dot)} />
      <span>{current.label}</span>
    </span>
  );
}
