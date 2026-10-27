"use client";

import React from "react";
import { motion, HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";

interface GlassCardProps extends HTMLMotionProps<"div"> {
  children: React.ReactNode;
  className?: string;
  glow?: "cyan" | "green" | "amber" | "red" | "violet" | "none";
  hoverEffect?: boolean;
}

export const GlassCard = React.forwardRef<HTMLDivElement, GlassCardProps>(
  ({ children, className, glow = "none", hoverEffect = true, ...props }, ref) => {
    const glowClasses = {
      none: "",
      cyan: "border-neon-cyan/20 hover:border-neon-cyan/40 hover:shadow-glow-cyan",
      green: "border-neon-green/20 hover:border-neon-green/40 hover:shadow-glow-green",
      amber: "border-neon-amber/20 hover:border-neon-amber/40 hover:shadow-glow-amber",
      red: "border-neon-red/25 hover:border-neon-red/50 hover:shadow-glow-red",
      violet: "border-neon-violet/20 hover:border-neon-violet/40 hover:shadow-glow-violet",
    };

    return (
      <motion.div
        ref={ref}
        whileHover={hoverEffect ? { y: -2, transition: { duration: 0.2 } } : undefined}
        className={cn(
          "rounded-2xl border border-white/10 bg-white/[0.035] backdrop-blur-xl p-6 shadow-glass",
          "transition-colors duration-200",
          hoverEffect && "hover:border-white/20 hover:bg-white/[0.05]",
          glowClasses[glow],
          className
        )}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);

GlassCard.displayName = "GlassCard";
