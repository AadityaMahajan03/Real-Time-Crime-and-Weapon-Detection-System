"use client";

import React from "react";
import { motion } from "framer-motion";
import { VideoPanel } from "@/components/live/video-panel";
import { ShieldCheck } from "lucide-react";

export default function LiveMonitorPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2"
      >
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
            Live Surveillance Wall
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Real-time multi-camera edge inference, neural bounding reticles, and acoustic alerts.
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-xl border border-neon-green/30 bg-neon-green/10 px-3 py-1.5 text-xs text-neon-green backdrop-blur-md self-start">
          <ShieldCheck className="h-4 w-4" />
          <span>Active Edge Perimeter Protection</span>
        </div>
      </motion.div>

      {/* Main Interactive Video Panel */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.1 }}
      >
        <VideoPanel />
      </motion.div>
    </div>
  );
}
