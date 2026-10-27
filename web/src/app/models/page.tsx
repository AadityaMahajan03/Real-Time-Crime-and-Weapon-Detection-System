"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Cpu,
  CheckCircle,
  BarChart3,
  Layers,
  Zap,
  Target,
  Clock,
  Sparkles,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { MOCK_AI_MODELS } from "@/lib/mock-data";
import { cn } from "@/lib/utils";

export default function AIModelsPage() {
  const [activeTab, setActiveTab] = useState<"cards" | "evaluation" | "architecture">("cards");

  return (
    <div className="space-y-6">
      {/* Title */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
      >
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
            AI Models & Neural Telemetry
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Model weights registry, held-out evaluation benchmarks, and spatiotemporal pipeline architecture.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center rounded-xl border border-white/10 bg-black/40 p-1 self-start">
          {[
            { id: "cards", label: "Model Cards", icon: Cpu },
            { id: "evaluation", label: "Evaluation", icon: BarChart3 },
            { id: "architecture", label: "Neural Pipeline", icon: Layers },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={cn(
                  "flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-medium transition-all duration-200",
                  isActive
                    ? "bg-white/10 text-neon-cyan shadow-sm font-semibold"
                    : "text-slate-400 hover:text-slate-200"
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </motion.div>

      {/* Tab 1: Model Cards */}
      {activeTab === "cards" && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-5"
        >
          {MOCK_AI_MODELS.map((model) => (
            <GlassCard key={model.id} glow={model.status === "active" ? "cyan" : "none"}>
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[10px] uppercase font-mono tracking-wider text-neon-cyan">
                    {model.category}
                  </span>
                  <h3 className="mt-0.5 text-base font-bold text-white">{model.name}</h3>
                  <p className="font-mono text-xs text-slate-400">{model.version}</p>
                </div>

                <span
                  className={cn(
                    "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                    model.status === "active"
                      ? "border-neon-green/30 bg-neon-green/10 text-neon-green"
                      : "border-slate-700 bg-slate-800/40 text-slate-400"
                  )}
                >
                  <span
                    className={cn(
                      "h-1.5 w-1.5 rounded-full",
                      model.status === "active" ? "bg-neon-green" : "bg-slate-500"
                    )}
                  />
                  <span>{model.status}</span>
                </span>
              </div>

              {/* Backbone details */}
              <div className="mt-3 rounded-xl border border-white/5 bg-white/[0.02] p-2.5 text-xs text-slate-300 font-mono">
                {model.backbone}
              </div>

              {/* Supported Classes */}
              <div className="mt-4">
                <span className="text-[10px] uppercase tracking-wider text-slate-400">
                  Target Classes
                </span>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {model.classes.map((cls) => (
                    <span
                      key={cls}
                      className="rounded-md border border-white/10 bg-white/5 px-2 py-0.5 text-[11px] font-medium text-slate-300"
                    >
                      {cls}
                    </span>
                  ))}
                </div>
              </div>

              {/* Metric stats grid */}
              <div className="mt-5 grid grid-cols-3 gap-2 border-t border-white/10 pt-4 text-center">
                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-2">
                  <span className="text-[10px] uppercase text-slate-400">
                    {model.accuracyMetric}
                  </span>
                  <div className="mt-1 font-mono text-base font-bold text-white">
                    {model.accuracyValue}%
                  </div>
                </div>

                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-2">
                  <span className="text-[10px] uppercase text-slate-400">Latency</span>
                  <div className="mt-1 font-mono text-base font-bold text-neon-cyan">
                    {model.latencyMs}ms
                  </div>
                </div>

                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-2">
                  <span className="text-[10px] uppercase text-slate-400">Throughput</span>
                  <div className="mt-1 font-mono text-base font-bold text-neon-green">
                    {model.fps} FPS
                  </div>
                </div>
              </div>

              {/* Precision & Recall */}
              <div className="mt-3 flex items-center justify-between text-xs font-mono text-slate-400">
                <span>Precision: {(model.precision * 100).toFixed(1)}%</span>
                <span>Recall: {(model.recall * 100).toFixed(1)}%</span>
                <span>F1-Score: {(model.f1Score * 100).toFixed(1)}%</span>
              </div>
            </GlassCard>
          ))}
        </motion.div>
      )}

      {/* Tab 2: Evaluation */}
      {activeTab === "evaluation" && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <GlassCard>
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                  Held-Out Benchmark Matrix (Validation Test Set)
                </h3>
                <p className="text-xs text-slate-400">
                  Evaluated on 4,200 curated test frames & 300 untrimmed surveillance videos
                </p>
              </div>
              <div className="rounded-xl border border-neon-green/30 bg-neon-green/10 px-3 py-1 text-xs font-mono text-neon-green">
                Verification: PASSED
              </div>
            </div>

            <div className="overflow-x-auto mt-4">
              <table className="w-full text-left text-xs font-mono">
                <thead className="border-b border-white/10 text-slate-400 text-[11px] uppercase">
                  <tr>
                    <th className="py-3 px-4">Threat Class</th>
                    <th className="py-3 px-4">Test Instances</th>
                    <th className="py-3 px-4">mAP@0.50</th>
                    <th className="py-3 px-4">Precision</th>
                    <th className="py-3 px-4">Recall</th>
                    <th className="py-3 px-4">False Positive Rate</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-slate-200">
                  <tr>
                    <td className="py-3.5 px-4 font-bold text-white">Knife / Blade</td>
                    <td className="py-3.5 px-4">1,480</td>
                    <td className="py-3.5 px-4 text-neon-cyan">94.2%</td>
                    <td className="py-3.5 px-4">95.8%</td>
                    <td className="py-3.5 px-4">92.6%</td>
                    <td className="py-3.5 px-4 text-neon-green">1.2%</td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-bold text-white">Gun / Firearm</td>
                    <td className="py-3.5 px-4">1,820</td>
                    <td className="py-3.5 px-4 text-neon-cyan">96.4%</td>
                    <td className="py-3.5 px-4">97.1%</td>
                    <td className="py-3.5 px-4">95.7%</td>
                    <td className="py-3.5 px-4 text-neon-green">0.8%</td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-bold text-white">Grenade / Explosive</td>
                    <td className="py-3.5 px-4">420</td>
                    <td className="py-3.5 px-4 text-neon-cyan">93.8%</td>
                    <td className="py-3.5 px-4">92.7%</td>
                    <td className="py-3.5 px-4">93.1%</td>
                    <td className="py-3.5 px-4 text-neon-green">1.9%</td>
                  </tr>
                  <tr>
                    <td className="py-3.5 px-4 font-bold text-white">Violent Altercation (S3D)</td>
                    <td className="py-3.5 px-4">480 clips</td>
                    <td className="py-3.5 px-4 text-neon-amber">91.2%</td>
                    <td className="py-3.5 px-4">89.6%</td>
                    <td className="py-3.5 px-4">91.4%</td>
                    <td className="py-3.5 px-4 text-neon-green">2.4%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </GlassCard>

          {/* Key Optimization Insight */}
          <GlassCard className="border-neon-cyan/20 bg-neon-cyan/[0.02]">
            <div className="flex items-start gap-3">
              <Sparkles className="h-5 w-5 text-neon-cyan shrink-0 mt-0.5" />
              <div className="text-xs">
                <h4 className="font-bold text-white">NMS Cross-Competition Solution</h4>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  The detector runs full 80-class COCO competition prior to Non-Maximum Suppression (NMS). This prevents everyday handheld objects (smartphones, pens, keys) from mistakenly collapsing into the knife category, reducing false weapon alerts by over 87%.
                </p>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* Tab 3: Neural Pipeline Architecture */}
      {activeTab === "architecture" && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <GlassCard>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">
              End-to-End Spatiotemporal Inference Graph
            </h3>
            <p className="mt-1 text-xs text-slate-400">
              Synchronized parallel pathways for spatial weapon detection and temporal action classification
            </p>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
              {/* Step 1 */}
              <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs">
                <div className="font-mono text-neon-cyan text-[11px] font-bold">01 • INGESTION</div>
                <h4 className="mt-1 font-bold text-white">RTSP / WebRTC Ingest</h4>
                <p className="mt-2 text-slate-400 leading-relaxed">
                  Direct frame capture into ring buffer via OpenCV/PyAV with drop-frame prevention.
                </p>
              </div>

              {/* Step 2 */}
              <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs">
                <div className="font-mono text-neon-cyan text-[11px] font-bold">02 • SPATIAL YOLO</div>
                <h4 className="mt-1 font-bold text-white">YOLO11n Detector</h4>
                <p className="mt-2 text-slate-400 leading-relaxed">
                  Sub-15ms inference extracting weapon silhouettes with calibrated 0.50 threshold.
                </p>
              </div>

              {/* Step 3 */}
              <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs">
                <div className="font-mono text-neon-amber text-[11px] font-bold">03 • TEMPORAL S3D</div>
                <h4 className="mt-1 font-bold text-white">S3D Action Head</h4>
                <p className="mt-2 text-slate-400 leading-relaxed">
                  32-frame rolling spatiotemporal window evaluates violent interaction kinetics.
                </p>
              </div>

              {/* Step 4 */}
              <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs">
                <div className="font-mono text-neon-green text-[11px] font-bold">04 • HARMONIZER</div>
                <h4 className="mt-1 font-bold text-white">Threat Index & Alarm</h4>
                <p className="mt-2 text-slate-400 leading-relaxed">
                  Composite scoring triggers Web Audio API buzzer siren and signs cryptographic SHA-256 evidence.
                </p>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}
