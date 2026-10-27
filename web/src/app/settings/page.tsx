"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Video,
  Cpu,
  Users,
  HelpCircle,
  Plus,
  Save,
  CheckCircle,
  HardDrive,
  Shield,
  Activity,
  Terminal,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { useAppStore } from "@/lib/store";
import { MOCK_SYSTEM_METRICS } from "@/lib/mock-data";
import { cn } from "@/lib/utils";

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<"cameras" | "system" | "users" | "help">("cameras");
  const { cameras, currentUser, setCurrentUser, detectionSettings, updateDetectionSettings } =
    useAppStore();

  const [savedSuccess, setSavedSuccess] = useState(false);

  function triggerSaveNotification() {
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  }

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
      >
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
            System Settings & Administration
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Configure camera inputs, hardware engine thresholds, access controls, and compliance guidelines.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex flex-wrap items-center rounded-xl border border-white/10 bg-black/40 p-1 self-start">
          {[
            { id: "cameras", label: "Cameras & Zones", icon: Video },
            { id: "system", label: "System Telemetry", icon: Cpu },
            { id: "users", label: "Users & Audit", icon: Users },
            { id: "help", label: "Help & Rules", icon: HelpCircle },
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

      {/* Save Notification */}
      {savedSuccess && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 rounded-xl border border-neon-green/30 bg-neon-green/10 p-3 text-xs text-neon-green"
        >
          <CheckCircle className="h-4 w-4" />
          <span>Configuration saved and applied to edge runtime successfully.</span>
        </motion.div>
      )}

      {/* Tab 1: Cameras & Zones */}
      {activeTab === "cameras" && (
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
                  Connected Surveillance Feeds
                </h3>
                <p className="text-xs text-slate-400">Manage RTSP and WebRTC camera stream streams</p>
              </div>
              <button
                onClick={triggerSaveNotification}
                className="inline-flex items-center gap-1.5 rounded-xl border border-neon-cyan/40 bg-neon-cyan/15 px-3 py-1.5 text-xs font-semibold text-neon-cyan hover:bg-neon-cyan/25"
              >
                <Plus className="h-3.5 w-3.5" />
                <span>Add Camera</span>
              </button>
            </div>

            <div className="overflow-x-auto mt-4">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-white/10 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  <tr>
                    <th className="py-3 px-3">Camera Name</th>
                    <th className="py-3 px-3">Location</th>
                    <th className="py-3 px-3">RTSP Endpoint</th>
                    <th className="py-3 px-3">Resolution</th>
                    <th className="py-3 px-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 font-mono">
                  {cameras.map((cam) => (
                    <tr key={cam.id} className="hover:bg-white/[0.02]">
                      <td className="py-3 px-3 font-bold text-white font-sans">{cam.name}</td>
                      <td className="py-3 px-3 text-slate-300 font-sans">{cam.location}</td>
                      <td className="py-3 px-3 text-neon-cyan truncate max-w-xs">{cam.rtspUrl}</td>
                      <td className="py-3 px-3 text-slate-400">{cam.resolution}</td>
                      <td className="py-3 px-3">
                        <span
                          className={cn(
                            "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase",
                            cam.status === "online"
                              ? "bg-neon-green/10 text-neon-green border border-neon-green/20"
                              : "bg-slate-800 text-slate-400 border border-slate-700"
                          )}
                        >
                          <span
                            className={cn(
                              "h-1.5 w-1.5 rounded-full",
                              cam.status === "online" ? "bg-neon-green" : "bg-slate-500"
                            )}
                          />
                          <span>{cam.status}</span>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* Tab 2: System Telemetry */}
      {activeTab === "system" && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          {/* Hardware Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <GlassCard>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">GPU Device</div>
              <div className="mt-2 font-mono text-sm font-bold text-white leading-snug">
                {MOCK_SYSTEM_METRICS.gpuName}
              </div>
              <div className="mt-3 flex items-center justify-between text-xs font-mono text-neon-cyan">
                <span>VRAM Usage:</span>
                <span>
                  {MOCK_SYSTEM_METRICS.gpuMemoryUsedMb} / {MOCK_SYSTEM_METRICS.gpuMemoryTotalMb} MB
                </span>
              </div>
            </GlassCard>

            <GlassCard>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">GPU Load</div>
              <div className="mt-2 font-mono text-3xl font-bold text-neon-green">
                {MOCK_SYSTEM_METRICS.gpuUtilization}%
              </div>
              <div className="mt-3 text-xs text-slate-400">TensorRT & CUDA Acceleration</div>
            </GlassCard>

            <GlassCard>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">CPU & RAM</div>
              <div className="mt-2 font-mono text-3xl font-bold text-neon-cyan">
                {MOCK_SYSTEM_METRICS.ramUsagePercent}%
              </div>
              <div className="mt-3 text-xs text-slate-400">
                CPU Utilization: {MOCK_SYSTEM_METRICS.cpuUtilization}%
              </div>
            </GlassCard>

            <GlassCard>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">
                Database Engine
              </div>
              <div className="mt-2 font-mono text-3xl font-bold text-neon-violet">
                {MOCK_SYSTEM_METRICS.sqliteWalSizeMb} MB
              </div>
              <div className="mt-3 text-xs text-slate-400">SQLite in WAL Mode (fasttrack.db)</div>
            </GlassCard>
          </div>

          {/* Operational Tuning Form */}
          <GlassCard>
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <div>
                <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                  Detection Runtime Controls
                </h3>
                <p className="text-xs text-slate-400">Fine-tune edge alarm triggers and alert rules</p>
              </div>
              <button
                onClick={triggerSaveNotification}
                className="flex items-center gap-1.5 rounded-xl border border-neon-cyan/40 bg-neon-cyan/15 px-3 py-1.5 text-xs font-semibold text-neon-cyan hover:bg-neon-cyan/25"
              >
                <Save className="h-3.5 w-3.5" />
                <span>Save Changes</span>
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
              <div>
                <label className="text-xs font-medium text-slate-300">Default Weapon Threshold</label>
                <input
                  type="range"
                  min="0.2"
                  max="0.9"
                  step="0.05"
                  value={detectionSettings.weaponThreshold}
                  onChange={(e) =>
                    updateDetectionSettings({ weaponThreshold: parseFloat(e.target.value) })
                  }
                  className="mt-2 w-full h-1.5 rounded-lg bg-white/10 accent-neon-cyan"
                />
                <div className="mt-1 flex justify-between text-xs font-mono text-slate-400">
                  <span>Sensitive (20%)</span>
                  <span className="text-neon-cyan font-bold">
                    {Math.round(detectionSettings.weaponThreshold * 100)}%
                  </span>
                  <span>Strict (90%)</span>
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-300">Violence Action Threshold</label>
                <input
                  type="range"
                  min="0.2"
                  max="0.9"
                  step="0.05"
                  value={detectionSettings.violenceThreshold}
                  onChange={(e) =>
                    updateDetectionSettings({ violenceThreshold: parseFloat(e.target.value) })
                  }
                  className="mt-2 w-full h-1.5 rounded-lg bg-white/10 accent-neon-amber"
                />
                <div className="mt-1 flex justify-between text-xs font-mono text-slate-400">
                  <span>Sensitive (20%)</span>
                  <span className="text-neon-amber font-bold">
                    {Math.round(detectionSettings.violenceThreshold * 100)}%
                  </span>
                  <span>Strict (90%)</span>
                </div>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* Tab 3: Users & Audit Log */}
      {activeTab === "users" && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <GlassCard>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">
              Operator Role & Access
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Currently active operator credentials used for incident chain-of-custody signing
            </p>

            <div className="mt-4 flex flex-wrap gap-3">
              <button
                onClick={() => {
                  setCurrentUser("Operator_Lead");
                  triggerSaveNotification();
                }}
                className={cn(
                  "flex items-center gap-3 rounded-2xl border p-4 text-left transition-all",
                  currentUser === "Operator_Lead"
                    ? "border-neon-cyan/50 bg-neon-cyan/10"
                    : "border-white/10 bg-white/[0.02] hover:bg-white/5"
                )}
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/20 text-neon-cyan font-bold">
                  OL
                </div>
                <div>
                  <div className="text-sm font-bold text-white">Operator_Lead</div>
                  <div className="text-xs text-slate-400">Full Triage & Escalation Authority</div>
                </div>
              </button>

              <button
                onClick={() => {
                  setCurrentUser("Reviewer_Patel");
                  triggerSaveNotification();
                }}
                className={cn(
                  "flex items-center gap-3 rounded-2xl border p-4 text-left transition-all",
                  currentUser === "Reviewer_Patel"
                    ? "border-neon-cyan/50 bg-neon-cyan/10"
                    : "border-white/10 bg-white/[0.02] hover:bg-white/5"
                )}
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/20 text-indigo-400 font-bold">
                  RP
                </div>
                <div>
                  <div className="text-sm font-bold text-white">Reviewer_Patel</div>
                  <div className="text-xs text-slate-400">Forensic Audit & Evidence Verification</div>
                </div>
              </button>
            </div>
          </GlassCard>
        </motion.div>
      )}

      {/* Tab 4: Help & Limitations */}
      {activeTab === "help" && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <GlassCard>
            <h3 className="text-sm font-bold uppercase tracking-wider text-white">
              Surveillance Limitations & Operating Rules
            </h3>
            <p className="mt-1 text-xs text-slate-400">
              Technical constraints and recommended surveillance deployment parameters
            </p>

            <div className="mt-6 space-y-4 text-xs text-slate-300">
              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3.5">
                <h4 className="font-bold text-white">Minimum Object Pixel Size</h4>
                <p className="mt-1 text-slate-400 leading-relaxed">
                  Weapons must occupy at least 400 square pixels (e.g. 20x20px) to be reliably resolved. For long corridor installations, optical zoom or 2K cameras are recommended.
                </p>
              </div>

              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3.5">
                <h4 className="font-bold text-white">Illumination & Low Light</h4>
                <p className="mt-1 text-slate-400 leading-relaxed">
                  Infrared cut filters and minimum 5 lux illumination are required. In pitch dark scenes without IR lighting, false negatives may increase.
                </p>
              </div>

              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3.5">
                <h4 className="font-bold text-white">Evidence Integrity & Cryptography</h4>
                <p className="mt-1 text-slate-400 leading-relaxed">
                  Every incident keyframe is cryptographically stamped with SHA-256 immediately upon detection. Triage actions taken by operators are permanently immutably logged in the forensic database.
                </p>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}
    </div>
  );
}
