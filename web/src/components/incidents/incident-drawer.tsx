"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  ShieldAlert,
  CheckCircle2,
  AlertOctagon,
  ArrowUpRight,
  Clock,
  Camera as CameraIcon,
  Hash,
  Activity,
  History,
  Send,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { SeverityBadge } from "@/components/ui/severity-badge";
import { StatusChip } from "@/components/ui/status-chip";
import { formatDate, formatTimestamp, cn } from "@/lib/utils";

export function IncidentDrawer() {
  const { selectedIncident, setSelectedIncident, updateIncidentAction } = useAppStore();
  const [operatorNotes, setOperatorNotes] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);
  const [mediaMode, setMediaMode] = useState<"snapshot" | "video">("snapshot");

  // Lock background body scroll and prevent background Lenis scroll hijacking
  useEffect(() => {
    if (selectedIncident) {
      const originalOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = originalOverflow;
      };
    }
  }, [selectedIncident]);

  if (!selectedIncident) return null;

  async function handleAction(action: "CONFIRMED_THREAT" | "MARKED_FALSE_ALARM" | "ESCALATED") {
    if (!selectedIncident) return;
    setIsSubmitting(true);
    await updateIncidentAction(selectedIncident.id, action, operatorNotes.trim() || undefined);
    setOperatorNotes("");
    const label =
      action === "CONFIRMED_THREAT"
        ? "Threat Confirmed & Sealed"
        : action === "MARKED_FALSE_ALARM"
        ? "Marked as False Alarm (Video Purged)"
        : "Incident Escalated to Senior Watch";
    setActionFeedback(label);
    setTimeout(() => setActionFeedback(null), 3500);
    setIsSubmitting(false);
  }

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 overflow-hidden" data-lenis-prevent>
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={() => setSelectedIncident(null)}
          className="absolute inset-0 bg-navy-950/70 backdrop-blur-md transition-opacity"
        />

        {/* Slide-over panel */}
        <div className="fixed inset-y-0 right-0 flex max-w-full pl-10 h-full">
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="w-screen max-w-xl h-full max-h-screen border-l border-white/10 bg-navy-900/95 shadow-2xl backdrop-blur-2xl flex flex-col"
            data-lenis-prevent
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-white/10 px-6 py-5 shrink-0">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-neon-red/10 border border-neon-red/30 text-neon-red shadow-[0_0_15px_rgba(244,63,94,0.3)]">
                  <ShieldAlert className="h-5 w-5" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-mono text-base font-bold text-white">
                      {selectedIncident.incidentCode}
                    </h3>
                    <SeverityBadge severity={selectedIncident.severity} />
                  </div>
                  <p suppressHydrationWarning className="text-xs text-slate-400 mt-0.5">
                    {formatDate(selectedIncident.timestamp)} at{" "}
                    {formatTimestamp(selectedIncident.timestamp)}
                  </p>
                </div>
              </div>

              <button
                onClick={() => setSelectedIncident(null)}
                className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/10 text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Scrollable body */}
            <div
              className="flex-1 overflow-y-auto overscroll-contain px-6 py-5 space-y-6"
              data-lenis-prevent
            >
              {/* Evidence Snapshot / Video */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <div className="flex items-center gap-1.5 rounded-lg bg-black/40 p-0.5 border border-white/10 text-[11px]">
                    <button
                      onClick={() => setMediaMode("snapshot")}
                      className={cn(
                        "px-2.5 py-1 rounded-md transition-colors",
                        mediaMode === "snapshot"
                          ? "bg-white/10 text-white font-medium shadow-sm"
                          : "text-slate-400 hover:text-slate-200"
                      )}
                    >
                      Snapshot Frame
                    </button>
                    <button
                      onClick={() => setMediaMode("video")}
                      className={cn(
                        "px-2.5 py-1 rounded-md transition-colors",
                        mediaMode === "video"
                          ? "bg-white/10 text-white font-medium shadow-sm"
                          : "text-slate-400 hover:text-slate-200"
                      )}
                    >
                      Surveillance Clip
                    </button>
                  </div>
                  <StatusChip status={selectedIncident.status} />
                </div>

                <div className="relative aspect-video w-full overflow-hidden rounded-2xl border border-white/10 bg-black shadow-inner">
                  {mediaMode === "video" ? (
                    <video
                      key={selectedIncident.videoUrl || selectedIncident.id}
                      src={selectedIncident.videoUrl || "/clips/threat-knife-combat.mp4"}
                      controls
                      autoPlay
                      loop
                      playsInline
                      className="h-full w-full object-cover"
                      onError={(e) => {
                        const target = e.currentTarget;
                        if (!target.src.endsWith("/clips/threat-knife-combat.mp4")) {
                          target.src = "/clips/threat-knife-combat.mp4";
                        }
                      }}
                    />
                  ) : (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      key={selectedIncident.snapshotUrl || selectedIncident.id}
                      src={selectedIncident.snapshotUrl || "/evidence/inc-1094.jpg"}
                      alt="Surveillance evidence"
                      className="h-full w-full object-cover"
                      onError={(e) => {
                        const target = e.currentTarget;
                        if (!target.src.endsWith("/evidence/inc-1094.jpg")) {
                          target.src = "/evidence/inc-1094.jpg";
                        }
                      }}
                    />
                  )}
                  <div className="absolute bottom-2 left-2 flex items-center gap-2 rounded-lg bg-black/60 px-2.5 py-1 text-[11px] font-mono text-slate-300 backdrop-blur-md border border-white/10 pointer-events-none">
                    <CameraIcon className="h-3 w-3 text-neon-cyan" />
                    <span>{selectedIncident.cameraName}</span>
                  </div>
                </div>

                {/* SHA-256 Hash */}
                <div className="flex items-center gap-2 rounded-xl border border-white/5 bg-white/[0.02] p-2.5 text-[11px] text-slate-400 font-mono overflow-hidden">
                  <Hash className="h-3.5 w-3.5 text-neon-cyan shrink-0" />
                  <span className="truncate">SHA-256: {selectedIncident.sha256Hash}</span>
                </div>
              </div>

              {/* Key Stats Grid */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400">
                    Weapon Score
                  </span>
                  <div className="mt-1 font-mono text-xl font-bold text-neon-red">
                    {Math.round(selectedIncident.peakWeaponScore * 100)}%
                  </div>
                  <span className="text-[10px] text-slate-400 capitalize">
                    {selectedIncident.weaponType || "None"}
                  </span>
                </div>

                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400">
                    Violence Score
                  </span>
                  <div className="mt-1 font-mono text-xl font-bold text-neon-amber">
                    {Math.round(selectedIncident.peakViolenceScore * 100)}%
                  </div>
                  <span className="text-[10px] text-slate-400">S3D Action Head</span>
                </div>

                <div className="col-span-2 sm:col-span-1 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400">
                    Threat Index
                  </span>
                  <div className="mt-1 font-mono text-xl font-bold text-neon-cyan">
                    {Math.round(selectedIncident.compositeThreatScore * 100)}%
                  </div>
                  <span className="text-[10px] text-slate-400">Harmonized</span>
                </div>
              </div>

              {/* Incident Description */}
              <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3.5">
                <h4 className="text-xs font-semibold text-slate-300">Observation Summary</h4>
                <p className="mt-1 text-sm text-slate-200">
                  {selectedIncident.shortDescription}
                </p>
                <div className="mt-2 text-xs text-slate-400">
                  <span className="font-semibold text-slate-300">Location: </span>
                  {selectedIncident.location}
                </div>
              </div>

              {/* Incident Actions */}
              <div className="space-y-3 rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                <h4 className="text-xs font-bold uppercase tracking-wider text-white">
                  Operator Triage Actions
                </h4>

                {/* Operator Note Input */}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={operatorNotes}
                    onChange={(e) => setOperatorNotes(e.target.value)}
                    placeholder="Add operational notes or officer badge..."
                    className="flex-1 rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-neon-cyan focus:outline-none focus:ring-1 focus:ring-neon-cyan"
                  />
                </div>

                {/* Feedback Toast Banner */}
                {actionFeedback && (
                  <div className="flex items-center gap-2 rounded-xl border border-neon-cyan/40 bg-neon-cyan/15 px-3 py-2 text-xs font-semibold text-neon-cyan animate-pulse">
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-neon-cyan" />
                    <span>{actionFeedback}</span>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
                  <button
                    disabled={isSubmitting}
                    onClick={() => handleAction("CONFIRMED_THREAT")}
                    className={cn(
                      "flex items-center justify-center gap-1.5 rounded-xl border px-3 py-2.5 text-xs font-semibold transition-all",
                      selectedIncident.status === "confirmed"
                        ? "border-neon-cyan bg-neon-cyan/30 text-white shadow-[0_0_20px_rgba(34,211,238,0.4)]"
                        : "border-neon-cyan/40 bg-neon-cyan/15 text-neon-cyan shadow-[0_0_15px_rgba(34,211,238,0.25)] hover:bg-neon-cyan/25"
                    )}
                  >
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>{selectedIncident.status === "confirmed" ? "✓ Confirmed" : "Confirm Threat"}</span>
                  </button>

                  <button
                    disabled={isSubmitting}
                    onClick={() => handleAction("MARKED_FALSE_ALARM")}
                    className={cn(
                      "flex items-center justify-center gap-1.5 rounded-xl border px-3 py-2.5 text-xs font-semibold transition-all",
                      selectedIncident.status === "false_alarm"
                        ? "border-amber-400 bg-amber-400/20 text-white shadow-[0_0_15px_rgba(251,191,36,0.3)]"
                        : "border-white/15 bg-white/5 text-slate-300 hover:bg-white/10 hover:text-white"
                    )}
                  >
                    <AlertOctagon className="h-3.5 w-3.5 text-slate-400" />
                    <span>{selectedIncident.status === "false_alarm" ? "✓ False Alarm" : "False Alarm"}</span>
                  </button>

                  <button
                    disabled={isSubmitting}
                    onClick={() => handleAction("ESCALATED")}
                    className={cn(
                      "flex items-center justify-center gap-1.5 rounded-xl border px-3 py-2.5 text-xs font-semibold transition-all",
                      selectedIncident.status === "escalated"
                        ? "border-neon-red bg-neon-red/30 text-white shadow-[0_0_20px_rgba(244,63,94,0.4)]"
                        : "border-neon-red/40 bg-neon-red/15 text-neon-red shadow-[0_0_15px_rgba(244,63,94,0.25)] hover:bg-neon-red/25"
                    )}
                  >
                    <ArrowUpRight className="h-3.5 w-3.5" />
                    <span>{selectedIncident.status === "escalated" ? "✓ Escalated" : "Escalate"}</span>
                  </button>
                </div>
              </div>

              {/* Timeline */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  <Clock className="h-3.5 w-3.5 text-neon-cyan" />
                  <span>Detection Timeline</span>
                </div>
                <div className="space-y-2 border-l border-white/10 pl-4 ml-2">
                  {selectedIncident.timeline.map((event, idx) => (
                    <div key={idx} className="relative text-xs">
                      <span className="absolute -left-[21px] top-1 h-2 w-2 rounded-full bg-neon-cyan shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
                      <div className="font-mono text-[10px] text-slate-400">
                        {event.timestamp} • {event.actor}
                      </div>
                      <div className="text-slate-200 mt-0.5">{event.description}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Chain of Custody Audit Log */}
              <div className="space-y-3 border-t border-white/10 pt-4">
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  <History className="h-3.5 w-3.5 text-neon-violet" />
                  <span>Chain of Custody Audit Trail</span>
                </div>
                <div className="space-y-2">
                  {selectedIncident.auditLog.map((log) => (
                    <div
                      key={log.id}
                      className="rounded-xl border border-white/5 bg-white/[0.02] p-2.5 text-xs"
                    >
                      <div className="flex items-center justify-between font-mono text-[10px] text-slate-400">
                        <span className="text-neon-cyan font-bold">{log.action}</span>
                        <span>{log.timestamp}</span>
                      </div>
                      <div className="mt-1 text-slate-300 font-medium">{log.details}</div>
                      <div className="mt-0.5 text-[10px] text-slate-500">By: {log.actor}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </AnimatePresence>
  );
}
