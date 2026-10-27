"use client";

import React, { useMemo } from "react";
import { motion } from "framer-motion";
import {
  Search,
  Filter,
  ShieldAlert,
  ArrowUpDown,
  ExternalLink,
  Camera as CameraIcon,
  RotateCcw,
} from "lucide-react";
import { GlassCard } from "@/components/ui/glass-card";
import { SeverityBadge } from "@/components/ui/severity-badge";
import { StatusChip } from "@/components/ui/status-chip";
import { useAppStore } from "@/lib/store";
import { formatDate, formatTimestamp } from "@/lib/utils";

export default function IncidentsPage() {
  const {
    incidents,
    loadIncidents,
    cameras,
    setSelectedIncident,
    filterSeverity,
    setFilterSeverity,
    filterStatus,
    setFilterStatus,
    filterCamera,
    setFilterCamera,
    searchQuery,
    setSearchQuery,
  } = useAppStore();

  React.useEffect(() => {
    loadIncidents();
    const interval = setInterval(() => {
      loadIncidents();
    }, 3500);
    return () => clearInterval(interval);
  }, [loadIncidents]);

  const filteredIncidents = useMemo(() => {
    return incidents.filter((inc) => {
      // Filter severity
      if (filterSeverity !== "all" && inc.severity !== filterSeverity) return false;
      // Filter status
      if (filterStatus !== "all" && inc.status !== filterStatus) return false;
      // Filter camera
      if (filterCamera !== "all" && inc.cameraId !== filterCamera) return false;
      // Search query
      if (searchQuery.trim() !== "") {
        const q = searchQuery.toLowerCase();
        const matchesCode = inc.incidentCode.toLowerCase().includes(q);
        const matchesDesc = inc.shortDescription.toLowerCase().includes(q);
        const matchesCam = inc.cameraName.toLowerCase().includes(q);
        const matchesLoc = inc.location.toLowerCase().includes(q);
        if (!matchesCode && !matchesDesc && !matchesCam && !matchesLoc) return false;
      }
      return true;
    });
  }, [incidents, filterSeverity, filterStatus, filterCamera, searchQuery]);

  function resetFilters() {
    setFilterSeverity("all");
    setFilterStatus("all");
    setFilterCamera("all");
    setSearchQuery("");
  }

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
            Incident Registry & Audit
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Cryptographically sealed threat logs and operator forensic chain of custody.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-400 self-start">
          <span className="font-bold text-white">{filteredIncidents.length}</span>
          <span>of</span>
          <span className="font-bold text-white">{incidents.length}</span>
          <span>incidents showing</span>
        </div>
      </motion.div>

      {/* Filter and Search Bar */}
      <GlassCard className="p-4">
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
          {/* Search Input */}
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by Incident ID, camera, keyword, or location..."
              className="w-full rounded-xl border border-white/10 bg-white/[0.03] pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:border-neon-cyan focus:outline-none focus:ring-1 focus:ring-neon-cyan"
            />
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Severity Filter */}
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="rounded-xl border border-white/10 bg-navy-900 px-3 py-2 text-xs text-slate-200 focus:border-neon-cyan focus:outline-none"
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>

            {/* Status Filter */}
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="rounded-xl border border-white/10 bg-navy-900 px-3 py-2 text-xs text-slate-200 focus:border-neon-cyan focus:outline-none"
            >
              <option value="all">All Statuses</option>
              <option value="open">Open</option>
              <option value="in_review">In Review</option>
              <option value="confirmed">Confirmed</option>
              <option value="false_alarm">False Alarm</option>
              <option value="escalated">Escalated</option>
              <option value="resolved">Resolved</option>
            </select>

            {/* Camera Filter */}
            <select
              value={filterCamera}
              onChange={(e) => setFilterCamera(e.target.value)}
              className="rounded-xl border border-white/10 bg-navy-900 px-3 py-2 text-xs text-slate-200 focus:border-neon-cyan focus:outline-none"
            >
              <option value="all">All Cameras</option>
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>

            {/* Reset Button */}
            {(filterSeverity !== "all" ||
              filterStatus !== "all" ||
              filterCamera !== "all" ||
              searchQuery !== "") && (
              <button
                onClick={resetFilters}
                className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                <span>Reset</span>
              </button>
            )}
          </div>
        </div>
      </GlassCard>

      {/* Incidents Table */}
      <GlassCard className="p-0 overflow-hidden">
        {filteredIncidents.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-12 text-center">
            <ShieldAlert className="h-10 w-10 text-slate-600 mb-3" />
            <h3 className="text-sm font-semibold text-slate-300">No Incidents Found</h3>
            <p className="mt-1 text-xs text-slate-500 max-w-sm">
              No incidents match your current filter parameters. Try clearing the search or severity criteria.
            </p>
            <button
              onClick={resetFilters}
              className="mt-4 rounded-xl border border-neon-cyan/30 bg-neon-cyan/15 px-3.5 py-1.5 text-xs font-semibold text-neon-cyan hover:bg-neon-cyan/25"
            >
              Reset Filters
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-white/10 bg-white/[0.02] text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="py-3.5 pl-6 pr-3">Incident ID</th>
                  <th className="py-3.5 px-3">Severity</th>
                  <th className="py-3.5 px-3">Camera / Zone</th>
                  <th className="py-3.5 px-3">Threat Scores</th>
                  <th className="py-3.5 px-3">Status</th>
                  <th className="py-3.5 px-3">Timestamp</th>
                  <th className="py-3.5 pl-3 pr-6 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {filteredIncidents.map((inc) => (
                  <tr
                    key={inc.id}
                    onClick={() => setSelectedIncident(inc)}
                    className="group cursor-pointer transition-colors duration-150 hover:bg-white/[0.04]"
                  >
                    {/* ID & Type with Forensic Snapshot */}
                    <td className="py-3 pl-6 pr-3">
                      <div className="flex items-center gap-3">
                        <div className="relative h-11 w-16 shrink-0 overflow-hidden rounded-lg border border-white/10 bg-black shadow-sm">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={inc.snapshotUrl || "/evidence/inc-1094.jpg"}
                            alt={inc.incidentCode}
                            className="h-full w-full object-cover"
                            onError={(e) => {
                              const target = e.currentTarget;
                              if (!target.src.endsWith("/evidence/inc-1094.jpg")) {
                                target.src = "/evidence/inc-1094.jpg";
                              }
                            }}
                          />
                        </div>
                        <div>
                          <div className="font-mono font-bold text-white group-hover:text-neon-cyan transition-colors">
                            {inc.incidentCode}
                          </div>
                          <div className="text-[11px] text-slate-400 truncate max-w-xs mt-0.5">
                            {inc.shortDescription}
                          </div>
                        </div>
                      </div>
                    </td>

                    {/* Severity Badge */}
                    <td className="py-4 px-3">
                      <SeverityBadge severity={inc.severity} />
                    </td>

                    {/* Camera */}
                    <td className="py-4 px-3">
                      <div className="flex items-center gap-1.5 font-medium text-slate-200">
                        <CameraIcon className="h-3.5 w-3.5 text-neon-cyan shrink-0" />
                        <span className="truncate">{inc.cameraName}</span>
                      </div>
                      <div className="text-[10px] text-slate-500 truncate mt-0.5">
                        {inc.location}
                      </div>
                    </td>

                    {/* Threat Scores */}
                    <td className="py-4 px-3 font-mono">
                      <div className="flex items-center gap-2">
                        <span className="text-neon-red font-bold">
                          W: {Math.round(inc.peakWeaponScore * 100)}%
                        </span>
                        <span className="text-slate-600">|</span>
                        <span className="text-neon-amber font-bold">
                          V: {Math.round(inc.peakViolenceScore * 100)}%
                        </span>
                      </div>
                    </td>

                    {/* Status */}
                    <td className="py-4 px-3">
                      <StatusChip status={inc.status} />
                    </td>

                    {/* Timestamp */}
                    <td className="py-4 px-3 font-mono text-slate-400">
                      <div suppressHydrationWarning>{formatTimestamp(inc.timestamp)}</div>
                      <div suppressHydrationWarning className="text-[10px] text-slate-500">{formatDate(inc.timestamp)}</div>
                    </td>

                    {/* Open Drawer Button */}
                    <td className="py-4 pl-3 pr-6 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedIncident(inc);
                        }}
                        className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] font-medium text-slate-300 group-hover:border-neon-cyan/40 group-hover:bg-neon-cyan/15 group-hover:text-neon-cyan transition-all"
                      >
                        <span>Review</span>
                        <ExternalLink className="h-3 w-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlassCard>
    </div>
  );
}
