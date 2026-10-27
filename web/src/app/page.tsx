"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  ShieldAlert,
  Video,
  CheckCircle,
  Zap,
  ArrowRight,
  TrendingUp,
} from "lucide-react";
import { KpiCard } from "@/components/ui/kpi-card";
import { GlassCard } from "@/components/ui/glass-card";
import { SeverityBadge } from "@/components/ui/severity-badge";
import { StatusChip } from "@/components/ui/status-chip";
import { useAppStore } from "@/lib/store";
import {
  MOCK_SEVERITY_DISTRIBUTION,
  MOCK_DETECTIONS_TIMELINE,
} from "@/lib/mock-data";
import { formatTimestamp } from "@/lib/utils";
import Link from "next/link";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  AreaChart,
  Area,
  XAxis,
  YAxis,
} from "recharts";

export default function DashboardPage() {
  const { incidents, cameras, setSelectedIncident } = useAppStore();

  const openIncidentsCount = incidents.filter((i) => i.status === "open").length;
  const onlineCameras = cameras.filter((c) => c.status === "online").length;

  return (
    <div className="space-y-8">
      {/* Page Title & Subtitle */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
      >
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
            Command Center Overview
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Real-time neural surveillance, threat index telemetry, and active alerts.
          </p>
        </div>

        <Link
          href="/live"
          className="inline-flex items-center gap-2 self-start rounded-xl border border-neon-cyan/40 bg-neon-cyan/15 px-4 py-2 text-xs font-semibold text-neon-cyan shadow-[0_0_15px_rgba(34,211,238,0.25)] hover:bg-neon-cyan/25 transition-all"
        >
          <span>Open Live Monitor</span>
          <ArrowRight className="h-4 w-4" />
        </Link>
      </motion.div>

      {/* Row of 4 KPI Glass Cards */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.35 }}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        <KpiCard
          label="Open Incidents"
          value={openIncidentsCount}
          icon={ShieldAlert}
          glow="red"
          trend={{ value: "+1 new", isPositive: false }}
        />
        <KpiCard
          label="Cameras Online"
          value={onlineCameras}
          suffix={`/${cameras.length}`}
          icon={Video}
          glow="green"
          trend={{ value: "66.7% operational", isNeutral: true }}
        />
        <KpiCard
          label="False Alarm Rate"
          value={1.8}
          suffix="%"
          decimals={1}
          icon={CheckCircle}
          glow="cyan"
          trend={{ value: "-0.6%", isPositive: true }}
        />
        <KpiCard
          label="Inference Latency (p95)"
          value={14.2}
          suffix=" ms"
          decimals={1}
          icon={Zap}
          glow="violet"
          trend={{ value: "70.4 FPS", isPositive: true }}
        />
      </motion.div>

      {/* Center Section: Donut Chart (Left) + Latest Incidents List (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Incident Severity Donut Chart (5 cols) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4 }}
          className="lg:col-span-5"
        >
          <GlassCard className="h-full flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-white/10">
                <div>
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                    Severity Distribution
                  </h3>
                  <p className="text-xs text-slate-400">Past 24 hours threat triage</p>
                </div>
                <span className="font-mono text-xs text-neon-cyan">32 Total</span>
              </div>

              {/* Donut Chart */}
              <div className="relative h-56 w-full my-3 flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={MOCK_SEVERITY_DISTRIBUTION}
                      cx="50%"
                      cy="50%"
                      innerRadius={56}
                      outerRadius={80}
                      paddingAngle={4}
                      dataKey="count"
                      labelLine={{ stroke: "#ffffff", strokeWidth: 1 }}
                      label={({ x, y, textAnchor, percent }) => {
                        if (!percent || percent < 0.03) return null;
                        return (
                          <text
                            x={x}
                            y={y}
                            fill="#FFFFFF"
                            textAnchor={textAnchor}
                            dominantBaseline="central"
                            fontSize={11}
                            fontWeight={700}
                          >
                            {`${(percent * 100).toFixed(0)}%`}
                          </text>
                        );
                      }}
                    >
                      {MOCK_SEVERITY_DISTRIBUTION.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.color}
                          stroke="#070B14"
                          strokeWidth={2}
                        />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#070B14",
                        border: "1px solid rgba(255, 255, 255, 0.2)",
                        borderRadius: "12px",
                        fontSize: "12px",
                        color: "#ffffff",
                      }}
                      itemStyle={{ color: "#ffffff" }}
                      labelStyle={{ color: "#ffffff" }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                {/* Center Donut White Text */}
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <span className="text-xl font-bold font-mono text-white">32</span>
                  <span className="text-[10px] uppercase tracking-wider text-white font-medium">Total</span>
                </div>
              </div>
            </div>

            {/* Clean Legend with Pure White Text */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-2 border-t border-white/10">
              {MOCK_SEVERITY_DISTRIBUTION.map((item) => (
                <div key={item.name} className="flex items-center gap-2 text-xs">
                  <span
                    className="h-2.5 w-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-white font-medium">{item.name}</span>
                  <span className="font-mono font-bold text-white ml-auto">{item.count}</span>
                </div>
              ))}
            </div>
          </GlassCard>
        </motion.div>

        {/* Right: Latest Incidents List (7 cols) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="lg:col-span-7"
        >
          <GlassCard className="h-full flex flex-col">
            <div className="flex items-center justify-between pb-3 border-b border-white/10">
              <div>
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                  Latest Incidents
                </h3>
                <p className="text-xs text-slate-400">Click any row to open forensic triage drawer</p>
              </div>
              <Link
                href="/incidents"
                className="text-xs font-semibold text-neon-cyan hover:underline flex items-center gap-1"
              >
                <span>View All</span>
                <ArrowRight className="h-3 w-3" />
              </Link>
            </div>

            {/* Incident Rows */}
            <div className="mt-3 divide-y divide-white/5 overflow-hidden">
              {incidents.slice(0, 5).map((inc) => (
                <div
                  key={inc.id}
                  onClick={() => setSelectedIncident(inc)}
                  className="group flex cursor-pointer items-center justify-between py-3 px-2 rounded-xl transition-all duration-200 hover:bg-white/[0.05]"
                >
                  <div className="flex items-center gap-3 min-w-0 pr-2">
                    <SeverityBadge severity={inc.severity} showIcon={false} />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 font-mono text-xs">
                        <span className="font-bold text-white group-hover:text-neon-cyan transition-colors">
                          {inc.incidentCode}
                        </span>
                        <span className="text-slate-500">•</span>
                        <span className="text-slate-400 truncate">{inc.cameraName}</span>
                      </div>
                      <p className="text-xs text-slate-300 truncate mt-0.5">
                        {inc.shortDescription}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <span suppressHydrationWarning className="hidden sm:inline font-mono text-xs text-slate-400">
                      {formatTimestamp(inc.timestamp)}
                    </span>
                    <StatusChip status={inc.status} />
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>
        </motion.div>
      </div>

      {/* Bottom Section: Detections Over Time Line/Area Chart */}
      <motion.div
        initial={{ opacity: 0, y: 25 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <GlassCard>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-white/10 gap-2">
            <div>
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-neon-cyan" />
                <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                  Threat Activity Timeline
                </h3>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Hourly weapon vs. violent action detections across all camera zones
              </p>
            </div>

            <div className="flex items-center gap-4 text-xs font-mono">
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-neon-cyan" />
                <span className="text-slate-300">Weapons</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-neon-amber" />
                <span className="text-slate-300">Violence</span>
              </div>
            </div>
          </div>

          <div className="h-64 w-full mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={MOCK_DETECTIONS_TIMELINE} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="cyanGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22D3EE" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#22D3EE" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="amberGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#FBBF24" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#FBBF24" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="time"
                  stroke="#64748B"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                />
                <YAxis
                  stroke="#64748B"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#070B14",
                    border: "1px solid rgba(255, 255, 255, 0.15)",
                    borderRadius: "12px",
                    fontSize: "12px",
                    color: "#fff",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="weapons"
                  stroke="#22D3EE"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#cyanGradient)"
                />
                <Area
                  type="monotone"
                  dataKey="violence"
                  stroke="#FBBF24"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#amberGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>
      </motion.div>
    </div>
  );
}
