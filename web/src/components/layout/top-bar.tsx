"use client";

import React, { useState, useEffect } from "react";
import { LiveBadge } from "@/components/ui/live-badge";
import { useAppStore } from "@/lib/store";
import { ShieldAlert, Video, User, ChevronDown, Check } from "lucide-react";
import { cn } from "@/lib/utils";

export function TopBar() {
  const [utcTime, setUtcTime] = useState("");
  const [userDropdownOpen, setUserDropdownOpen] = useState(false);

  const { isSidebarCollapsed, incidents, cameras, currentUser, setCurrentUser, setSelectedIncident } =
    useAppStore();

  useEffect(() => {
    function updateClock() {
      const now = new Date();
      setUtcTime(
        now.toLocaleTimeString("en-US", {
          timeZone: "UTC",
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }) + " UTC"
      );
    }
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  const onlineCameras = cameras.filter((c) => c.status === "online").length;
  const reviewRequired = incidents.filter((i) => i.status === "open" || i.status === "in_review");

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-white/10 bg-navy-950/60 px-4 md:px-8 backdrop-blur-2xl transition-all duration-300">
      {/* Left items: Live Badge + Human Review Required Chip */}
      <div className="flex items-center gap-3">
        <LiveBadge label="SYS-ACTIVE" active={true} />

        {reviewRequired.length > 0 && (
          <button
            onClick={() => {
              if (reviewRequired.length > 0) {
                setSelectedIncident(reviewRequired[0]);
              }
            }}
            className="hidden sm:inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs font-medium text-amber-400 backdrop-blur-md hover:bg-amber-500/20 transition-colors"
          >
            <ShieldAlert className="h-3.5 w-3.5 text-neon-amber shrink-0" />
            <span>Human Review Required ({reviewRequired.length})</span>
          </button>
        )}
      </div>

      {/* Right items: Cameras Online, UTC Clock, User Selector */}
      <div className="flex items-center gap-4">
        {/* Cameras Online indicator */}
        <div className="hidden lg:flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-slate-300">
          <Video className="h-3.5 w-3.5 text-neon-green" />
          <span className="text-slate-400">Cameras:</span>
          <span className="font-mono font-semibold text-white">
            {onlineCameras}/{cameras.length}
          </span>
        </div>

        {/* UTC Clock */}
        <div suppressHydrationWarning className="hidden sm:block font-mono text-xs text-slate-300 tracking-wider">
          {utcTime || "00:00:00 UTC"}
        </div>

        {/* User Avatar Menu */}
        <div className="relative">
          <button
            onClick={() => setUserDropdownOpen(!userDropdownOpen)}
            className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] p-1.5 pr-2.5 text-xs hover:border-white/20 hover:bg-white/[0.08] transition-colors"
          >
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-tr from-cyan-600 to-indigo-600 font-mono text-xs font-bold text-white shadow-sm">
              {currentUser === "Operator_Lead" ? "OL" : "RP"}
            </div>
            <span className="hidden md:inline font-medium text-slate-200">
              {currentUser}
            </span>
            <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
          </button>

          {userDropdownOpen && (
            <div className="absolute right-0 mt-2 w-48 rounded-xl border border-white/10 bg-navy-900/95 p-1.5 shadow-xl backdrop-blur-2xl ring-1 ring-white/10 z-50">
              <div className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400 border-b border-white/10">
                Operator Role
              </div>
              <button
                onClick={() => {
                  setCurrentUser("Operator_Lead");
                  setUserDropdownOpen(false);
                }}
                className={cn(
                  "flex w-full items-center justify-between rounded-lg px-3 py-2 text-xs transition-colors",
                  currentUser === "Operator_Lead"
                    ? "bg-white/10 text-white font-medium"
                    : "text-slate-300 hover:bg-white/5"
                )}
              >
                <span>Operator_Lead</span>
                {currentUser === "Operator_Lead" && <Check className="h-3.5 w-3.5 text-neon-cyan" />}
              </button>
              <button
                onClick={() => {
                  setCurrentUser("Reviewer_Patel");
                  setUserDropdownOpen(false);
                }}
                className={cn(
                  "flex w-full items-center justify-between rounded-lg px-3 py-2 text-xs transition-colors",
                  currentUser === "Reviewer_Patel"
                    ? "bg-white/10 text-white font-medium"
                    : "text-slate-300 hover:bg-white/5"
                )}
              >
                <span>Reviewer_Patel</span>
                {currentUser === "Reviewer_Patel" && <Check className="h-3.5 w-3.5 text-neon-cyan" />}
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
