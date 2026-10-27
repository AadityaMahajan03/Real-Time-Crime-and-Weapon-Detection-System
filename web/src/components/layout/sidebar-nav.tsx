"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Video,
  ShieldAlert,
  Cpu,
  Settings,
  ChevronLeft,
  ChevronRight,
  Shield,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Live Monitor", href: "/live", icon: Video },
  { name: "Incidents", href: "/incidents", icon: ShieldAlert },
  { name: "AI Models", href: "/models", icon: Cpu },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function SidebarNav() {
  const pathname = usePathname();
  const { isSidebarCollapsed, toggleSidebar, incidents } = useAppStore();

  const openIncidentsCount = incidents.filter((i) => i.status === "open").length;

  return (
    <>
      {/* Desktop & Tablet Sidebar */}
      <aside
        className={cn(
          "hidden md:flex flex-col fixed top-0 left-0 bottom-0 z-40 bg-navy-950/70 backdrop-blur-2xl border-r border-white/10 transition-all duration-300 ease-in-out",
          isSidebarCollapsed ? "w-20" : "w-64"
        )}
      >
        {/* Brand Header */}
        <div className="flex h-16 items-center justify-between px-5 border-b border-white/10">
          <Link href="/" className="flex items-center gap-3 overflow-hidden group">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 text-white shadow-[0_0_15px_rgba(34,211,238,0.4)]">
              <Shield className="h-5 w-5" />
            </div>
            {!isSidebarCollapsed && (
              <div className="flex flex-col">
                <span className="font-mono text-sm font-bold tracking-wider text-white">
                  FAST-TRACK
                </span>
                <span className="text-[10px] uppercase tracking-widest text-neon-cyan font-mono">
                  AI Defense
                </span>
              </div>
            )}
          </Link>
          <button
            onClick={toggleSidebar}
            aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            {isSidebarCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 space-y-1.5 px-3 py-6">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive =
              item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);

            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "group relative flex items-center gap-3.5 rounded-xl px-3.5 py-3 text-sm font-medium transition-all duration-200",
                  isActive
                    ? "bg-white/[0.08] text-white shadow-inner font-semibold"
                    : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200"
                )}
              >
                {/* Neon left border glow for active item */}
                {isActive && (
                  <span className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-neon-cyan shadow-[0_0_12px_rgba(34,211,238,0.8)]" />
                )}

                <Icon
                  className={cn(
                    "h-5 w-5 shrink-0 transition-colors",
                    isActive
                      ? "text-neon-cyan"
                      : "text-slate-400 group-hover:text-slate-200"
                  )}
                />

                {!isSidebarCollapsed && (
                  <span className="truncate">{item.name}</span>
                )}

                {/* Badge for incidents */}
                {!isSidebarCollapsed && item.name === "Incidents" && openIncidentsCount > 0 && (
                  <span className="ml-auto rounded-full bg-neon-red/20 px-2 py-0.5 font-mono text-[11px] font-bold text-neon-red border border-neon-red/30">
                    {openIncidentsCount}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Footer info in sidebar */}
        {!isSidebarCollapsed && (
          <div className="p-4 border-t border-white/10">
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3 text-xs">
              <div className="flex items-center justify-between text-slate-400">
                <span>Threat Defense</span>
                <span className="font-mono text-neon-green">ACTIVE</span>
              </div>
              <div className="mt-1 text-[11px] text-slate-500 font-mono">
                CUDA 12.8 / RTX 3050
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* Mobile Bottom Navigation Bar */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-40 flex h-16 items-center justify-around border-t border-white/10 bg-navy-950/90 backdrop-blur-2xl px-2">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive =
            item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex flex-col items-center justify-center gap-1 py-1 px-3 text-[10px] transition-colors relative",
                isActive ? "text-neon-cyan font-semibold" : "text-slate-400 hover:text-slate-200"
              )}
            >
              {isActive && (
                <span className="absolute top-0 h-0.5 w-8 rounded-full bg-neon-cyan shadow-[0_0_8px_rgba(34,211,238,0.8)]" />
              )}
              <Icon className="h-5 w-5" />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>
    </>
  );
}
