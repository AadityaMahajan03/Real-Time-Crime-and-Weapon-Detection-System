"use client";

import React from "react";
import { useAppStore } from "@/lib/store";
import { SidebarNav } from "@/components/layout/sidebar-nav";
import { TopBar } from "@/components/layout/top-bar";
import { IncidentDrawer } from "@/components/incidents/incident-drawer";
import { cn } from "@/lib/utils";

export function AppShell({ children }: { children: React.ReactNode }) {
  const { isSidebarCollapsed } = useAppStore();

  return (
    <div className="relative z-10 flex min-h-screen">
      {/* Slim Collapsible Sidebar */}
      <SidebarNav />

      {/* Main Application Column - Dynamically offset right to clear fixed sidebar */}
      <div
        className={cn(
          "flex flex-1 flex-col transition-all duration-300 w-full min-w-0",
          isSidebarCollapsed ? "md:pl-20" : "md:pl-64"
        )}
      >
        {/* Sticky Top Bar */}
        <TopBar />

        {/* Page Content Viewport */}
        <main className="flex-1 px-4 py-8 md:px-8 max-w-7xl w-full mx-auto pb-24 md:pb-12">
          {children}
        </main>
      </div>

      {/* Global Slide-Over Incident Forensic Drawer */}
      <IncidentDrawer />
    </div>
  );
}
