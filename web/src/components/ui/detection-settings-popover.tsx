"use client";

import React, { useState, useRef, useEffect } from "react";
import { Sliders, Volume2, VolumeX, Shield, Check } from "lucide-react";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export function DetectionSettingsPopover() {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const { detectionSettings, updateDetectionSettings } = useAppStore();

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative inline-block text-left" ref={containerRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs font-medium backdrop-blur-md transition-all duration-200",
          isOpen
            ? "border-neon-cyan/40 bg-neon-cyan/10 text-neon-cyan shadow-[0_0_15px_rgba(34,211,238,0.25)]"
            : "border-white/10 bg-white/[0.04] text-slate-300 hover:border-white/20 hover:bg-white/[0.08]"
        )}
      >
        <Sliders className="h-3.5 w-3.5" />
        <span>Detection Settings</span>
      </button>

      {isOpen && (
        <div className="absolute right-0 top-full z-[100] mt-2 w-80 rounded-2xl border border-white/10 bg-navy-900/95 p-5 shadow-2xl backdrop-blur-2xl ring-1 ring-white/10">
          <div className="flex items-center justify-between border-b border-white/10 pb-3">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-neon-cyan" />
              <h4 className="text-xs font-bold uppercase tracking-wider text-white">
                Inference Tuning
              </h4>
            </div>
            <span className="text-[10px] font-mono text-slate-400">YOLO + S3D</span>
          </div>

          <div className="mt-4 space-y-4">
            {/* Weapon Threshold */}
            <div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300">Weapon Threshold</span>
                <span className="font-mono text-neon-cyan">
                  {Math.round(detectionSettings.weaponThreshold * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.90"
                step="0.05"
                value={detectionSettings.weaponThreshold}
                onChange={(e) =>
                  updateDetectionSettings({ weaponThreshold: parseFloat(e.target.value) })
                }
                className="mt-1.5 h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-white/10 accent-neon-cyan"
              />
              <span className="text-[10px] text-slate-500">
                Calibrated to 50% to prevent false knife triggers
              </span>
            </div>

            {/* Violence Threshold */}
            <div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300">Violence Threshold</span>
                <span className="font-mono text-neon-amber">
                  {Math.round(detectionSettings.violenceThreshold * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.90"
                step="0.05"
                value={detectionSettings.violenceThreshold}
                onChange={(e) =>
                  updateDetectionSettings({ violenceThreshold: parseFloat(e.target.value) })
                }
                className="mt-1.5 h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-white/10 accent-neon-amber"
              />
            </div>

            {/* FPS Limit */}
            <div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300">FPS Limit</span>
                <span className="font-mono text-slate-200">
                  {detectionSettings.fpsLimit} FPS
                </span>
              </div>
              <input
                type="range"
                min="10"
                max="60"
                step="5"
                value={detectionSettings.fpsLimit}
                onChange={(e) =>
                  updateDetectionSettings({ fpsLimit: parseInt(e.target.value) })
                }
                className="mt-1.5 h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-white/10 accent-slate-300"
              />
            </div>

            {/* Siren Toggle */}
            <div className="flex items-center justify-between border-t border-white/10 pt-3">
              <div className="flex items-center gap-2">
                {detectionSettings.sirenEnabled ? (
                  <Volume2 className="h-4 w-4 text-neon-red" />
                ) : (
                  <VolumeX className="h-4 w-4 text-slate-500" />
                )}
                <div>
                  <div className="text-xs font-medium text-slate-200">Acoustic Siren</div>
                  <div className="text-[10px] text-slate-500">Web Audio API buzzer</div>
                </div>
              </div>
              <button
                type="button"
                onClick={() =>
                  updateDetectionSettings({
                    sirenEnabled: !detectionSettings.sirenEnabled,
                  })
                }
                className={cn(
                  "relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none",
                  detectionSettings.sirenEnabled ? "bg-neon-red" : "bg-white/20"
                )}
              >
                <span
                  className={cn(
                    "pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out",
                    detectionSettings.sirenEnabled ? "translate-x-4" : "translate-x-0"
                  )}
                />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
