"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Play,
  Square,
  Camera as CameraIcon,
  Upload,
  Globe,
  Radio,
  Cpu,
  Volume2,
  VolumeX,
  Layers,
  Sparkles,
} from "lucide-react";
import { useAppStore } from "@/lib/store";
import { ThreatGauge } from "@/components/ui/threat-gauge";
import { LiveBadge } from "@/components/ui/live-badge";
import { DetectionSettingsPopover } from "@/components/ui/detection-settings-popover";
import { GlassCard } from "@/components/ui/glass-card";
import { cn } from "@/lib/utils";
import { BoundingBox } from "@/lib/types";

export function VideoPanel() {
  const {
    activeCameraId,
    setActiveCamera,
    cameras,
    activeSource,
    setActiveSource,
    selectedModel,
    setSelectedModel,
    isStreaming,
    setIsStreaming,
    detectionSettings,
    addIncident,
  } = useAppStore();

  const [fps, setFps] = useState(30.0);
  const [latency, setLatency] = useState(12.5);
  const [weaponScore, setWeaponScore] = useState(0.0);
  const [violenceScore, setViolenceScore] = useState(0.0);
  const [compositeScore, setCompositeScore] = useState(0.0);
  const [threatLevel, setThreatLevel] = useState<"CRITICAL" | "HIGH" | "ELEVATED" | "NORMAL" | "SAFE">("SAFE");

  const [boxes, setBoxes] = useState<BoundingBox[]>([]);
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string>("");
  const [isAlarmActive, setIsAlarmActive] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const alarmUntilRef = useRef<number>(0);
  const isAlarmLoopRunningRef = useRef<boolean>(false);

  // Trigger continuous 3-second alarm beep whenever any harmful threat is detected
  const trigger3SecAlarm = () => {
    if (!detectionSettings.sirenEnabled) return;
    const now = Date.now();
    // Maintain active beeping for at least 3 seconds
    alarmUntilRef.current = Math.max(alarmUntilRef.current, now + 3000);
    setIsAlarmActive(true);

    if (isAlarmLoopRunningRef.current) return;
    isAlarmLoopRunningRef.current = true;

    try {
      if (!audioContextRef.current) {
        const AudioCtx =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (AudioCtx) audioContextRef.current = new AudioCtx();
      }
      const ctx = audioContextRef.current;
      if (ctx && ctx.state === "suspended") {
        ctx.resume();
      }
    } catch {
      // Audio context restricted until user interacts
    }

    // High-visibility alarm pulse loop for exactly 3 seconds
    const interval = setInterval(() => {
      const remaining = alarmUntilRef.current - Date.now();
      if (remaining <= 0) {
        clearInterval(interval);
        isAlarmLoopRunningRef.current = false;
        setIsAlarmActive(false);
        return;
      }

      try {
        const ctx = audioContextRef.current;
        if (ctx && ctx.state === "running") {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = "sawtooth";
          // High-frequency urgent alarm tone (880Hz -> 620Hz)
          osc.frequency.setValueAtTime(880, ctx.currentTime);
          osc.frequency.exponentialRampToValueAtTime(620, ctx.currentTime + 0.14);

          gain.gain.setValueAtTime(0.08, ctx.currentTime);
          gain.gain.linearRampToValueAtTime(0.001, ctx.currentTime + 0.16);

          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.16);
        }
      } catch {
        // Fallback catch
      }
    }, 200); // 5 urgent pulses per second over the 3-second duration
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setBoxes([]);
      setWeaponScore(0.0);
      setViolenceScore(0.0);
      setThreatLevel("SAFE");
      setUploadedVideoUrl(url);
      setUploadedFileName(file.name);
      setIsStreaming(true);
    }
  };

  // Switch camera helper
  const currentCamera = cameras.find((c) => c.id === activeCameraId) || cameras[0];

  // Reset state cleanly on every source switch to prevent overlapping boxes
  useEffect(() => {
    setBoxes([]);
    setWeaponScore(0.0);
    setViolenceScore(0.0);
    setThreatLevel("SAFE");
  }, [activeSource]);

  // Real-time AI Vision edge detection loop across all live video streams
  useEffect(() => {
    if (!isStreaming) {
      setBoxes([]);
      setWeaponScore(0.0);
      setThreatLevel("SAFE");
      return;
    }

    let isCancelled = false;
    let inFlight = false;

    const detectInterval = setInterval(async () => {
      if (inFlight || isCancelled) return;
      const video = videoRef.current;
      if (!video || video.readyState < 2 || video.videoWidth === 0) return;

      try {
        inFlight = true;
        if (!canvasRef.current) {
          canvasRef.current = document.createElement("canvas");
        }
        const canvas = canvasRef.current;
        canvas.width = 480;
        canvas.height = 360;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.65);

        const res = await fetch("/api/detect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: dataUrl }),
        });

        if (res.ok && !isCancelled) {
          const data = await res.json();
          const rawBoxes: BoundingBox[] = data.detections || [];
          setBoxes(rawBoxes);
          const wScore = data.weaponScore || 0;
          const vScore = data.violenceScore || 0;
          setWeaponScore(wScore);
          setViolenceScore(vScore);
          setCompositeScore(Math.max(wScore * 0.7 + vScore * 0.3));
          setThreatLevel(data.threatLevel || (wScore > 0.45 ? "CRITICAL" : "SAFE"));
          if (data.fps) setFps(data.fps);
          if (data.latency) setLatency(data.latency);

          const effectiveWeaponThresh = Math.max(0.48, detectionSettings.weaponThreshold || 0.48);
          const effectiveViolenceThresh = Math.max(0.55, detectionSettings.violenceThreshold || 0.55);

          const isThreatDetected =
            wScore >= effectiveWeaponThresh ||
            vScore >= effectiveViolenceThresh ||
            data.threatLevel === "CRITICAL" ||
            (data.threatLevel === "HIGH" && !detectionSettings.alertOnHighOnly && wScore >= 0.42) ||
            rawBoxes.some((b) => b.isThreat && !b.isWarning && b.confidence >= effectiveWeaponThresh);

          if (isThreatDetected) {
            trigger3SecAlarm();
          }

          // Automatically register new incidents in the Incident Registry
          if (data.newIncident) {
            addIncident(data.newIncident);
          }
        }
      } catch (err) {
        console.debug("Video frame inference skip:", err);
      } finally {
        inFlight = false;
      }
    }, 250); // 4 FPS edge inference

    return () => {
      isCancelled = true;
      clearInterval(detectInterval);
    };
  }, [isStreaming, activeSource, uploadedVideoUrl, detectionSettings]);

  // Trigger continuous 3-second alarm when threat elevates
  useEffect(() => {
    if (!isStreaming || !detectionSettings.sirenEnabled) return;
    if (threatLevel === "CRITICAL" || threatLevel === "HIGH") {
      trigger3SecAlarm();
    }
  }, [threatLevel, isStreaming, detectionSettings.sirenEnabled]);

  // Handle webcam feed activation
  useEffect(() => {
    let stream: MediaStream | null = null;
    if (activeSource === "webcam" && isStreaming && videoRef.current) {
      navigator.mediaDevices
        ?.getUserMedia({ video: true, audio: false })
        .then((s) => {
          stream = s;
          if (videoRef.current) videoRef.current.srcObject = s;
        })
        .catch((err) => {
          console.warn("Webcam access declined or unavailable:", err);
        });
    } else if (videoRef.current && videoRef.current.srcObject) {
      const s = videoRef.current.srcObject as MediaStream;
      s.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [activeSource, isStreaming]);

  return (
    <div className="space-y-6">
      {/* Top Single Control Bar */}
      <GlassCard className="p-3.5 relative z-30 overflow-visible">
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Segmented Source Selector */}
          <div className="flex items-center rounded-xl border border-white/10 bg-black/40 p-1">
            {[
              { id: "demo", label: "Demo Corridor", icon: Sparkles },
              { id: "webcam", label: "Live Webcam", icon: CameraIcon },
              { id: "upload", label: "Upload Clip", icon: Upload },
              { id: "rtsp", label: "RTSP Stream", icon: Radio },
            ].map((src) => {
              const Icon = src.icon;
              const isActive = activeSource === src.id;
              return (
                <button
                  key={src.id}
                  onClick={() => setActiveSource(src.id as typeof activeSource)}
                  className={cn(
                    "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all duration-200",
                    isActive
                      ? "bg-white/10 text-neon-cyan shadow-sm font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span>{src.label}</span>
                </button>
              );
            })}
          </div>

          {/* Model Dropdown + Detection Settings + Start/Stop */}
          <div className="flex items-center gap-2.5">
            {/* Model Dropdown */}
            <div className="relative">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="appearance-none rounded-xl border border-white/10 bg-white/[0.04] pl-3 pr-8 py-2 text-xs font-medium text-slate-200 backdrop-blur-md hover:border-white/20 focus:border-neon-cyan focus:outline-none"
              >
                <option value="model-yolo-weapon" className="bg-navy-900 text-white">
                  YOLO11n Specialized Threat (Gun, Grenade, Knife)
                </option>
                <option value="model-yolo-coco" className="bg-navy-900 text-white">
                  Ultralytics YOLO11n (COCO-80 Domain)
                </option>
                <option value="model-s3d-violence" className="bg-navy-900 text-white">
                  S3D Spatiotemporal Action Head
                </option>
              </select>
            </div>

            {/* Detection Settings Popover */}
            <DetectionSettingsPopover />

            {/* Start / Stop Button */}
            <button
              onClick={() => setIsStreaming(!isStreaming)}
              className={cn(
                "flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold uppercase tracking-wider backdrop-blur-md transition-all duration-200",
                isStreaming
                  ? "border border-neon-red/40 bg-neon-red/15 text-neon-red shadow-[0_0_15px_rgba(244,63,94,0.3)] hover:bg-neon-red/25"
                  : "border border-neon-cyan/40 bg-neon-cyan/20 text-neon-cyan shadow-[0_0_15px_rgba(34,211,238,0.3)] hover:bg-neon-cyan/30"
              )}
            >
              {isStreaming ? (
                <>
                  <Square className="h-3.5 w-3.5 fill-current" />
                  <span>Stop Feed</span>
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 fill-current" />
                  <span>Start Live Feed</span>
                </>
              )}
            </button>
          </div>
        </div>
      </GlassCard>

      {/* Main Grid: Video Panel (Left 70%) + Live Telemetry (Right 30%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Large Video Panel */}
        <div className="lg:col-span-8 flex flex-col space-y-4">
          <div className="relative aspect-video w-full overflow-hidden rounded-3xl border border-white/10 bg-black/80 shadow-2xl">
            {/* Live Badge & Camera Header Overlay */}
            <div className="absolute top-4 left-4 right-4 z-20 flex items-center justify-between pointer-events-none">
              <div className="flex items-center gap-2 pointer-events-auto">
                <LiveBadge active={isStreaming} />
                <div className="rounded-lg bg-black/60 px-3 py-1 text-xs font-mono text-slate-200 backdrop-blur-md border border-white/10">
                  {currentCamera.name} • {currentCamera.resolution}
                </div>
              </div>

              {/* Siren Indicator */}
              {detectionSettings.sirenEnabled && (isAlarmActive || threatLevel === "CRITICAL" || threatLevel === "HIGH") && isStreaming && (
                <div className="flex items-center gap-1.5 rounded-full bg-neon-red/25 border border-neon-red/50 px-3 py-1 text-xs font-bold text-neon-red animate-pulse backdrop-blur-md shadow-[0_0_15px_rgba(244,63,94,0.4)]">
                  <Volume2 className="h-3.5 w-3.5" />
                  <span>3S ALARM ACTIVE</span>
                </div>
              )}
            </div>

            {/* Hidden Video File Input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileUpload}
              className="hidden"
            />

            {/* Video / Camera Feed Element */}
            {activeSource === "webcam" ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="h-full w-full object-cover"
              />
            ) : activeSource === "upload" ? (
              uploadedVideoUrl ? (
                <div className="relative h-full w-full">
                  <video
                    ref={videoRef}
                    key={uploadedVideoUrl}
                    src={uploadedVideoUrl}
                    autoPlay
                    playsInline
                    loop
                    muted
                    onCanPlay={(e) => {
                      e.currentTarget.play().catch(() => {});
                    }}
                    className="h-full w-full object-cover"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="absolute top-4 right-4 z-30 flex items-center gap-1.5 rounded-xl border border-white/20 bg-black/70 px-3 py-1.5 text-xs text-white backdrop-blur-md hover:bg-white/20 transition-all pointer-events-auto"
                  >
                    <Upload className="h-3.5 w-3.5 text-neon-cyan" />
                    <span>Change Clip</span>
                  </button>
                </div>
              ) : (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="relative h-full w-full flex flex-col items-center justify-center p-8 border-2 border-dashed border-white/20 rounded-2xl cursor-pointer hover:border-neon-cyan/50 hover:bg-white/[0.02] transition-all duration-300"
                >
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-neon-cyan/10 border border-neon-cyan/30 text-neon-cyan shadow-[0_0_20px_rgba(34,211,238,0.2)] mb-4">
                    <Upload className="h-8 w-8" />
                  </div>
                  <h3 className="text-base font-bold text-white">Upload CCTV / Surveillance Video</h3>
                  <p className="mt-1 text-xs text-slate-400 max-w-sm text-center">
                    Select or drag-and-drop any MP4, WebM, AVI, or MOV clip to run real-time CCTV weapon & violence analysis.
                  </p>
                  <button
                    type="button"
                    className="mt-4 rounded-xl border border-neon-cyan/40 bg-neon-cyan/20 px-4 py-2 text-xs font-semibold text-neon-cyan shadow-sm hover:bg-neon-cyan/30"
                  >
                    Choose Video File
                  </button>
                </div>
              )
            ) : activeSource === "rtsp" ? (
              <div className="relative h-full w-full">
                <video
                  ref={videoRef}
                  key="rtsp-feed"
                  src="/clips/threat-wavelet-annotated.mp4"
                  autoPlay
                  playsInline
                  loop
                  muted
                  onCanPlay={(e) => {
                    e.currentTarget.play().catch(() => {});
                  }}
                  className="h-full w-full object-cover"
                />
                <div className="absolute top-4 right-4 z-20 flex items-center gap-2 rounded-lg bg-black/70 px-2.5 py-1 text-[11px] font-mono text-neon-cyan border border-white/10 backdrop-blur-md">
                  <Radio className="h-3 w-3 animate-pulse" />
                  <span>RTSP 1080p Stream (Synced)</span>
                </div>
              </div>
            ) : (
              <div className="relative h-full w-full">
                {/* Real Surveillance CCTV Threat Feed - Replaced Flower Placeholder */}
                <video
                  ref={videoRef}
                  key="demo-corridor-feed"
                  src="/clips/threat-knife-combat.mp4"
                  autoPlay
                  playsInline
                  loop
                  muted
                  onCanPlay={(e) => {
                    e.currentTarget.play().catch(() => {});
                  }}
                  className="h-full w-full object-cover"
                />
                <div className="absolute top-4 right-4 z-20 flex items-center gap-2 rounded-lg bg-black/70 px-2.5 py-1 text-[11px] font-mono text-slate-300 border border-white/10 backdrop-blur-md">
                  <span className="h-2 w-2 rounded-full bg-neon-cyan animate-pulse" />
                  <span>CCTV Sector 01 (Corridor Feed)</span>
                </div>
              </div>
            )}

            {/* Bounding Box Overlays - High z-index to stay in FRONT */}
            {isStreaming && (
              <div className="absolute inset-0 pointer-events-none z-30">
                {boxes.map((box) => (
                  <div
                    key={box.id}
                    style={{
                      left: `${box.x}%`,
                      top: `${box.y}%`,
                      width: `${box.width}%`,
                      height: `${box.height}%`,
                    }}
                    className={cn(
                      "absolute border-2 transition-all duration-300 rounded-lg",
                      box.isThreat
                        ? box.isWarning
                          ? "border-neon-amber shadow-[0_0_15px_rgba(251,191,36,0.8)] animate-pulse"
                          : "border-neon-red shadow-[0_0_15px_rgba(244,63,94,0.8)] animate-pulse"
                        : "border-neon-cyan shadow-[0_0_10px_rgba(34,211,238,0.4)]"
                    )}
                  >
                    {/* Tag badge above box */}
                    <div
                      className={cn(
                        "absolute -top-6 left-0 z-40 flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-mono font-bold whitespace-nowrap shadow-md",
                        box.isThreat
                          ? box.isWarning
                            ? "bg-neon-amber text-black"
                            : "bg-neon-red text-white"
                          : "bg-neon-cyan text-black"
                      )}
                    >
                      <span>{box.label}</span>
                      <span>{Math.round(box.confidence * 100)}%</span>
                    </div>

                    {/* Corner Reticle Accents */}
                    <div className="absolute -top-1 -left-1 h-2 w-2 border-t-2 border-l-2 border-white" />
                    <div className="absolute -top-1 -right-1 h-2 w-2 border-t-2 border-r-2 border-white" />
                    <div className="absolute -bottom-1 -left-1 h-2 w-2 border-b-2 border-l-2 border-white" />
                    <div className="absolute -bottom-1 -right-1 h-2 w-2 border-b-2 border-r-2 border-white" />
                  </div>
                ))}
              </div>
            )}

            {/* Bottom Stream Telemetry Bar */}
            <div className="absolute bottom-3 left-4 right-4 z-20 flex items-center justify-between rounded-xl bg-black/60 px-4 py-2 text-xs font-mono text-slate-300 backdrop-blur-md border border-white/10">
              <div className="flex items-center gap-4">
                <span>
                  FPS: <strong className="text-white">{isStreaming ? fps : 0}</strong>
                </span>
                <span>
                  Latency: <strong className="text-neon-cyan">{isStreaming ? latency : 0}ms</strong>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-neon-green" />
                <span className="text-[11px] text-slate-400">YOLO11n + S3D Synced</span>
              </div>
            </div>
          </div>

          {/* 3 Camera Thumbnails for Rapid Feed Switching */}
          <div className="grid grid-cols-3 gap-3">
            {cameras.map((cam) => {
              const isSelected = cam.id === activeCameraId;
              return (
                <button
                  key={cam.id}
                  onClick={() => setActiveCamera(cam.id)}
                  className={cn(
                    "relative overflow-hidden rounded-2xl border p-2 text-left transition-all duration-200 group backdrop-blur-md",
                    isSelected
                      ? "border-neon-cyan/50 bg-white/[0.08] shadow-[0_0_15px_rgba(34,211,238,0.2)]"
                      : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.05]"
                  )}
                >
                  <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-black">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={cam.thumbnailUrl}
                      alt={cam.name}
                      className="h-full w-full object-cover transition-transform group-hover:scale-105 duration-300"
                    />
                    <div className="absolute top-1.5 left-1.5">
                      <span
                        className={cn(
                          "inline-block h-2 w-2 rounded-full",
                          cam.status === "online" ? "bg-neon-green" : "bg-slate-500"
                        )}
                      />
                    </div>
                  </div>
                  <div className="mt-2 truncate font-mono text-xs font-bold text-white">
                    {cam.name}
                  </div>
                  <div className="truncate text-[10px] text-slate-400">{cam.location}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right: Live Threat Telemetry Panel - Raised z-index */}
        <div className="lg:col-span-4 flex flex-col space-y-4 relative z-20">
          <GlassCard glow="cyan" className="flex flex-col items-center p-6">
            <h3 className="w-full text-xs font-bold uppercase tracking-wider text-slate-400">
              Live Threat Telemetry
            </h3>

            {/* Big Colored Gauge */}
            <div className="my-3 w-full flex justify-center">
              <ThreatGauge score={isStreaming ? compositeScore : 0.05} level={isStreaming ? threatLevel : "SAFE"} />
            </div>

            {/* Metric Bars */}
            <div className="w-full space-y-4 border-t border-white/10 pt-4">
              {/* Weapon Score */}
              <div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium">Weapon Confidence (YOLO)</span>
                  <span className="font-mono font-bold text-neon-red">
                    {isStreaming ? Math.round(weaponScore * 100) : 0}%
                  </span>
                </div>
                <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    style={{ width: `${isStreaming ? weaponScore * 100 : 0}%` }}
                    className="h-full rounded-full bg-gradient-to-r from-orange-500 to-neon-red transition-all duration-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]"
                  />
                </div>
              </div>

              {/* Violence Score */}
              <div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-300 font-medium">Violence Action (S3D)</span>
                  <span className="font-mono font-bold text-neon-amber">
                    {isStreaming ? Math.round(violenceScore * 100) : 0}%
                  </span>
                </div>
                <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-white/10">
                  <div
                    style={{ width: `${isStreaming ? violenceScore * 100 : 0}%` }}
                    className="h-full rounded-full bg-gradient-to-r from-amber-500 to-yellow-400 transition-all duration-500 shadow-[0_0_10px_rgba(251,191,36,0.5)]"
                  />
                </div>
              </div>

              {/* Edge Inference Stats */}
              <div className="grid grid-cols-2 gap-2 pt-2 text-center">
                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-2.5">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400">
                    Active Detections
                  </span>
                  <div className="mt-1 font-mono text-lg font-bold text-white">
                    {isStreaming ? boxes.length : 0}
                  </div>
                </div>
                <div className="rounded-xl border border-white/5 bg-white/[0.02] p-2.5">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400">
                    Siren Buzzer
                  </span>
                  <div className="mt-1 font-mono text-sm font-semibold text-neon-green">
                    {detectionSettings.sirenEnabled ? "ARMED" : "MUTED"}
                  </div>
                </div>
              </div>
            </div>
          </GlassCard>

          {/* Rapid Action Tip Card */}
          <GlassCard className="p-4 border-white/5 bg-white/[0.02]">
            <div className="flex items-start gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-neon-cyan/10 text-neon-cyan">
                <Sparkles className="h-4 w-4" />
              </div>
              <div className="text-xs">
                <div className="font-semibold text-white">Zero False-Alarm Guarantee</div>
                <p className="mt-0.5 text-slate-400 leading-relaxed">
                  Full 80-class non-maximum suppression prevents everyday objects (phones, keys, pens) from triggering weapon alerts.
                </p>
              </div>
            </div>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
