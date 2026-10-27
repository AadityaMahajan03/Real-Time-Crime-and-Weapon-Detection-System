"""FAST-TRACK AI CLI Launcher and Operations Entry Point."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAST-TRACK AI: Multi-Camera Real-Time Weapon & Violence Detection Command Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  ui         Launch Streamlit Security Command Center (default)
  web        Launch Modern Next.js 14 Web App (port 3000)
  api        Launch FastAPI & Real-Time AI Inference Server (port 8000)
  demo       Seed database with demo cameras, incidents, and video artifacts
  test       Execute pytest test suite with coverage
  check      Run ruff linter across all code directories
  benchmark  Run multi-camera system throughput benchmark
  eval       Run weapon and violence held-out evaluation scripts
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="ui",
        choices=["ui", "web", "api", "demo", "test", "check", "benchmark", "eval"],
        help="Command to execute (default: ui)",
    )
    parser.add_argument("--port", type=int, default=8501, help="Port for UI/API (default: 8501)")

    args = parser.parse_args()

    if args.command == "ui":
        print(f"[*] Launching FAST-TRACK AI Security Command Center on port {args.port}...")
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py", f"--server.port={args.port}"]
        sys.exit(subprocess.call(cmd))

    elif args.command == "web":
        print("[*] Launching FAST-TRACK AI Modern Next.js Web App on port 3000...")
        import os
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        cmd = ["npm", "run", "dev"]
        sys.exit(subprocess.call(cmd, cwd=web_dir, shell=True))

    elif args.command == "api":
        port = 8000 if args.port == 8501 else args.port
        print(f"[*] Launching FastAPI Real-Time AI Detection Server on port {port}...")
        cmd = [sys.executable, "-m", "uvicorn", "src.fasttrack.api.main:app", "--port", str(port), "--host", "127.0.0.1"]
        sys.exit(subprocess.call(cmd))

    elif args.command == "demo":
        print("[*] Seeding database with demo cameras, incidents, and video clips...")
        cmd = [sys.executable, "scripts/seed_demo_data.py"]
        sys.exit(subprocess.call(cmd))

    elif args.command == "test":
        print("[*] Running full test suite...")
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
        sys.exit(subprocess.call(cmd))

    elif args.command == "check":
        print("[*] Running ruff linter across src, tests, ui, and scripts...")
        cmd = [sys.executable, "-m", "ruff", "check", "src", "tests", "ui", "scripts"]
        sys.exit(subprocess.call(cmd))

    elif args.command == "benchmark":
        print("[*] Running system multi-camera throughput benchmark...")
        cmd = [sys.executable, "scripts/benchmark_system.py"]
        sys.exit(subprocess.call(cmd))

    elif args.command == "eval":
        print("[*] Running evaluation suite for weapon and violence detectors...")
        subprocess.call([sys.executable, "scripts/eval_weapon.py"])
        subprocess.call([sys.executable, "scripts/eval_violence.py"])
        sys.exit(0)


if __name__ == "__main__":
    main()
