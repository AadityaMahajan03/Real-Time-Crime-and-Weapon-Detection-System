"""Generate local verified playable H.264 video clips and snapshots for all incidents."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.fasttrack.storage.video_encoder import FFmpegVideoEncoder

def main():
    encoder = FFmpegVideoEncoder()
    clips_dir = Path("web/public/clips")
    evidence_dir = Path("web/public/evidence")
    data_clips_dir = Path("data/clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    data_clips_dir.mkdir(parents=True, exist_ok=True)

    incidents_config = [
        {
            "id": "inc-1094",
            "cam": "CAM-01 [SECTOR 4 SOUTH]",
            "threat": "KNIFE 92%",
            "color": (0, 0, 255),
            "bg": (25, 30, 35),
        },
        {
            "id": "inc-1093",
            "cam": "CAM-02 [GATE ALPHA]",
            "threat": "FIREARM DRAW 95%",
            "color": (0, 0, 255),
            "bg": (20, 25, 40),
        },
        {
            "id": "inc-1092",
            "cam": "CAM-01 [CORRIDOR B]",
            "threat": "FIGHTING 74%",
            "color": (0, 165, 255),
            "bg": (30, 25, 25),
        },
        {
            "id": "inc-1091",
            "cam": "CAM-03 [LOADING BAY]",
            "threat": "BOX CUTTER 44%",
            "color": (0, 220, 255),
            "bg": (25, 35, 25),
        },
        {
            "id": "inc-1090",
            "cam": "CAM-02 [GATE ALPHA]",
            "threat": "HOLSTER SIDEARM 88%",
            "color": (0, 0, 255),
            "bg": (20, 20, 30),
        },
    ]

    w, h = 640, 360

    for cfg in incidents_config:
        frames = []
        inc_id = cfg["id"]
        for i in range(60):  # 2.4 seconds at 25 fps
            f = np.zeros((h, w, 3), dtype=np.uint8)
            f[:] = cfg["bg"]
            # CCTV grid & timestamp
            cv2.line(f, (0, int(h / 2)), (w, int(h / 2)), (40, 45, 50), 1)
            cv2.line(f, (int(w / 2), 0), (int(w / 2), h), (40, 45, 50), 1)
            header_text = f"{cfg['cam']} - 2026-10-25 11:45:02 UTC"
            cv2.putText(f, header_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

            # Target subject
            tx = int(240 + i * 2)
            ty = int(100 + np.sin(i * 0.2) * 5)
            cv2.rectangle(f, (tx, ty), (tx + 80, ty + 170), (100, 120, 140), -1)

            # Threat reticle
            cv2.rectangle(f, (tx + 25, ty + 35), (tx + 75, ty + 95), cfg["color"], 2)
            cv2.putText(f, cfg["threat"], (tx + 5, max(20, ty + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg["color"], 2)
            frames.append(f)

        # Output to both web/public/clips and data/clips
        out_clip = clips_dir / f"{inc_id}.mp4"
        encoder.encode_h264_clip(frames, out_clip, fps=25.0)

        data_clip = data_clips_dir / f"{inc_id}.mp4"
        encoder.encode_h264_clip(frames, data_clip, fps=25.0)

        # Save snapshot frame
        snap_file = evidence_dir / f"{inc_id}.jpg"
        cv2.imwrite(str(snap_file), frames[25])
        print(f"Generated clip & snapshot for {inc_id} ({out_clip.stat().st_size} bytes)")

    print("All local incident clips and snapshots created successfully!")

if __name__ == "__main__":
    main()
