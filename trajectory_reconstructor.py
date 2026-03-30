import pandas as pd
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "./Mirla_YOLO_test_videos/unseen_videos/GroupExample/250825_10session_site1_1_tracked.csv"
OUTPUT_JSON = "trajectories.json"

SMOOTH_WINDOW = 5       # moving average window
MIN_TRAJ_LENGTH = 5     # filter short tracks

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# CLEAN DATA
# -----------------------------
# Remove invalid tracks
df = df[df["track_id"] != -1]

# Sort properly
df = df.sort_values(by=["track_id", "frame"])

# -----------------------------
# COMPUTE CENTER POINTS
# -----------------------------
df["cx"] = (df["x1"] + df["x2"]) / 2
df["cy"] = (df["y1"] + df["y2"]) / 2

# -----------------------------
# BUILD TRAJECTORIES
# -----------------------------
trajectories = defaultdict(list)

for _, row in df.iterrows():
    tid = int(row["track_id"])

    trajectories[tid].append({
        "frame": int(row["frame"]),
        "time": float(row["time_sec"]),
        "x": float(row["cx"]),
        "y": float(row["cy"])
    })

# -----------------------------
# SMOOTHING FUNCTION
# -----------------------------
def smooth_trajectory(points, window=5):
    if len(points) < window:
        return points

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])

    kernel = np.ones(window) / window

    xs_smooth = np.convolve(xs, kernel, mode='same')
    ys_smooth = np.convolve(ys, kernel, mode='same')

    for i in range(len(points)):
        points[i]["x"] = float(xs_smooth[i])
        points[i]["y"] = float(ys_smooth[i])

    return points

# -----------------------------
# VELOCITY COMPUTATION
# -----------------------------
def compute_velocity(points):
    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i-1]["x"]
        dy = points[i]["y"] - points[i-1]["y"]
        dt = points[i]["time"] - points[i-1]["time"]

        if dt > 0:
            vx = dx / dt
            vy = dy / dt
            speed = np.sqrt(vx**2 + vy**2)
        else:
            vx, vy, speed = 0, 0, 0

        points[i]["vx"] = float(vx)
        points[i]["vy"] = float(vy)
        points[i]["speed"] = float(speed)

    # first point has no velocity
    points[0]["vx"] = 0.0
    points[0]["vy"] = 0.0
    points[0]["speed"] = 0.0

    return points

# -----------------------------
# PROCESS TRAJECTORIES
# -----------------------------
processed_trajectories = []

for tid, points in trajectories.items():

    # skip very short tracks
    if len(points) < MIN_TRAJ_LENGTH:
        continue

    # smooth
    points = smooth_trajectory(points, SMOOTH_WINDOW)

    # compute velocity
    points = compute_velocity(points)

    processed_trajectories.append({
        "track_id": tid,
        "num_points": len(points),
        "trajectory": points
    })

print(f"Final trajectories: {len(processed_trajectories)}")

# -----------------------------
# SAVE TO JSON
# -----------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(processed_trajectories, f, indent=2)

print(f"Saved to {OUTPUT_JSON}")

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(8, 6))

for traj in processed_trajectories:
    xs = [p["x"] for p in traj["trajectory"]]
    ys = [p["y"] for p in traj["trajectory"]]

    plt.plot(xs, ys, linewidth=1)

plt.gca().invert_yaxis()  # match image coordinates
plt.title("Reconstructed Trajectories")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()