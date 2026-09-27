"""Plot outputs/all_logs.json and outputs/summary.json into outputs/*.png."""
import json
import os

import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
logs = json.load(open(os.path.join(OUT, "all_logs.json")))
summ = json.load(open(os.path.join(OUT, "summary.json")))

COL = {"dense": "#2a78d6", "dense-continued": "#2a78d6", "moe-copy": "#eb6834", "moe-drop": "#1baf7a"}
LABEL = {"dense-continued": "dense, continued", "moe-copy": "MoE, sparse upcycle (copy)",
         "moe-drop": "MoE, drop-upcycle (r=0.5)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.8})
switch = summ["config"]["dense_steps"]

# ---------------------------------------------------------------- loss curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, zoom in zip(axes, (False, True)):
    d = logs["dense"]
    if not zoom:
        ax.plot([r["step"] for r in d], [r["val"] for r in d], color=COL["dense"], lw=2,
                label="dense phase (from scratch)")
    for name in ("dense-continued", "moe-copy", "moe-drop"):
        r = logs[name]
        ax.plot([x["step"] for x in r], [x["val"] for x in r], color=COL[name], lw=2,
                ls="--" if name == "dense-continued" else "-", marker="o", ms=4,
                label=f"{LABEL[name]}  (final {r[-1]['val']:.3f})")
    ax.axvline(switch, color="#8a8984", lw=1, ls=":")
    ax.text(switch, ax.get_ylim()[1], " dense → MoE conversion", va="top", fontsize=9, color="#52514e")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss (nats/char)")
    ax.set_title("Full run" if not zoom else "After conversion (zoomed)", loc="left")
    ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "loss_curves.png"), dpi=150)

# ---------------------------------------------------------------- routing health
n_layer = summ["config"]["n_layer"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
for ax, name in zip(axes[:2], ("moe-copy", "moe-drop")):
    r = [x for x in logs[name] if x.get("maxvio") and x["maxvio"][0] is not None]
    steps = [x["step"] for x in r]
    shades = ["#9ec5f4", "#5b9be6", "#2a78d6", "#154a8a"]  # one hue, light→dark = layer depth
    for L in range(n_layer):
        ax.plot(steps, [x["maxvio"][L] for x in r], color=shades[L], lw=2, label=f"layer {L}")
    ax.set_title(f"{LABEL[name]}: MaxVio", loc="left")
    ax.set_xlabel("training step")
    ax.set_ylabel("MaxVio  (0 = perfectly even)")
    ax.legend(frameon=False, fontsize=9)

spec = summ["moe-drop_specialisation_last_layer"]
classes = list(spec)
im = axes[2].imshow([spec[c] for c in classes], cmap="Blues", vmin=0, aspect="auto")
axes[2].set_yticks(range(len(classes)), classes)
axes[2].set_xticks(range(len(spec[classes[0]])), [f"E{i}" for i in range(len(spec[classes[0]]))])
axes[2].grid(False)
for i, c in enumerate(classes):
    for j, v in enumerate(spec[c]):
        axes[2].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if v > 0.5 else "#0b0b0b")
axes[2].set_title("drop-upcycle, last layer: share of each char class → top-1 expert", loc="left", fontsize=10)
fig.colorbar(im, ax=axes[2], fraction=0.04)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "routing.png"), dpi=150)
print("saved loss_curves.png and routing.png")
