"""Generate LinkedIn post figures for ToxNav."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DARK_BG  = "#0d1117"
CARD_BG  = "#161b22"
BORDER   = "#30363d"
ACCENT1  = "#58a6ff"   # blue
ACCENT2  = "#3fb950"   # green
ACCENT3  = "#f78166"   # red/orange
ACCENT4  = "#d2a8ff"   # purple
ACCENT5  = "#ffa657"   # amber
WHITE    = "#e6edf3"
MUTED    = "#8b949e"


def arrow(ax, x0, y0, x1, y1, color=ACCENT1, lw=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"))


def box(ax, cx, cy, w, h, bg, label, sub="", lc=WHITE):
    patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.04",
                           facecolor=bg, edgecolor=lc,
                           linewidth=1.3, alpha=0.93, zorder=3)
    ax.add_patch(patch)
    ax.text(cx, cy + (0.10 if sub else 0), label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=lc, zorder=4)
    if sub:
        ax.text(cx, cy - 0.19, sub,
                ha="center", va="center", fontsize=7.5, color=MUTED, zorder=4)


# ──────────────────────────────────────────────────────────
# Figure 1 — The Workflow
# ──────────────────────────────────────────────────────────
def fig1_workflow():
    fig, ax = plt.subplots(figsize=(15, 5.8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    ax.text(7.5, 5.4, "ToxNav   Molecular Toxicity Prediction Workflow",
            ha="center", va="center", fontsize=15, fontweight="bold", color=WHITE)

    nodes = [
        (1.3,  2.9, 1.9, 1.15, "#1c2e45", "SMILES Input",    "drug molecule",      ACCENT1),
        (3.85, 2.9, 2.0, 1.15, "#1c2e45", "KPGT Encoder",    "graph featurization",ACCENT4),
        (6.4,  2.9, 2.0, 1.15, "#2a1a1a", "Toxicity Scores", "30 endpoints",       ACCENT3),
        (8.95, 2.9, 2.0, 1.15, "#1a2440", "GPT-4o Agent",    "agentic reasoning",  ACCENT2),
        (11.5, 2.9, 2.0, 1.15, "#1c2e45", "ChEMBL Search",   "analog retrieval",   ACCENT5),
        (13.8, 2.9, 2.0, 1.15, "#163020", "Safer Alts",      "ranked output",      ACCENT2),
    ]
    for cx, cy, w, h, bg, lbl, sub, lc in nodes:
        box(ax, cx, cy, w, h, bg, lbl, sub, lc)

    for i in range(len(nodes) - 1):
        x0 = nodes[i][0]   + nodes[i][2]/2   + 0.05
        x1 = nodes[i+1][0] - nodes[i+1][2]/2 - 0.05
        arrow(ax, x0, 2.9, x1, 2.9, color=ACCENT1, lw=2)

    tech = ["RDKit", "DGL + PyTorch", "TOXRIC labels", "Azure OpenAI", "REST API", "KPGT re-score"]
    for (cx, _, _, _, _, _, _, _), t in zip(nodes, tech):
        ax.text(cx, 1.75, t, ha="center", va="center", fontsize=8, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=CARD_BG,
                          edgecolor=BORDER, linewidth=0.8))

    ax.text(7.5, 0.55,
            "Built with   PyTorch   DGL   LangGraph   Streamlit   Azure Container Apps",
            ha="center", va="center", fontsize=9, color=MUTED)

    plt.tight_layout(pad=0.3)
    plt.savefig("figures/fig1_workflow.jpg", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("Saved fig1_workflow.jpg")


# ──────────────────────────────────────────────────────────
# Figure 2 — Agentic Loop (LangGraph) — accurate graph
# ──────────────────────────────────────────────────────────
def fig2_langgraph():
    fig, ax = plt.subplots(figsize=(14, 10.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0.2, 10.5)
    ax.axis("off")

    ax.text(7.0, 10.15, "ToxNav   Agentic Loop   LangGraph StateGraph",
            ha="center", va="center", fontsize=14, fontweight="bold", color=WHITE)

    # Layout:
    #   TOP ROW  (y=8.0): START → check_toxric → predict_kpgt → run_llm_validation
    #   MID ROW  (y=5.6):                                         pick_best
    #   LOW ROW  (y=3.4): END ← explain
    #
    # check_toxric and explain share the same x=2.6 → clean vertical drop for TOXRIC match.
    # run_llm_validation and pick_best share the same x=11.2 → clean vertical drop for "unsure".
    # Loop-back arc from pick_best sweeps ABOVE the top row (arc3 rad=+0.38).
    # No connecting arrow from validate to inner-graph box — avoids double-line.

    CX_L = 3.1    # check_toxric / explain / END  (moved right to clear START)
    CX_M = 7.0    # predict_kpgt
    CX_R = 11.4   # run_llm_validation / pick_best
    Y_T  = 8.0    # top row
    Y_M  = 5.6    # pick_best
    Y_E  = 3.4    # explain
    Y_N  = 1.7    # END

    # (cx, cy, w, h, bg, label, sub, lc)
    N = {
        "start":    (0.75, Y_T, 1.1,  0.85, "#163020", "START",             "",                   ACCENT2),
        "check":    (CX_L, Y_T, 2.6,  0.85, "#1c2e45", "check_toxric",      "TOXRIC lookup",      ACCENT1),
        "predict":  (CX_M, Y_T, 2.6,  0.85, "#1a2440", "predict_kpgt",      "KPGT inference",     ACCENT4),
        "validate": (CX_R, Y_T, 2.8,  0.85, "#2a1a1a", "run_llm_validation","GPT-4o + tools",     ACCENT3),
        "pick":     (CX_R, Y_M, 2.8,  0.85, "#252020", "pick_best",         "rank by similarity", ACCENT5),
        "explain":  (CX_L, Y_E, 2.6,  0.85, "#1a2440", "explain",           "GPT-4o narration",   ACCENT4),
        "end":      (CX_L, Y_N, 1.1,  0.85, "#163020", "END",               "",                   ACCENT2),
    }
    for k, (cx, cy, w, h, bg, lbl, sub, lc) in N.items():
        box(ax, cx, cy, w, h, bg, lbl, sub, lc)

    # Arrow helper — label gets a dark pill background so it sits cleanly on lines
    def A(x0, y0, x1, y1, color=ACCENT1, rad=0.0, label="", lx=None, ly=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.9,
                                   connectionstyle=f"arc3,rad={rad}"))
        if label:
            tx = lx if lx is not None else (x0+x1)/2
            ty = ly if ly is not None else (y0+y1)/2
            ax.text(tx, ty, label, fontsize=8, color=color,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=DARK_BG,
                              edgecolor="none", alpha=0.9))

    hw = 0.43   # half-height offset to hit node edge

    # ── Top row: forward flow ─────────────────────────────────────────
    A(N["start"][0]+0.55, Y_T, CX_L-1.3,  Y_T, color=ACCENT2)
    A(CX_L+1.3,  Y_T, CX_M-1.3,  Y_T, color=ACCENT1,
      label="no match", lx=(CX_L+CX_M)/2, ly=Y_T+0.32)
    A(CX_M+1.3,  Y_T, CX_R-1.4,  Y_T, color=ACCENT1)

    # ── validate → pick_best: ONE clean vertical line ─────────────────
    A(CX_R, Y_T-hw, CX_R, Y_M+hw, color=ACCENT3,
      label="unsure", lx=CX_R+0.62, ly=(Y_T+Y_M)/2)

    # ── validate → explain: diagonal, curved slightly right to stay clear ─
    A(CX_R-1.4, Y_T, CX_L+1.3, Y_E,
      color=ACCENT2, rad=0.18,
      label="confident / degrading", lx=6.5, ly=6.5)

    # ── check_toxric → explain: straight vertical ─────────────────────
    A(CX_L, Y_T-hw, CX_L, Y_E+hw, color=ACCENT2,
      label="TOXRIC match", lx=CX_L+1.05, ly=(Y_T+Y_E)/2)

    # ── pick_best → explain: diagonal ────────────────────────────────
    A(CX_R-1.4, Y_M, CX_L+1.3, Y_E,
      color=ACCENT3, rad=-0.15,
      label="max iter", lx=7.8, ly=4.35)

    # ── THE LOOP: pick_best → check_toxric ────────────────────────────
    # Arc from right side of pick_best to TOP of check_toxric.
    # rad=+0.38 bows above the top row; check_toxric is now well clear of START.
    A(CX_R+1.4, Y_M, CX_L, Y_T+hw,
      color=ACCENT5, rad=0.38,
      label="continue  (new candidate — LOOP BACK)", lx=7.2, ly=9.35)

    # ── explain → END ────────────────────────────────────────────────
    A(CX_L, Y_E-hw, CX_L, Y_N+hw, color=ACCENT2)

    # ── Inner graph inset (no connecting arrow to avoid double-line) ──
    ix, iy, iw, ih = 7.5, 2.55, 5.8, 2.4
    ax.add_patch(FancyBboxPatch((ix, iy-ih/2), iw, ih,
                                boxstyle="round,pad=0.1",
                                facecolor="#0d1520", edgecolor=ACCENT4,
                                linewidth=1.1, linestyle="--", alpha=0.9, zorder=2))
    ax.text(ix+iw/2, iy+ih/2-0.22,
            "Inner graph  (called inside run_llm_validation)",
            ha="center", va="center", fontsize=8.5, color=ACCENT4, fontweight="bold")
    IN = {
        "illm":   (ix+1.15, iy-0.05, 1.7, 0.78, "#1a1a30", "llm",   "AzureChatOpenAI",     ACCENT4),
        "itools": (ix+3.75, iy-0.05, 1.8, 0.78, "#1a2440", "tools", "validate / suggest",  ACCENT1),
    }
    for k, (cx, cy, w, h, bg, lbl, sub, lc) in IN.items():
        box(ax, cx, cy, w, h, bg, lbl, sub, lc)
    mx = (IN["illm"][0]+IN["itools"][0])/2
    A(IN["illm"][0]+0.85,  iy-0.05, IN["itools"][0]-0.9, iy-0.05,
      color=ACCENT1, label="tool call", lx=mx, ly=iy+0.27)
    A(IN["itools"][0]-0.9, iy+0.18, IN["illm"][0]+0.85,  iy+0.18,
      color=ACCENT4, rad=-0.55,
      label="result", lx=mx, ly=iy+0.73)
    ax.text(ix+0.4,     iy-ih/2+0.25, "START", fontsize=7.5, color=ACCENT2)
    ax.text(ix+iw-0.45, iy-ih/2+0.25, "END",   fontsize=7.5, color=ACCENT2, ha="right")

    # ── Legend ───────────────────────────────────────────────────────
    legend = [(ACCENT2, "Normal / satisfied"),
              (ACCENT3, "Unsure / max iterations"),
              (ACCENT5, "Loop back — new candidate")]
    for i, (c, t) in enumerate(legend):
        lx = 0.4 + i * 4.5
        ax.plot([lx, lx+0.5], [0.6, 0.6], color=c, lw=2.2)
        ax.text(lx+0.65, 0.6, t, fontsize=8.5, color=c, va="center")

    plt.tight_layout(pad=0.3)
    plt.savefig("figures/fig2_langgraph.jpg", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("Saved fig2_langgraph.jpg")


# ──────────────────────────────────────────────────────────
# Figure 3 — Production Cloud Stack
# ──────────────────────────────────────────────────────────
def fig3_cloud():
    fig, ax = plt.subplots(figsize=(13.5, 8.4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-1.0, 7.8)
    ax.axis("off")

    ax.text(6.75, 7.4, "ToxNav   Production Cloud Stack on Azure",
            ha="center", va="center", fontsize=14, fontweight="bold", color=WHITE)

    # Lane background bands
    lane_colors = ["#0f1e30", "#101a10", "#0d1a0d", "#17101a"]
    lane_ys = [5.1, 3.45, 1.85, 0.25]
    lane_h  = [1.35, 1.35, 1.35, 1.35]
    lane_labels = ["TRAIN", "BUILD", "DEPLOY", "SECURE"]
    for ly, lh, llbl, lc in zip(lane_ys, lane_h, lane_labels, lane_colors):
        patch = FancyBboxPatch((0.9, ly - lh/2), 12.0, lh,
                               boxstyle="round,pad=0.1",
                               facecolor=lc, edgecolor=BORDER,
                               linewidth=0.8, alpha=0.6, zorder=1)
        ax.add_patch(patch)
        ax.text(0.55, ly, llbl, ha="center", va="center",
                fontsize=8, fontweight="bold", color=MUTED, rotation=90)

    def row(nodes, y, color):
        for cx, w, lbl, sub in nodes:
            box(ax, cx, y, w, 0.92, "#0d1117", lbl, sub, color)
        for i in range(len(nodes) - 1):
            x0 = nodes[i][0]   + nodes[i][1]/2   + 0.06
            x1 = nodes[i+1][0] - nodes[i+1][1]/2 - 0.06
            arrow(ax, x0, y, x1, y, color=color, lw=1.8)

    train = [
        (2.4,  2.3, "Azure ML Workspace", "ToxicityKpgt / eastus"),
        (5.3,  2.2, "GPU Cluster",        "Standard_NC4as_T4_v3"),
        (8.15, 2.0, "train.py",           "KPGT fine-tune"),
        (11.0, 2.0, "best.pt",            "model checkpoint"),
    ]
    build = [
        (2.4,  2.1, "Dockerfile",         "python:3.11-slim"),
        (5.3,  2.1, "az acr build",       "remote AMD64 build"),
        (8.15, 2.3, "Container Registry", "ACR  toxnav:latest"),
        (11.0, 2.0, "Docker Image",       "~2 GB  CPU-only"),
    ]
    deploy = [
        (2.9,  2.3, "Container Apps",  "toxnav-env"),
        (5.9,  2.2, "Streamlit App",   "port 8501"),
        (8.9,  2.4, "Public HTTPS",    "*.azurecontainerapps.io"),
        (11.7, 1.8, "Scale-to-zero",   "0 to 1 replicas"),
    ]
    secure = [
        (2.9,  2.3, "Managed Identity", "System-assigned"),
        (5.9,  2.2, "Azure OpenAI",     "gpt-4o endpoint"),
        (8.9,  2.1, "RBAC Role",        "Cognitive Svcs User"),
        (11.7, 1.8, "No Secrets",       "zero creds in image"),
    ]

    row([(cx, w, lbl, sub) for cx, w, lbl, sub in train],  5.1, ACCENT1)
    row([(cx, w, lbl, sub) for cx, w, lbl, sub in build],  3.45, ACCENT5)
    row([(cx, w, lbl, sub) for cx, w, lbl, sub in deploy], 1.85, ACCENT2)
    row([(cx, w, lbl, sub) for cx, w, lbl, sub in secure], 0.25, ACCENT4)

    # Gap notes between lanes (replace cross-lane arrows with clean inline text)
    ax.text(6.75, 4.275, "best.pt bundled into Docker image",
            ha="center", va="center", fontsize=8, color=MUTED, style="italic")
    ax.text(6.75, 2.675, "image pulled from ACR at deploy time",
            ha="center", va="center", fontsize=8, color=MUTED, style="italic")

    ax.text(6.75, -0.65,
            "Azure ML   Azure Container Registry   Azure Container Apps   Azure OpenAI",
            ha="center", va="center", fontsize=9, color=MUTED)

    plt.tight_layout(pad=0.3)
    plt.savefig("figures/fig3_cloud.jpg", dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("Saved fig3_cloud.jpg")


os.makedirs("figures", exist_ok=True)
fig1_workflow()
fig2_langgraph()
fig3_cloud()
print("All figures saved to figures/")
