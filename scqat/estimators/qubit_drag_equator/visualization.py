import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_drag_equator(plot_data: xr.Dataset) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    beta = plot_data["beta"].values
    y0 = plot_data["seq0"].values
    y1 = plot_data["seq1"].values
    y2 = plot_data["seq2"].values
    
    ax.plot(beta, y0, "o-", color="royalblue", label="Seq 0: X90-(Y180)^N")
    ax.plot(beta, y1, "o-", color="crimson", label="Seq 1: X90-(-Y180)^N")
    ax.plot(beta, y2, "o-", color="forestgreen", label="Seq 2: X90-(X180)^N (Ref)")
    
    opt_beta = float(plot_data.attrs.get("opt_beta", np.nan))
    if np.isfinite(opt_beta):
        ax.axvline(opt_beta, linestyle="--", color="purple", label=f"Optimum beta: {opt_beta:.4f}")
        
    ax.set_title("DRAG Parameter Calibration (3-Line Equator)")
    ax.set_xlabel("DRAG beta Coefficient")
    ax.set_ylabel("Signal (a.u.)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    return fig
