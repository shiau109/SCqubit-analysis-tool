import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_drag_alternating(plot_data: xr.Dataset) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    beta = plot_data["beta"].values
    npi = plot_data["nb_of_pulses"].values
    signal = plot_data["signal"].values  # shape (nb_of_pulses, beta)
    
    # 2D Heatmap
    X, Y = np.meshgrid(beta, npi)
    im = ax1.pcolormesh(X, Y, signal, shading="auto", cmap="RdBu_r")
    fig.colorbar(im, ax=ax1, label="Signal (a.u.)")
    ax1.set_title("DRAG 180/-180 Error Amplification")
    ax1.set_xlabel("DRAG Coefficient (alpha/beta)")
    ax1.set_ylabel("Number of Alternating Pulses")
    
    # 1D line cuts at optimal, low, high beta
    opt_beta = float(plot_data.attrs.get("opt_beta", np.nan))
    opt_idx = np.argmin(np.abs(beta - opt_beta)) if np.isfinite(opt_beta) else len(beta) // 2
    
    ax2.plot(npi, signal[:, 0], "o--", color="crimson", label=f"Low beta: {beta[0]:.3f}")
    ax2.plot(npi, signal[:, opt_idx], "o-", color="purple", label=f"Optimal beta: {beta[opt_idx]:.3f}")
    ax2.plot(npi, signal[:, -1], "o--", color="royalblue", label=f"High beta: {beta[-1]:.3f}")
    
    ax2.set_title("Error Accumulation vs Pulse Count")
    ax2.set_xlabel("Number of Alternating Pulses")
    ax2.set_ylabel("Signal (a.u.)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    return fig
