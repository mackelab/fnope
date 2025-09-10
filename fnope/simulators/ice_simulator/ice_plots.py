import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch


color_opts = {
    "colors": {
        "prior": "#069d3f",  # green prior
        #"posterior": "#140289",  # blue posterior
        "posterior": "#9b2226",  # orange posterior
        #"observation": "#a90505",  # red for observations/true values
        "observation":"#140289", #dark red for observations/true values
        #"boundary_condition": "#825600",  # brown for boundary conditions
        "boundary_condition": "black",  # brown for boundary conditions
        "contrast1": "#808080",
        "contrast2": "#000000",
    },
    "color_maps": {
        "ice": mpl.cm.get_cmap("YlGnBu"),
        "age": mpl.cm.get_cmap("magma"),
        "prior_pairplot": mpl.cm.get_cmap("Blues"),
        "posterior_pairplot": mpl.cm.get_cmap("Reds"),
        "noise": mpl.cm.get_cmap("tab10"),
    },
    "color_cycles": {
        "standard": plt.rcParams["axes.prop_cycle"].by_key()["color"],
    },
}

prior_alpha = 0.175
posterior_alpha = 0.3
samples_alpha = 0.15


def plot_posterior_nice(
    x: np.ndarray,
    mb_mask: np.ndarray,
    tmb: np.ndarray,
    posterior_smb_samples: torch.Tensor,
    layer_mask: np.ndarray,
    LMI_boundary: float,
    posterior_layer_samples: np.ndarray,
    posterior_layer_ages: np.ndarray,
    true_layer: np.ndarray,
    shelf_base: np.ndarray,
    shelf_surface: np.ndarray,
    true_smb=None,
    true_age=None,
    plot_samples=False,
    title=None,
    plot_only_predictive=False,
    figsize=(8, 6),
):
    """
    Plot the posterior SMB (top) and BMB (bottom) values, along with the posterior predictive (layers, middle)

    Args:
        x (ndarray): Discretization of flowline domain.
        mb_mask (ndarray): Boolean mask indicating the locations where SMB is inferred
        tmb (ndarray): Fixed total mass balance values [m/a].
        posterior_smb_samples (ndarray): Posterior SMB samples.
        layer_mask (ndarray): Boolean mask indicating the locations where the layer elevations were used for training.
        LMI_boudnary (float): LMI boundary for this layer.
        posterior_layer_samples (ndarray): Posterior layer samples.
        posterior_layer_ages (ndarray):  Posterior layer ages.
        true_layer (ndarray): True layer elevations.
        shelf_base (ndarray): Shelf base elevations.
        shelf_surface (ndarray): Shelf surface elevations.
        true_smb (ndarray, optional): True SMB values.
        true_age (float, optional): Age of GT layer (if known).
        plot_samples (bool, optional): Flag indicating whether to plot individual samples. Defaults to False.
        title (str, optional): Title of the plot. Defaults to None.

    Returns:
        None
    """
    if true_age is not None:
        x_label = "Distance along flowline [km]"
        gt_label = "GT"
    else:
        x_label = "Distance from GL [km]"
        gt_label = "Observed"

    ax_label_loc = -0.12

    # percentiles are approx. 2sigma
    percentiles = [5, 95]
    # age percentiles are approx. 1sigma
    age_percentiles = [16, 84]

    # Calculate prior and posterior mean and quantiles
    post_mean_smb = torch.mean(posterior_smb_samples, axis=0)
    post_uq_smb = torch.quantile(posterior_smb_samples, percentiles[1] / 100, axis=0)
    post_lq_smb = torch.quantile(posterior_smb_samples, percentiles[0] / 100, axis=0)

    # Calculate prior and posterior layer elevation mean and quantiles
    post_layer_mean = np.mean(posterior_layer_samples, axis=0)
    post_uq_layer = np.quantile(posterior_layer_samples, percentiles[1] / 100, axis=0)
    post_lq_layer = np.quantile(posterior_layer_samples, percentiles[0] / 100, axis=0)

    # Calculate prior and posterior layer age mean and quantiles
    posterior_age_median = np.quantile(posterior_layer_ages, 0.5)
    posterior_age_uq = np.quantile(posterior_layer_ages, age_percentiles[1] / 100)
    posterior_age_lq = np.quantile(posterior_layer_ages, age_percentiles[0] / 100)

    if not plot_only_predictive:
        fig, axs = plt.subplots(
            3, 1, sharex=True, gridspec_kw={"height_ratios": [1, 4, 1]}
        )
    else:
        fig, axs = plt.subplots(1, 1, figsize=figsize)

    if not plot_only_predictive:
        # First axis plots the prior and posterior SMB distributions
        ax = axs[0]
        ax.plot(
            x[mb_mask] / 1e3,
            post_mean_smb,
            color=color_opts["colors"]["posterior"],
            label="Posterior Mean",
            linewidth=1.5,
        )
        if plot_samples:
            for i in range(20):
                ax.plot(
                    x[mb_mask] / 1e3,
                    posterior_smb_samples[i],
                    color=color_opts["colors"]["posterior"],
                    alpha=samples_alpha,
                    linewidth=0.5,
                )
        else:
            ax.fill_between(
                x[mb_mask] / 1e3,
                post_lq_smb,
                post_uq_smb,
                color=color_opts["colors"]["posterior"],
                alpha=posterior_alpha,
                linewidth=0.0,
            )

        ax.set_ylabel("$\dot{a}$ [m/a]")

        # Plot real smb if possible
        if true_smb is not None:
            true_smb_mask = np.where(x < x[mb_mask][-1] + 1e-3)
            ax.plot(
                x[true_smb_mask] / 1e3,
                true_smb[true_smb_mask],
                color=color_opts["colors"]["observation"],
                linewidth=1.0,
                label="True SMB",
            )

        ax.text(
            ax_label_loc,
            0.95,
            "a",
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            ha="right",
        )

    # Second axis plots the prior and posterior predictive distributions
    if not plot_only_predictive:
        ax = axs[1]
    else:
        ax = axs

    # We split the axis into two to show the layers in higher detail
    split_layers = True
    if split_layers:
        divider = make_axes_locatable(ax)
        ax_ratio = 4.0
        ax2 = divider.new_vertical(size=str(100 * ax_ratio) + "%", pad=0.05)
        fig.add_axes(ax2)
        ax1s = [ax, ax2]
        ax.set_ylim(np.min(shelf_base), np.max(shelf_base) + 5)
        dist = np.mean(shelf_surface[layer_mask] - post_layer_mean)
        ax2.set_ylim(np.min((post_layer_mean - 0.4 * dist)), np.max(shelf_surface))
        ax2.tick_params(bottom=False, labelbottom=False)
        ax2.spines[["bottom", "top", "right"]].set_visible(False)
        
        ax2.set_ylabel('Elevation [m.a.s.l]')

        ax2.yaxis.set_label_coords(ax_label_loc, 0.35, transform=ax2.transAxes)
        loc = plticker.MultipleLocator(base=20.0)
        ax.xaxis.set_major_locator(loc)
        if not plot_only_predictive:
            ax2.text(
                ax_label_loc,
                0.95,
                "b",
                transform=ax2.transAxes,
                fontsize=12,
                va="top",
                ha="right",
            )
        d = 0.01
        kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False, linewidth=1.0)
        ax2.plot((-d, +d), (-d, +d), **kwargs)
        kwargs.update(transform=ax.transAxes)
        ax.plot((-d, +d), (1 - d * ax_ratio, 1 + d * ax_ratio), **kwargs)

    else:
        ax1s = [ax]
    for axi in ax1s:
        # First, plot the shelf surface and base
        (s1,) = axi.plot(x / 1e3, shelf_surface, color="black", linewidth=1.0)
        (s2,) = axi.plot(x / 1e3, shelf_base, color="black", linewidth=1.0)
        s3 = axi.fill_between(
            x / 1e3,
            shelf_surface,
            shelf_base,
            color="black",
            alpha=0.075,
            linewidth=0.0,
        )
        (po1,) = axi.plot(
            x[layer_mask] / 1e3,
            post_layer_mean,
            color=color_opts["colors"]["posterior"],
            zorder=5,
            linewidth=0.8,
        )
        # annotate posterior layer age
        # axi.annotate(
        #     r"age = ${0:.0f}^{{+{1:.0f}}}_{{-{2:.0f}}}$ a".format(
        #         posterior_age_median,
        #         posterior_age_uq - posterior_age_median,
        #         posterior_age_median - posterior_age_lq,
        #     ),
        #     xy=(x[layer_mask][0] / 1e3, post_layer_mean[0]),
        #     xycoords="data",
        #     textcoords="offset points",
        #     xytext=(10, -40),
        #     color=color_opts["colors"]["posterior"],
        # )

        if plot_samples:
            for i in range(20):
                axi.plot(
                    x[layer_mask] / 1e3,
                    posterior_layer_samples[i],
                    color=color_opts["colors"]["posterior"],
                    alpha=samples_alpha,
                    linewidth=0.5,
                )
        else:
            po2 = axi.fill_between(
                x[layer_mask] / 1e3,
                post_lq_layer,
                post_uq_layer,
                color=color_opts["colors"]["posterior"],
                alpha=posterior_alpha,
                zorder=5,
                linewidth=0.0,
            )

        (ob1,) = axi.plot(
            x[layer_mask] / 1e3,
            true_layer[layer_mask],
            color=color_opts["colors"]["observation"],
            zorder=10,
            linewidth=1.5,
        )

        if true_age is not None:
            nx = len(x[layer_mask])
            axi.annotate(
                r"true age = {:.0f} a".format(true_age),
                xy=(x[layer_mask][0] / 1e3, post_layer_mean[0]),
                xycoords="data",
                textcoords="offset points",
                xytext=(0, 25),
                color=color_opts["colors"]["observation"],
            )

        # Draw vertical line at LMI boundary
        axi.axvline(
            x=LMI_boundary / 1e3,
            color=color_opts["colors"]["boundary_condition"],
            linestyle="--",
            label="LMI boundary",
            linewidth=1.5,
        )

    # labels = ["Shelf","Prior","Posterior",gt_label]
    labels = ["Ice shelf", gt_label,"Posterior" ]

    ax2.legend(
        handles=[(s1, s3), (ob1,), (po1, po2)],
        labels=labels,
        bbox_to_anchor=(0.0, 0, 0.95, 1.1),
        loc="upper right",
        ncol=2,
    )
    ax.set_yticks([-750,-250])
    ax.set_xlabel("Distance from grounding line [km]")

    if not plot_only_predictive:
        # Thid axis = plot prior and posterior for BMB
        bmb_vals = (post_mean_smb - tmb[mb_mask] < 10).numpy()
        bmb_mask = np.zeros_like(mb_mask, dtype=bool)
        bmb_mask[mb_mask] = bmb_vals
        ax = axs[2]

        # Could similarly split the bmb axis into two
        split_bmb = False
        if split_bmb:
            divider = make_axes_locatable(ax)
            ax_ratio = 1.0 / 2.0
            ax2 = divider.new_vertical(size=str(100 * ax_ratio) + "%", pad=0.05)
            fig.add_axes(ax2)
            ax1s = [ax, ax2]
            ax.set_ylim(-0.5, 1.55)
            dist = np.mean(shelf_surface[layer_mask] - post_layer_mean)
            ax2.set_ylim(1.55, 5.0)
            ax2.set_yticks((2, 5))
            ax2.tick_params(bottom=False, labelbottom=False)
            ax2.spines[["bottom", "top", "right"]].set_visible(False)

            ax.set_ylabel("$\dot{b}$ [m/a]")
            ax.yaxis.set_label_coords(ax_label_loc + 0.05, 0.85, transform=ax.transAxes)
            loc = plticker.MultipleLocator(base=20.0)
            ax.xaxis.set_major_locator(loc)
            ax2.text(
                ax_label_loc,
                1.2,
                "c",
                transform=ax2.transAxes,
                fontsize=12,
                va="top",
                ha="right",
            )

            d = 0.01
            kwargs = dict(
                transform=ax2.transAxes, color="k", clip_on=False, linewidth=1.0
            )
            ax2.plot((-d, +d), (-10 * d, +10 * d), **kwargs)
            kwargs.update(transform=ax.transAxes)
            ax.plot((-d, +d), (1 - 10 * d * ax_ratio, 1 + 10 * d * ax_ratio), **kwargs)
            ax.set_xlabel(x_label)
            ax.spines["bottom"].set_bounds(x[0] / 1e3 - 0.001, x[-1] / 1e3)

        else:
            ax1s = [ax]
            ax.set_ylabel("$\dot{b}$ [m/a]")

            ax.set_xlabel(x_label)
            ax.text(
                ax_label_loc,
                0.95,
                "c",
                transform=ax.transAxes,
                fontsize=12,
                va="top",
                ha="right",
            )
            ax.spines["bottom"].set_bounds(x[0] / 1e3 - 0.001, x[-1] / 1e3)
            # ax.set_ylim(-0.5,1.5)
            # ax.set_yticks((0.0,1.0))

        for ax in ax1s:
            ax.plot(
                x[bmb_mask] / 1e3,
                post_mean_smb[bmb_vals] - tmb[bmb_mask],
                color=color_opts["colors"]["posterior"],
                label="Posterior",
                linewidth=1.0,
            )
            if plot_samples:
                for i in range(20):
                    ax.plot(
                        x[bmb_mask] / 1e3,
                        posterior_smb_samples[i][bmb_vals] - tmb[bmb_mask],
                        color=color_opts["colors"]["posterior"],
                        alpha=samples_alpha,
                        linewidth=0.5,
                    )
            else:
                ax.fill_between(
                    x[bmb_mask] / 1e3,
                    post_lq_smb[bmb_vals] - tmb[bmb_mask],
                    post_uq_smb[bmb_vals] - tmb[bmb_mask],
                    color=color_opts["colors"]["posterior"],
                    alpha=posterior_alpha,
                    linewidth=0.0,
                )

            # Plot real bmb if possible
            if true_smb is not None:
                true_bmb = true_smb - tmb
                ax.plot(
                    x[true_smb_mask] / 1e3,
                    true_bmb[true_smb_mask],
                    color=color_opts["colors"]["observation"],
                    linewidth=1.0,
                    label="True BMB",
                )

        for ax in axs[:-1]:
            ax.tick_params(
                axis="x",  # changes apply to the x-axis
                which="both",  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
            )
            ax.spines[["bottom", "top", "right"]].set_visible(False)
    return fig, axs
