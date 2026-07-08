import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from theory_stochastic_1r1s import compute_moments
from theory_deterministic_1r1s import get_deterministic_curves

plt.rcParams['mathtext.default'] = 'it'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

parameters = [
    (1, 1e+3, 25, 25),
    (1e+3, 1, 30, 20),
    (1e+3, 1, 20, 30),
]
t_points = np.linspace(0, 2, 1000)

inset_xlims = {
    'A': (0.0, 0.1),
    'B': (0.0, 1e-3/2),
    'C': (0.0, 1e-3/2),
    'D': (0.0, 0.1),
}

curves_file = 'curves.npz'

if os.path.exists(curves_file):
    curves = np.load(curves_file, allow_pickle=True)
    t_points = curves['t_points']
    sto_curves = curves['sto_curves']
    det_curves = curves['det_curves']
    inset_data = curves['inset_data']
else:
    sto_curves = np.zeros((3, 3, len(t_points)))
    det_curves = np.zeros((3, 3, len(t_points)))

    for i, (k, g, alpha, beta) in enumerate(parameters):
        X, Y, Z = compute_moments(
            k=k, g=g, alpha=alpha, beta=beta,
            t_points=t_points, n=1, dps=50,
            fast_A=True
        )
        sto_curves[i, 0] = X
        sto_curves[i, 1] = Y
        sto_curves[i, 2] = Z

        c_S_0, c_R, c_S_R = get_deterministic_curves(t_points, k, g, alpha, beta)
        det_curves[i, 0] = c_S_0
        det_curves[i, 1] = c_R
        det_curves[i, 2] = c_S_R

    inset_data = {}
    for p in ['A', 'B', 'C', 'D']:
        i = {'A': 0, 'B': 1, 'C': 2, 'D': 0}[p]
        k, g, alpha, beta = parameters[i]
        inset_xmin, inset_xmax = inset_xlims[p]
        t_inset = np.linspace(inset_xmin, inset_xmax, 1000)

        if p in ['A', 'B', 'C']:
            X_ins, Y_ins, Z_ins = compute_moments(
                k=k, g=g, alpha=alpha, beta=beta,
                t_points=t_inset, n=1, dps=50
            )
            inset_data[p] = {'t': t_inset, 'X': X_ins, 'Y': Y_ins, 'Z': Z_ins}
        else:
            c_S_0_ins, c_R_ins, c_S_R_ins = get_deterministic_curves(
                t_inset, k, g, alpha, beta
            )
            inset_data[p] = {'t': t_inset, 'cS': c_S_0_ins, 'cR': c_R_ins, 'cSR': c_S_R_ins}

    np.savez_compressed(
        curves_file,
        t_points=t_points,
        sto_curves=sto_curves,
        det_curves=det_curves,
        inset_data=inset_data,
        parameters=np.array(parameters, dtype=object),
        inset_xlims=inset_xlims
    )

fs_scaling = 1.5
label_fontsize = 16 * fs_scaling
tick_fontsize = 14 * fs_scaling
annotation_fontsize = 20 * fs_scaling
inset_label_fontsize = label_fontsize * 0.8
inset_tick_fontsize = tick_fontsize * 0.8

lw_scaling = 2.25

base_w, base_h = 4, 3
gap_x, gap_y = 1.5, 1.5
legend_h = 0.4
legend_gap = 1.5
margin_l = 1.0
margin_r = 0.5
margin_t = 0.6
margin_b = 0.8

fig_w = margin_l + 3 * base_w + 2 * gap_x + margin_r
fig_h = margin_b + legend_h + legend_gap + 2 * base_h + gap_y + margin_t

fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x_inch, y_inch, w_inch, h_inch):
    return [x_inch / fig_w, y_inch / fig_h, w_inch / fig_w, h_inch / fig_h]

x_cols = [
    margin_l,
    margin_l + base_w + gap_x,
    margin_l + 2 * (base_w + gap_x)
]

y_row_bot = margin_b + legend_h + legend_gap
y_row_top = y_row_bot + base_h + gap_y

axes = {}

panels_top = ['A', 'B', 'C']
for i, p in enumerate(panels_top):
    ax = fig.add_axes(get_rect(x_cols[i], y_row_top, base_w, base_h))
    axes[p] = ax

    X, Y, Z = sto_curves[i]

    ax.plot(t_points, X, alpha=0.75, c='tab:red', lw=2.0 * lw_scaling)
    ax.plot(t_points, Y, alpha=0.75, c='tab:orange', lw=2.0 * lw_scaling)
    ax.plot(t_points, Z, alpha=0.75, c='tab:green', lw=2.0 * lw_scaling)
    ax.axhline(50, c='k', ls='--', lw=1.0 * lw_scaling)

    ax.set_xlim(0, 2)
    ax.set_xticks([0, 0.5, 1, 1.5, 2], ['0.0', '0.5', '1.0', '1.5', '2.0'])
    ax.set_xlabel(r'$t$', fontsize=label_fontsize)

    ax.set_ylim(-0.25, 54)
    ax.set_yticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50])
    ax.set_ylabel(r'$n(t)$', fontsize=label_fontsize)

    ax.tick_params(labelsize=tick_fontsize)

    trans = mtransforms.offset_copy(ax.transAxes, fig=fig, x=-35, y=12, units='points')
    ax.text(0.0, 1.0, p, transform=trans, fontsize=annotation_fontsize,
            fontweight='bold', va='bottom', ha='right')

    if p in ['A', 'B', 'C']:
        inset_ratio = 0.42
        inset_xmin, inset_xmax = inset_xlims[p]

        ax_ins = ax.inset_axes(
            [1 - inset_ratio - 0.05, 1 - inset_ratio - 0.110, inset_ratio, inset_ratio]
        )

        ins = inset_data[p]
        t_inset = ins['t']
        X_ins, Y_ins, Z_ins = ins['X'], ins['Y'], ins['Z']

        ax_ins.plot(t_inset, X_ins, alpha=0.75, c='tab:red', lw=2.0 * lw_scaling)
        ax_ins.plot(t_inset, Y_ins, alpha=0.75, c='tab:orange', lw=2.0 * lw_scaling)
        ax_ins.plot(t_inset, Z_ins, alpha=0.75, c='tab:green', lw=2.0 * lw_scaling)
        ax_ins.axhline(50, c='k', ls='--', lw=1.0 * lw_scaling)

        ax_ins.set_xlim(inset_xmin, inset_xmax)
        ax_ins.set_ylim(-0.25, 54)
        ax_ins.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_ins.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax_ins.xaxis.get_offset_text().set_fontsize(inset_tick_fontsize)

        ax_ins.set_xlabel(r'$t$', fontsize=inset_label_fontsize)
        ax_ins.set_ylabel(r'$n(t)$', fontsize=inset_label_fontsize)
        ax_ins.tick_params(axis='both', which='both', labelsize=inset_tick_fontsize)

panels_bot = ['D', 'E', 'F']
for i, p in enumerate(panels_bot):
    ax = fig.add_axes(get_rect(x_cols[i], y_row_bot, base_w, base_h))
    axes[p] = ax

    c_S_0, c_R, c_S_R = det_curves[i]

    ax.plot(t_points, c_S_0, alpha=0.75, c='tab:red', lw=2.0 * lw_scaling)
    ax.plot(t_points, c_R, alpha=0.75, c='tab:orange', lw=2.0 * lw_scaling)
    ax.plot(t_points, c_S_R, alpha=0.75, c='tab:green', lw=2.0 * lw_scaling)
    ax.axhline(50, c='k', ls='--', lw=1.0 * lw_scaling)

    ax.set_xlim(0, 2)
    ax.set_xticks([0, 0.5, 1, 1.5, 2], ['0.0', '0.5', '1.0', '1.5', '2.0'])
    ax.set_xlabel(r'$t$', fontsize=label_fontsize)

    ax.set_ylim(-0.25, 54)
    ax.set_yticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50])
    ax.set_ylabel(r'$c(t)$', fontsize=label_fontsize)

    ax.tick_params(labelsize=tick_fontsize)

    # INSET for D only
    if p == 'D':
        inset_ratio = 0.42
        inset_xmin, inset_xmax = inset_xlims['D']

        ax_ins = ax.inset_axes(
            [1 - inset_ratio - 0.05, 1 - inset_ratio - 0.110, inset_ratio, inset_ratio]
        )

        ins = inset_data['D']
        t_inset = ins['t']
        c_S_0_ins, c_R_ins, c_S_R_ins = ins['cS'], ins['cR'], ins['cSR']

        ax_ins.plot(t_inset, c_S_0_ins, alpha=0.75, c='tab:red', lw=2.0 * lw_scaling)
        ax_ins.plot(t_inset, c_R_ins, alpha=0.75, c='tab:orange', lw=2.0 * lw_scaling)
        ax_ins.plot(t_inset, c_S_R_ins, alpha=0.75, c='tab:green', lw=2.0 * lw_scaling)
        ax_ins.axhline(50, c='k', ls='--', lw=1.0 * lw_scaling)

        ax_ins.set_xlim(inset_xmin, inset_xmax)
        ax_ins.set_ylim(-0.25, 54)
        ax_ins.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_ins.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        ax_ins.xaxis.get_offset_text().set_fontsize(inset_tick_fontsize)

        ax_ins.set_xlabel(r'$t$', fontsize=inset_label_fontsize)
        ax_ins.set_ylabel(r'$c(t)$', fontsize=inset_label_fontsize)
        ax_ins.tick_params(axis='both', which='both', labelsize=inset_tick_fontsize)

    trans = mtransforms.offset_copy(ax.transAxes, fig=fig, x=-35, y=12, units='points')
    ax.text(0.0, 1.0, p, transform=trans, fontsize=annotation_fontsize,
            fontweight='bold', va='bottom', ha='right')

legend_elements = [
    Line2D([0], [0], color='tab:red', lw=2.0 * lw_scaling, alpha=0.75, label=r'$S_0$'),
    Line2D([0], [0], color='tab:orange', lw=2.0 * lw_scaling, alpha=0.75, label=r'$R$'),
    Line2D([0], [0], color='tab:green', lw=2.0 * lw_scaling, alpha=0.75, label=r'$S_R$')
]

legend_y_anchor = (margin_b + legend_h * 0.55) / fig_h
fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, legend_y_anchor),
    ncol=4,
    fontsize=label_fontsize,
    frameon=False,
    handlelength=2,
    handletextpad=1,
    columnspacing=2
)

plt.savefig('fig3_1r1s.pdf', dpi=600, bbox_inches='tight')
plt.show()