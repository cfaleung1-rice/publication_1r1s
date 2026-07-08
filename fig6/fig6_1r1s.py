import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from theory_deterministic_1r1s import regime_C

plt.rcParams['mathtext.default'] = 'it'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

fs_scaling = 1.5
label_fontsize = 16 * fs_scaling
tick_fontsize = 14 * fs_scaling
annotation_fontsize = 20 * fs_scaling
inset_label_fontsize = label_fontsize * 0.8
inset_tick_fontsize = tick_fontsize * 0.8

lw_scaling = 2.25

inset_w, inset_h = 0.85, 0.85*1.25
inset_sep = 0.45
inset_pad_x, inset_pad_y = 0.45, 0.4
bar_width, bar_internal_gap = 0.25, 0.10

measured_g_vals = [0.68, 0.43, 0.57]
expt_beta_vals = [11.1, 11.1, 15.0]

def resource_model(t, alpha, beta, g):
    t = np.asarray(t, dtype=float)
    _, R, _ = regime_C(t, 0, g, alpha, beta)
    return R

def load_and_process_data(filename):
    df = pd.read_csv(filename, header=None)
    t_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values
    t_data = np.round(t_data, 1)
    y_data = np.abs(y_data)
    return t_data, y_data

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

t_v1, y_v1 = load_and_process_data('varma_fig7_glc.csv')
t_v2, y_v2 = load_and_process_data('varma_fig11_glc.csv')
t_enj, y_enj = load_and_process_data('enjalbert_fig2a_glc.csv')

t_fit_v1, t_fit_v2, t_fit_enj = t_v1 - t_v1[0], t_v2 - t_v2[0], t_enj - t_enj[0]

bounds = ([0.01, 0.01, 0.01], [50.0, 50.0, 5.0])
popt1, _ = curve_fit(resource_model, t_fit_v1, y_v1, p0=[2.0, 12.0, 0.5], bounds=bounds)
popt2, _ = curve_fit(resource_model, t_fit_v2, y_v2, p0=[2.0, 12.0, 0.5], bounds=bounds)
popt3, _ = curve_fit(resource_model, t_fit_enj, y_enj, p0=[2.0, 16.0, 0.5], bounds=bounds)

popts = [popt1, popt2, popt3]
t_fits = [t_fit_v1, t_fit_v2, t_fit_enj]
y_expts = [y_v1, y_v2, y_enj]
names = ['VarmaFig7', 'VarmaFig11', 'EnjalbertFig2A']
g_expt = [0.68, 0.43, 0.57]
beta_expt = [11.1, 11.1, 15.0]

for i in range(3):
    y_pred = resource_model(t_fits[i], *popts[i])
    r2 = compute_r2(y_expts[i], y_pred)
    g_fit = popts[i][2]
    beta_fit = popts[i][1]
    g_err = 100 * abs(g_fit - g_expt[i]) / g_expt[i]
    beta_err = 100 * abs(beta_fit - beta_expt[i]) / beta_expt[i]
    print(names[i] + ': R2=' + str(round(r2,4)) + ' g_err=' + str(round(g_err,1)) + '% beta_err=' + str(round(beta_err,1)) + '%')

base_w, base_h = 6, 4
gap_y, margin_l, margin_r, margin_t, margin_b = 1.5, 1.0, 0.5, 0.6, 0.8

fig_w, fig_h = margin_l + base_w + margin_r, margin_b + (3 * base_h) + (2 * gap_y) + margin_t
fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x, y, w, h):
    return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

axes = {}
row_y = [margin_b + (2 * base_h) + (2 * gap_y), margin_b + base_h + gap_y, margin_b]
panels = ['A', 'B', 'C']

for i, p in enumerate(panels):
    axes[p] = fig.add_axes(get_rect(margin_l, row_y[i], base_w, base_h))
    x_ins_g = margin_l + inset_pad_x
    x_ins_beta = x_ins_g + inset_w + inset_sep
    y_ins = row_y[i] + inset_pad_y
    axes[f'{p}_g'] = fig.add_axes(get_rect(x_ins_g, y_ins, inset_w, inset_h))
    axes[f'{p}_beta'] = fig.add_axes(get_rect(x_ins_beta, y_ins, inset_w, inset_h))

for i, p in enumerate(panels):
    ax, popt = axes[p], popts[i]
    t_data, y_data = t_fits[i], y_expts[i]
    t_fine = np.linspace(0, max(t_data), 200)

    ax.plot(t_fine, resource_model(t_fine, *popt), c='tab:red', lw=5)
    ax.scatter(t_data, y_data, marker='x', c='k', s=120, lw=2.5, zorder=10)

    ax.set_ylabel('Glucose (mM)', fontsize=label_fontsize)
    ax.set_xlabel(r'$t$ (hr)', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_xlim(0, max(t_data))

    def plot_inset_bars(ins_ax, vals, title, ylim):
        x_pos = [0, bar_width + bar_internal_gap]
        ins_ax.bar(x_pos[0], vals[0], edgecolor='k', facecolor='white', hatch='//', width=bar_width, linewidth=1)
        ins_ax.bar(x_pos[1], vals[1], edgecolor='tab:red', facecolor='tab:red', width=bar_width, linewidth=1)
        ins_ax.set_xticks(x_pos)
        ins_ax.set_xticklabels(['Expt', 'Fit'])
        ins_ax.set_ylim(ylim)
        ins_ax.set_title(title, fontsize=inset_label_fontsize, pad=8)
        ins_ax.tick_params(labelsize=inset_tick_fontsize)

    plot_inset_bars(axes[f'{p}_g'], [measured_g_vals[i], popt[2]], r'$\bar{g}$', (0, 0.8))
    plot_inset_bars(axes[f'{p}_beta'], [expt_beta_vals[i], popt[1]], r'$\bar{\beta}$', (10, 16))

    trans = mtransforms.offset_copy(ax.transAxes, fig=fig, x=-40, y=15, units='points')
    ax.text(0.0, 1.0, p, transform=trans, fontsize=annotation_fontsize, fontweight='bold', va='bottom', ha='right')

plt.savefig('fig6_1r1s.pdf', dpi=600, bbox_inches='tight')
plt.show()