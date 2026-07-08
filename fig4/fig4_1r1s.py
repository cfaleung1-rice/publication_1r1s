import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from scipy.ndimage import gaussian_filter

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from theory_stochastic_1r1s import get_met, analytical_met_phase_A, analytical_met_phase_B, analytical_met_phase_C

gamma_total = 50
k_val = 1.0
g_over_k_range = (-4.5, 4.5)
f_range = (0.05, 0.95)
n_traj = 1000
grid_size = 45
seed_base = 42
sigma_smooth = 1.5
sigma_contour = 1.5
show_contour_E = False
show_contour_F = False

inset_ratio_B = 0.5
inset_xmin_B, inset_xmax_B = -0.5, 3.5

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
    cmap_full = plt.get_cmap(cmap_name)
    colors = cmap_full(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f'trunc({cmap_name})', colors, N=n)

def format_log_label_int(x):
    return rf'$10^{{{int(x)}}}$'

def get_analytical_mfet_C(f_plot, gamma=50, k=1.0):
    mfet = np.full_like(f_plot, np.nan, dtype=float)
    for idx, f in enumerate(f_plot):
        alpha = int(round(gamma * f))
        beta = gamma - alpha
        if alpha > 0 and beta > 0:
            mfet[idx] = analytical_met_phase_A(alpha, beta, k)
    return mfet

def get_analytical_mfet_D(f_plot, g, gamma=50, k=1.0):
    mfet = np.full_like(f_plot, np.nan, dtype=float)
    for idx, f in enumerate(f_plot):
        alpha = int(round(gamma * f))
        beta = gamma - alpha
        if alpha > 0 and beta > 0:
            if alpha >= beta:
                mfet[idx] = analytical_met_phase_B(alpha, beta, k)
            else:
                mfet[idx] = analytical_met_phase_C(alpha, beta, k, g)
    return mfet

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

base_size = 4
gap_x = 1.5
gap_y = 1.5
cbar_h = 0.3
cbar_gap = 1.5
margin_l = 1.0
margin_r = 0.5
margin_t = 0.6
margin_b = 0.8
size_e = (2 * base_size) + gap_y

middle_gap = gap_x
fig_w = margin_l + (4 * base_size) + (3 * gap_x) + middle_gap + margin_r
fig_h = margin_b + cbar_h + cbar_gap + size_e + gap_y + base_size + margin_t
fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x_inch, y_inch, w_inch, h_inch):
    return [x_inch / fig_w, y_inch / fig_h, w_inch / fig_w, h_inch / fig_h]

x_A = margin_l
x_B = margin_l + base_size + gap_x
x_C = x_B + base_size + middle_gap
x_D = x_C + base_size + gap_x

x_E = x_A
x_F = x_C

y_cbar = margin_b
y_large = margin_b + cbar_h + cbar_gap
y_small = y_large + size_e + gap_y

axes = {}
axes['A'] = fig.add_axes(get_rect(x_A, y_small, base_size, base_size))
axes['B'] = fig.add_axes(get_rect(x_B, y_small, base_size, base_size))
axes['C'] = fig.add_axes(get_rect(x_C, y_small, base_size, base_size))
axes['D'] = fig.add_axes(get_rect(x_D, y_small, base_size, base_size))

axes['E'] = fig.add_axes(get_rect(x_E, y_large, size_e, size_e))
axes['F'] = fig.add_axes(get_rect(x_F, y_large, size_e, size_e))

axes['caxE'] = fig.add_axes(get_rect(x_E, y_cbar, size_e, cbar_h))
axes['caxF'] = fig.add_axes(get_rect(x_F, y_cbar, size_e, cbar_h))

fs_scaling = 1.8
label_fontsize = 16 * fs_scaling
tick_fontsize = 14 * fs_scaling
contour_fontsize = 14 * fs_scaling
cbar_label_fontsize = 16 * fs_scaling
cbar_tick_fontsize = 14 * fs_scaling
annotation_fontsize = 20 * fs_scaling
inset_label_fontsize = label_fontsize * 0.8
inset_tick_fontsize = tick_fontsize * 0.8

lw_scaling = 2.25

g_vals_min, g_vals_max = -10, 10
g_vals_asymp = np.logspace(g_vals_min, g_vals_max, 25)
ratio_vals = g_vals_asymp / k_val
log_ratio_vals = np.log10(ratio_vals)
n_traj_asymp = 1000

n_points_inset = 100
g_vals_asymp_inset = np.logspace(g_vals_min, g_vals_max, n_points_inset)
log_ratio_vals_inset = np.log10(g_vals_asymp_inset / k_val)

a1, b1 = 35, 15
sim_times_1 = []
for idx, g in enumerate(g_vals_asymp):
    mean_tau, _ = get_met(a1, b1, 0, k_val, g, n_traj=n_traj_asymp, seed=seed_base + 10000 + idx)
    sim_times_1.append(mean_tau)
asymp_fast_1 = analytical_met_phase_A(a1, b1, k_val)
asymp_slow_1 = analytical_met_phase_B(a1, b1, k_val)
axes['A'].semilogy(log_ratio_vals, sim_times_1, color=None, linestyle='', marker='x', markeredgecolor='k', markeredgewidth=2, markersize=10, zorder=10)
axes['A'].axhline(asymp_fast_1, color='tab:green', lw=3 * lw_scaling, alpha=0.75)
axes['A'].axhline(asymp_slow_1, color='tab:orange', lw=3 * lw_scaling, alpha=0.75)
axes['A'].set_xlabel(r'$\ln(g/k)$', fontsize=label_fontsize)
axes['A'].set_ylabel(r'$\langle\tau_R\rangle$', fontsize=label_fontsize)
axes['A'].set_xlim(g_vals_min, g_vals_max)
axes['A'].set_xticks(np.linspace(g_vals_min, g_vals_max, 5))

a2, b2 = 15, 35
sim_times_2 = []
for idx, g in enumerate(g_vals_asymp):
    mean_tau, _ = get_met(a2, b2, 0, k_val, g, n_traj=n_traj_asymp, seed=seed_base + 20000 + idx)
    sim_times_2.append(mean_tau)
sim_times_2_inset = []
for idx, g in enumerate(g_vals_asymp_inset):
    mean_tau, _ = get_met(a2, b2, 0, k_val, g, n_traj=n_traj_asymp, seed=seed_base + 30000 + idx)
    sim_times_2_inset.append(mean_tau)
asymp_fast_2 = analytical_met_phase_A(a2, b2, k_val)
asymp_slow_2 = analytical_met_phase_C(a2, b2, k_val, g_vals_asymp)
g_dense = np.logspace(g_vals_min, g_vals_max, 1000)
log_ratio_dense = np.log10(g_dense / k_val)
asymp_slow_2_dense = analytical_met_phase_C(a2, b2, k_val, g_dense)
axes['B'].semilogy(log_ratio_vals, sim_times_2, color=None, linestyle='', marker='x', markeredgecolor='k', markeredgewidth=2, markersize=10, zorder=10)
axes['B'].axhline(asymp_fast_2, color='tab:green', lw=3 * lw_scaling, alpha=0.75)
axes['B'].semilogy(log_ratio_dense, asymp_slow_2_dense, color='tab:orange', lw=3 * lw_scaling, zorder=5, alpha=0.75)
axes['B'].set_xlabel(r'$\ln(g/k)$', fontsize=label_fontsize)
axes['B'].set_ylabel(r'$\langle\tau_R\rangle$', fontsize=label_fontsize)
axes['B'].set_xlim(g_vals_min, g_vals_max)
axes['B'].set_xticks(np.linspace(g_vals_min, g_vals_max, 5))

axB_ins = axes['B'].inset_axes(
    [1 - inset_ratio_B - 0.02, 1 - inset_ratio_B - 0.02, inset_ratio_B, inset_ratio_B]
)
mask_dense_ins = (log_ratio_dense >= inset_xmin_B) & (log_ratio_dense <= inset_xmax_B)
x_dense_ins = log_ratio_dense[mask_dense_ins]
y_orange_ins = asymp_slow_2_dense[mask_dense_ins]
y_green_ins = asymp_fast_2
if len(y_orange_ins) > 0 and np.all(np.isfinite(y_orange_ins)):
    y_orange_max_ins = np.nanmax(y_orange_ins)
else:
    y_orange_max_ins = y_green_ins * 10.0
mask_sim_ins = (log_ratio_vals_inset >= inset_xmin_B) & (log_ratio_vals_inset <= inset_xmax_B)
x_sim_ins = log_ratio_vals_inset[mask_sim_ins]
y_sim_ins = np.array(sim_times_2_inset)[mask_sim_ins]
axB_ins.axhline(y_green_ins, color='tab:green', lw=3 * lw_scaling, zorder=5, alpha=0.75)
if len(x_dense_ins) > 1:
    axB_ins.semilogy(x_dense_ins, y_orange_ins, color='tab:orange', lw=3 * lw_scaling, zorder=6, alpha=0.75)
if len(x_sim_ins) > 0:
    axB_ins.plot(x_sim_ins, y_sim_ins, linestyle='', marker='x',
                 markeredgecolor='k', markeredgewidth=1.5, markersize=5, zorder=10)
axB_ins.set_xlim(inset_xmin_B, inset_xmax_B)
y_lower_ins = y_green_ins * 0.7
y_upper_ins = y_orange_max_ins * 1.3
axB_ins.set_ylim(y_lower_ins, y_upper_ins)
axB_ins.tick_params(axis='both', which='both', labelsize=tick_fontsize * 0.8)
axB_ins.set_xlabel(r'$\ln(g/k)$', fontsize=label_fontsize * 0.8)
axB_ins.set_ylabel(r'$\langle\tau_R\rangle$', fontsize=label_fontsize * 0.8)
axB_ins.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
axB_ins.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=20))
axB_ins.yaxis.set_minor_formatter(ticker.NullFormatter())
axB_ins.xaxis.set_minor_locator(ticker.AutoMinorLocator())

axes['A'].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
axes['A'].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
axes['A'].yaxis.set_minor_formatter(ticker.NullFormatter())
axes['A'].xaxis.set_minor_locator(ticker.AutoMinorLocator())

f_vals = np.linspace(f_range[0], f_range[1], grid_size)
logr_vals = np.linspace(g_over_k_range[0], g_over_k_range[1], grid_size)
f_grid, logr_grid = np.meshgrid(f_vals, logr_vals)

met_samples_file = 'met_samples.npz'
if os.path.exists(met_samples_file):
    met_data = np.load(met_samples_file)
    met = met_data['met']
    f_vals = met_data['f_vals']
    logr_vals = met_data['logr_vals']
    a_grid = met_data['a_grid']
    b_grid = met_data['b_grid']
    f_grid, logr_grid = np.meshgrid(f_vals, logr_vals)
else:
    met = np.full_like(f_grid, np.nan)
    a_grid = np.round(gamma_total * f_grid).astype(int)
    b_grid = gamma_total - a_grid
    for i in range(grid_size):
        for j in range(grid_size):
            alpha = int(a_grid[i, j])
            beta = int(b_grid[i, j])
            if alpha == 0 or beta == 0:
                continue
            g = 10 ** logr_vals[i] * k_val
            mean_tau, _ = get_met(alpha, beta, 0, k_val, g, n_traj=n_traj, seed=seed_base + i*grid_size + j)
            met[i, j] = mean_tau
    np.savez(met_samples_file,
             met=met,
             f_vals=f_vals,
             logr_vals=logr_vals,
             a_grid=a_grid,
             b_grid=b_grid,
             gamma_total=gamma_total,
             k_val=k_val,
             n_traj=n_traj,
             seed_base=seed_base)

target_logr_C = 3.0
iC = np.argmin(np.abs(logr_vals - target_logr_C))
simC = met[iC, :]
g_C = 10 ** target_logr_C * k_val
f_plot = f_vals
alpha_disc = np.arange(1, gamma_total)
f_disc = alpha_disc / float(gamma_total)
anal_C_disc = np.array([
    analytical_met_phase_A(a, gamma_total - a, k_val) for a in alpha_disc
])
axes['C'].semilogy(f_disc, anal_C_disc, color='tab:green', lw=3 * lw_scaling, zorder=5, alpha=0.75)
validC = np.isfinite(simC) & (simC > 0)
if np.any(validC):
    axes['C'].plot(f_plot[validC][::2], simC[validC][::2], linestyle='', marker='x',
                   markeredgecolor='k', markeredgewidth=2, markersize=10, zorder=10)
axes['C'].set_xlabel(r'$\alpha / (\alpha + \beta)$', fontsize=label_fontsize)
axes['C'].set_ylabel(r'$\langle\tau_R\rangle$', fontsize=label_fontsize)
axes['C'].set_xlim(0, 1)
if np.any(validC):
    ymin_C = min(np.nanmin(anal_C_disc), np.nanmin(simC[validC])) * 0.4
    ymax_C = max(np.nanmax(anal_C_disc), np.nanmax(simC[validC])) * 2.5
    axes['C'].set_ylim(ymin_C, ymax_C)

target_logr_D = -3.0
iD = np.argmin(np.abs(logr_vals - target_logr_D))
simD = met[iD, :]
g_D = 10 ** target_logr_D * k_val
anal_D_disc = np.array([
    analytical_met_phase_B(a, gamma_total - a, k_val) if a >= (gamma_total - a)
    else analytical_met_phase_C(a, gamma_total - a, k_val, g_D)
    for a in alpha_disc
])
axes['D'].semilogy(f_disc, anal_D_disc, color='tab:orange', lw=3 * lw_scaling, zorder=5, alpha=0.75)
validD = np.isfinite(simD) & (simD > 0)
if np.any(validD):
    axes['D'].plot(f_plot[validD][::2], simD[validD][::2], linestyle='', marker='x',
                   markeredgecolor='k', markeredgewidth=2, markersize=10, zorder=10)
axes['D'].set_xlabel(r'$\alpha / (\alpha + \beta)$', fontsize=label_fontsize)
axes['D'].set_ylabel(r'$\langle\tau_R\rangle$', fontsize=label_fontsize)
axes['D'].set_xlim(0, 1)
if np.any(validD):
    ymin_D = min(np.nanmin(anal_D_disc), np.nanmin(simD[validD])) * 0.4
    ymax_D = max(np.nanmax(anal_D_disc), np.nanmax(simD[validD])) * 2.5
    axes['D'].set_ylim(ymin_D, ymax_D)

for key in ['C', 'D']:
    axes[key].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axes[key].xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axes[key].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
    axes[key].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    axes[key].yaxis.set_minor_formatter(ticker.NullFormatter())
    axes[key].tick_params(axis='both', which='both', labelsize=tick_fontsize)

met_clean = np.where(np.isinf(met), np.nan, met)
g_grid = 10 ** logr_grid * k_val
scaled_sum = met_clean * k_val
valid_sum = scaled_sum[np.isfinite(scaled_sum) & (scaled_sum > 0)]
vmin_sum_val = np.min(valid_sum) if len(valid_sum) else 1e-8
vmax_sum_val = np.max(valid_sum) if len(valid_sum) else 1.0
norm_sum = LogNorm(vmin=vmin_sum_val, vmax=vmax_sum_val)
plot_scaled_sum = np.nan_to_num(scaled_sum, nan=vmax_sum_val)
cmap_e = truncate_colormap('magma', minval=0.20, maxval=1.00)
rgba_e = cmap_e(norm_sum(plot_scaled_sum))
rgba_e_smooth = np.empty_like(rgba_e)
for c in range(4):
    rgba_e_smooth[..., c] = gaussian_filter(rgba_e[..., c], sigma=sigma_smooth, mode='nearest')
with np.errstate(divide='ignore', invalid='ignore'):
    log_sum = np.log10(plot_scaled_sum)
smooth_log_sum = gaussian_filter(log_sum, sigma=sigma_contour)
levels_sum = np.arange(np.floor(np.nanmin(smooth_log_sum)/2)*2, np.ceil(np.nanmax(smooth_log_sum)/2)*2 + 1, 2)
extent = [f_range[0], f_range[1], g_over_k_range[0], g_over_k_range[1]]
axes['E'].imshow(rgba_e_smooth, origin='lower', extent=extent, aspect='auto', interpolation='bilinear')
axes['E'].axhline(0, color='black', linestyle='--', linewidth=1.5 * lw_scaling, zorder=1)
axes['E'].axvline(0.5, ymin=0, ymax=0.5, color='black', linestyle='--', linewidth=1.5 * lw_scaling, zorder=1)
if show_contour_E:
    cs_e = axes['E'].contour(f_vals, logr_vals, smooth_log_sum, levels=levels_sum, colors='black', linestyles='-', alpha=0.6, linewidths=3)
    axes['E'].clabel(cs_e, inline=True, fontsize=contour_fontsize, fmt=format_log_label_int)
sm_e = plt.cm.ScalarMappable(norm=norm_sum, cmap=cmap_e)
sm_e.set_array([])
cb_e = fig.colorbar(sm_e, cax=axes['caxE'], orientation='horizontal')
cb_e.set_label(r'$k \langle\tau_R\rangle$', fontsize=cbar_label_fontsize)
cb_e.ax.tick_params(labelsize=cbar_tick_fontsize)
cb_e.ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
cb_e.ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
cb_e.ax.xaxis.set_minor_formatter(ticker.NullFormatter())
cb_e.ax.xaxis.set_major_formatter(LogFormatterMathtext())

met_samples_file_F = 'met_samples_F.npz'
g_fixed_F = k_val
if os.path.exists(met_samples_file_F):
    met_data_F = np.load(met_samples_file_F)
    met_F = met_data_F['met']
    f_vals_F = met_data_F['f_vals']
    logr_vals_F = met_data_F['logr_vals']
    a_grid_F = met_data_F['a_grid']
    b_grid_F = met_data_F['b_grid']
    f_grid_F, logr_grid_F = np.meshgrid(f_vals_F, logr_vals_F)
else:
    f_vals_F = np.linspace(f_range[0], f_range[1], grid_size)
    logr_vals_F = np.linspace(g_over_k_range[0], g_over_k_range[1], grid_size)
    f_grid_F, logr_grid_F = np.meshgrid(f_vals_F, logr_vals_F)
    met_F = np.full_like(f_grid_F, np.nan)
    a_grid_F = np.round(gamma_total * f_grid_F).astype(int)
    b_grid_F = gamma_total - a_grid_F
    for i in range(grid_size):
        for j in range(grid_size):
            alpha = int(a_grid_F[i, j])
            beta = int(b_grid_F[i, j])
            if alpha == 0 or beta == 0:
                continue
            k_var = g_fixed_F / (10 ** logr_vals_F[i])
            mean_tau, _ = get_met(alpha, beta, 0, k_var, g_fixed_F, n_traj=n_traj, seed=seed_base + 100000 + i*grid_size + j)
            met_F[i, j] = mean_tau
    np.savez(met_samples_file_F,
             met=met_F,
             f_vals=f_vals_F,
             logr_vals=logr_vals_F,
             a_grid_F=a_grid_F,
             b_grid_F=b_grid_F,
             gamma_total=gamma_total,
             g_fixed=g_fixed_F,
             n_traj=n_traj,
             seed_base=seed_base)

met_F_clean = np.where(np.isinf(met_F), np.nan, met_F)
scaled_F = met_F_clean * g_fixed_F
valid_sum_F = scaled_F[np.isfinite(scaled_F) & (scaled_F > 0)]
vmin_F_val = np.min(valid_sum_F) if len(valid_sum_F) else 1e-8
vmax_F_val = np.max(valid_sum_F) if len(valid_sum_F) else 1.0
norm_F = LogNorm(vmin=vmin_F_val, vmax=vmax_F_val)
plot_scaled_F = np.nan_to_num(scaled_F, nan=vmax_F_val)
cmap_f = truncate_colormap('magma', minval=0.20, maxval=1.00)
rgba_f = cmap_f(norm_F(plot_scaled_F))
rgba_f_smooth = np.empty_like(rgba_f)
for c in range(4):
    rgba_f_smooth[..., c] = gaussian_filter(rgba_f[..., c], sigma=sigma_smooth, mode='nearest')
with np.errstate(divide='ignore', invalid='ignore'):
    log_sum_F = np.log10(plot_scaled_F)
smooth_log_sum_F = gaussian_filter(log_sum_F, sigma=sigma_contour)
levels_sum_F = np.arange(np.floor(np.nanmin(smooth_log_sum_F)/2)*2, np.ceil(np.nanmax(smooth_log_sum_F)/2)*2 + 1, 2)
axes['F'].imshow(rgba_f_smooth, origin='lower', extent=extent, aspect='auto', interpolation='bilinear')
axes['F'].axhline(0, color='black', linestyle='--', linewidth=1.5 * lw_scaling, zorder=1)
axes['F'].axvline(0.5, ymin=0, ymax=0.5, color='black', linestyle='--', linewidth=1.5 * lw_scaling, zorder=1)
if show_contour_F:
    cs_f = axes['F'].contour(f_vals_F, logr_vals_F, smooth_log_sum_F, levels=levels_sum_F, colors='black', linestyles='-', alpha=0.6, linewidths=3)
    axes['F'].clabel(cs_f, inline=True, fontsize=contour_fontsize, fmt=format_log_label_int)
sm_f = plt.cm.ScalarMappable(norm=norm_F, cmap=cmap_f)
sm_f.set_array([])
cb_f = fig.colorbar(sm_f, cax=axes['caxF'], orientation='horizontal')
cb_f.set_label(r'$g \langle\tau_R\rangle$', fontsize=cbar_label_fontsize)
cb_f.ax.tick_params(labelsize=cbar_tick_fontsize)
cb_f.ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
cb_f.ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
cb_f.ax.xaxis.set_minor_formatter(ticker.NullFormatter())
cb_f.ax.xaxis.set_major_formatter(LogFormatterMathtext())

for key in ['A', 'B']:
    axes[key].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
    axes[key].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
    axes[key].yaxis.set_minor_formatter(ticker.NullFormatter())
    axes[key].xaxis.set_minor_locator(ticker.AutoMinorLocator())

def sparse_log_formatter(x, pos):
    if x <= 0:
        return ""
    exponent = int(np.round(np.log10(x)))
    return rf'$10^{{{exponent}}}$' if exponent % 2 == 0 else ''

axes['B'].yaxis.set_major_formatter(ticker.FuncFormatter(sparse_log_formatter))

for key in ['E', 'F']:
    axes[key].set_xlabel(r'$\alpha / (\alpha + \beta)$', fontsize=label_fontsize)
    axes[key].set_ylabel(r'$\ln{(g/k)}$', fontsize=label_fontsize)

for key in ['A', 'B', 'C', 'D', 'E', 'F']:
    axes[key].tick_params(axis='both', which='both', labelsize=tick_fontsize)
    trans = mtransforms.offset_copy(axes[key].transAxes, fig=fig, x=-20, y=10, units='points')
    axes[key].text(0.0, 1.0, key, transform=trans, fontsize=annotation_fontsize, fontweight='bold', va='bottom', ha='right')

plt.savefig('fig4_1r1s.pdf', dpi=600, bbox_inches='tight')
plt.show()