import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, ListedColormap

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from theory_stochastic_1r1s import get_met, simulate_trajectories, compute_moments_A_fast, compute_moments_B_fast, compute_moments_C_fast

gamma_total = 50
k_val = 1.0
log_min, log_max = -4.0, 4.0
f_min, f_max = 0.1, 0.9
grid_size = 9
n_traj = 1000
n_time_pts = 1000
seed_base = 42

plt.rcParams['mathtext.default'] = 'it'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

fs_scale = 1.25
label_fontsize = 16 * fs_scale
tick_fontsize = 14 * fs_scale
cbar_label_fontsize = 16 * fs_scale
cbar_tick_fontsize = 14 * fs_scale
annotation_fontsize = 20 * fs_scale

f_vals = np.linspace(f_min, f_max, grid_size)
logr_vals = np.linspace(log_min, log_max, grid_size)

f_curves = 'RMSE_curves.npz'

if os.path.exists(f_curves):
    loaded_curves = np.load(f_curves)
    saved_curves = {
        't': loaded_curves['t'],
        'f_mean': loaded_curves['f_mean'],
        'f_err': loaded_curves['f_err'],
        'A': loaded_curves['A'],
        'B': loaded_curves['B'],
        'C': loaded_curves['C']
    }

    RMSE_A = np.full((grid_size, grid_size), np.nan)
    RMSE_B = np.full((grid_size, grid_size), np.nan)
    RMSE_C = np.full((grid_size, grid_size), np.nan)
    baseline_noise = np.full((grid_size, grid_size), np.nan)

    for i in range(grid_size):
        for j in range(grid_size):
            mean_sim = saved_curves['f_mean'][i, j]
            if np.all(np.isnan(mean_sim)):
                continue
            f_err_ij = saved_curves['f_err'][i, j]
            if not np.all(np.isnan(f_err_ij)):
                baseline_noise[i, j] = np.sqrt(np.nanmean(f_err_ij**2))
            else:
                baseline_noise[i, j] = 0.0

            ana_A = saved_curves['A'][i, j]
            if not np.all(np.isnan(ana_A)):
                RMSE_A[i, j] = np.sqrt(np.nanmean((mean_sim - ana_A)**2))
            ana_B = saved_curves['B'][i, j]
            if not np.all(np.isnan(ana_B)):
                RMSE_B[i, j] = np.sqrt(np.nanmean((mean_sim - ana_B)**2))
            ana_C = saved_curves['C'][i, j]
            if not np.all(np.isnan(ana_C)):
                RMSE_C[i, j] = np.sqrt(np.nanmean((mean_sim - ana_C)**2))

    best_model = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            mean_sim = saved_curves['f_mean'][i, j]
            errs = np.array([RMSE_A[i, j], RMSE_B[i, j], RMSE_C[i, j]])
            valid_mask = ~np.isnan(errs)
            noise = baseline_noise[i, j]

            amplitude = np.nanmax(mean_sim) - np.nanmin(mean_sim)
            if amplitude <= 0:
                amplitude = 1.0

            abs_tie = 0.02 * amplitude
            abs_fail = 0.05 * amplitude

            tie_thresh = max(2.0 * noise, abs_tie)
            fail_thresh = max(5.0 * noise, abs_fail)

            if np.sum(valid_mask) > 0:
                valid_errs = np.sort(errs[valid_mask])
                best_err = valid_errs[0]
                if best_err > fail_thresh:
                    best_model[i, j] = 4
                elif np.sum(valid_mask) >= 2 and (valid_errs[1] - best_err) < tie_thresh:
                    best_model[i, j] = 3
                else:
                    best_model[i, j] = np.nanargmin(errs)
            else:
                best_model[i, j] = 4
else:
    RMSE_A = np.full((grid_size, grid_size), np.nan)
    RMSE_B = np.full((grid_size, grid_size), np.nan)
    RMSE_C = np.full((grid_size, grid_size), np.nan)
    baseline_noise = np.full((grid_size, grid_size), np.nan)
    saved_curves = {
        't': np.full((grid_size, grid_size, n_time_pts), np.nan),
        'f_mean': np.full((grid_size, grid_size, n_time_pts), np.nan),
        'f_err': np.full((grid_size, grid_size, n_time_pts), np.nan),
        'A': np.full((grid_size, grid_size, n_time_pts), np.nan),
        'B': np.full((grid_size, grid_size, n_time_pts), np.nan),
        'C': np.full((grid_size, grid_size, n_time_pts), np.nan)
    }
    for i, logr in enumerate(logr_vals):
        g = (10 ** logr) * k_val
        for j, f in enumerate(f_vals):
            alpha = int(round(gamma_total * f))
            alpha = max(1, min(gamma_total - 1, alpha))
            beta = gamma_total - alpha
            seed = seed_base + i * grid_size + j
            mean_tau, std_tau = get_met(alpha, beta, 0, k_val, g, n_traj=1000, seed=seed)
            dynamic_t_max = max(mean_tau + 3 * std_tau, 1e-3)
            time_points = np.linspace(0.0, dynamic_t_max, n_time_pts)
            saved_curves['t'][i, j] = time_points
            _, _, y_trajs, _ = simulate_trajectories(
                alpha, beta, 0, k_val, g, dynamic_t_max,
                n_traj=n_traj, t_points=time_points, seed=seed
            )
            mean_y_sim = np.mean(y_trajs, axis=0)
            saved_curves['f_mean'][i, j] = mean_y_sim
            sem = np.std(y_trajs, axis=0, ddof=1) / np.sqrt(n_traj)
            saved_curves['f_err'][i, j] = sem
            baseline_noise[i, j] = np.sqrt(np.mean(sem**2))

            _, mom_y_A, _ = compute_moments_A_fast(k_val, alpha, beta, time_points, 1)
            saved_curves['A'][i, j] = np.array(mom_y_A)
            RMSE_A[i, j] = np.sqrt(np.mean((mean_y_sim - saved_curves['A'][i, j])**2))
            if alpha >= beta:
                _, mom_y_B, _ = compute_moments_B_fast(k_val, g, alpha, beta, time_points, 1)
                saved_curves['B'][i, j] = np.array(mom_y_B)
                RMSE_B[i, j] = np.sqrt(np.mean((mean_y_sim - saved_curves['B'][i, j])**2))
            if alpha < beta:
                _, mom_y_C, _ = compute_moments_C_fast(k_val, g, alpha, beta, time_points, 1)
                saved_curves['C'][i, j] = np.array(mom_y_C)
                RMSE_C[i, j] = np.sqrt(np.mean((mean_y_sim - saved_curves['C'][i, j])**2))
        if (i + 1) % 2 == 0:
            pass

    best_model = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            mean_sim = saved_curves['f_mean'][i, j]
            errs = np.array([RMSE_A[i, j], RMSE_B[i, j], RMSE_C[i, j]])
            valid_mask = ~np.isnan(errs)
            noise = baseline_noise[i, j]

            amplitude = np.nanmax(mean_sim) - np.nanmin(mean_sim)
            if amplitude <= 0:
                amplitude = 1.0

            abs_tie = 0.02 * amplitude
            abs_fail = 0.05 * amplitude

            tie_thresh = max(2.0 * noise, abs_tie)
            fail_thresh = max(5.0 * noise, abs_fail)

            if np.sum(valid_mask) > 0:
                valid_errs = np.sort(errs[valid_mask])
                best_err = valid_errs[0]
                if best_err > fail_thresh:
                    best_model[i, j] = 4
                elif np.sum(valid_mask) >= 2 and (valid_errs[1] - best_err) < tie_thresh:
                    best_model[i, j] = 3
                else:
                    best_model[i, j] = np.nanargmin(errs)
            else:
                best_model[i, j] = 4

    np.savez(f_curves, **saved_curves)

base_size = 4.0
gap_x, gap_y, gap_y_def = 1.2, 1.2, 1.0
cbar_h, cbar_gap = 0.15, 1.0
margin_l, margin_r, margin_t, margin_b = 1.0, 0.5, 0.6, 0.8
size_C = (2 * base_size) + gap_y
h_def = (size_C - 2 * gap_y_def) / 3.0
fig_w = margin_l + base_size + gap_x + size_C + gap_x + base_size + margin_r
fig_h = margin_b + cbar_h + cbar_gap + size_C + margin_t
fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x_inch, y_inch, w_inch, h_inch):
    return [x_inch / fig_w, y_inch / fig_h, w_inch / fig_w, h_inch / fig_h]

x_col0 = margin_l
x_col1 = margin_l + base_size + gap_x
x_col2 = margin_l + base_size + gap_x + size_C + gap_x
y_cbar = margin_b
y_row0 = margin_b + cbar_h + cbar_gap
y_row1 = margin_b + cbar_h + cbar_gap + base_size + gap_y

axes = {}
axes['A'] = fig.add_axes(get_rect(x_col0, y_row1, base_size, base_size))
axes['B'] = fig.add_axes(get_rect(x_col0, y_row0, base_size, base_size))
axes['C'] = fig.add_axes(get_rect(x_col1, y_row0, size_C, size_C))
y_D = y_row0 + (2 * h_def) + (2 * gap_y_def)
y_E = y_row0 + h_def + gap_y_def
y_F = y_row0
axes['D'] = fig.add_axes(get_rect(x_col2, y_D, base_size, h_def))
axes['E'] = fig.add_axes(get_rect(x_col2, y_E, base_size, h_def))
axes['F'] = fig.add_axes(get_rect(x_col2, y_F, base_size, h_def))
axes['caxAB'] = fig.add_axes(get_rect(x_col0, y_cbar, base_size, cbar_h))
axes['caxC'] = fig.add_axes(get_rect(x_col1, y_cbar, size_C, cbar_h))

dw = (f_max - f_min) / (grid_size - 1)
dh = (log_max - log_min) / (grid_size - 1)
extent = [f_min - dw/2, f_max + dw/2, log_min - dh/2, log_max + dh/2]

RMSE_feeding = np.where(~np.isnan(RMSE_C), RMSE_C, RMSE_B)
valid_A = RMSE_A[~np.isnan(RMSE_A) & (RMSE_A > 0)]
valid_B = RMSE_feeding[~np.isnan(RMSE_feeding) & (RMSE_feeding > 0)]
vmin_AB = max(min(np.nanmin(valid_A), np.nanmin(valid_B)), 1e-8)
vmax_AB = max(np.nanmax(valid_A), np.nanmax(valid_B))
norm_AB = LogNorm(vmin=vmin_AB, vmax=vmax_AB)

imA = axes['A'].imshow(RMSE_A, extent=extent, origin='lower', aspect='auto', norm=norm_AB, cmap='Reds')
axes['A'].set_xlabel(r'$\alpha / (\alpha + \beta)$', fontsize=label_fontsize)
axes['A'].set_ylabel(r'$\ln{(g/k)}$', fontsize=label_fontsize)

imB = axes['B'].imshow(RMSE_feeding, extent=extent, origin='lower', aspect='auto', norm=norm_AB, cmap='Reds')
axes['B'].axvline(x=0.45, color='black', linestyle='--', linewidth=1.5)
axes['B'].set_xlabel(r'$\alpha / (\alpha + \beta)$', fontsize=label_fontsize)
axes['B'].set_ylabel(r'$\ln{(g/k)}$', fontsize=label_fontsize)

cb_AB = fig.colorbar(imA, cax=axes['caxAB'], orientation='horizontal')
cb_AB.set_label(r'$\rm{\mathbb{E}}[Y_t]$ RMSE', fontsize=cbar_label_fontsize)
cb_AB.ax.tick_params(labelsize=cbar_tick_fontsize)
cb_AB.ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
cb_AB.ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

cmap_full = plt.cm.get_cmap('Spectral')
slice_colors = cmap_full(np.linspace(0.15, 0.925, 5))
cmap_5 = ListedColormap(slice_colors)
labels_5 = [
    r'$k \ll g$',
    r'$k \gg g$' + '\n' + r'$(\alpha \geq \beta)$',
    r'$k \gg g$' + '\n' + r'$(\alpha < \beta)$',
    'Tie',
    'Intermediate'
]
imC = axes['C'].imshow(best_model, extent=extent, origin='lower', aspect='auto',
                       cmap=cmap_5, vmin=-0.5, vmax=4.5, interpolation='nearest')
axes['C'].set_xlabel(r'$\alpha / (\alpha + \beta)$', fontsize=label_fontsize)
axes['C'].set_ylabel(r'$\ln{(g/k)}$', fontsize=label_fontsize)
cb_C = fig.colorbar(imC, cax=axes['caxC'], orientation='horizontal', ticks=[0, 1, 2, 3, 4])
cb_C.ax.tick_params(labelsize=cbar_tick_fontsize)
cb_C.ax.set_xticklabels(labels_5, fontsize=cbar_label_fontsize)

for ax_key in ['A', 'B', 'C']:
    axes[ax_key].set_xticks(f_vals)
    axes[ax_key].set_yticks(logr_vals)

def plot_outline_on_C(ax, f_target, logr_target, label_text):
    j = np.argmin(np.abs(f_vals - f_target))
    i = np.argmin(np.abs(logr_vals - logr_target))
    x_left = f_vals[j] - dw/2
    y_bottom = logr_vals[i] - dh/2
    rect = plt.Rectangle((x_left, y_bottom), dw, dh, fill=False, edgecolor='black', linestyle=':', linewidth=3)
    ax.add_patch(rect)
    ax.text(f_vals[j], logr_vals[i], label_text, color='black',
            ha='center', va='center', fontweight='bold', fontsize=annotation_fontsize)
    return i, j

idx_D = plot_outline_on_C(axes['C'], 0.3, 3.0, 'D')
idx_E = plot_outline_on_C(axes['C'], 0.7, -3.0, 'E')
idx_F = plot_outline_on_C(axes['C'], 0.3, -3.0, 'F')

def plot_trajectory(ax_key, indices):
    i, j = indices
    ax = axes[ax_key]
    t_curve = saved_curves['t'][i, j]
    sim_curve = saved_curves['f_mean'][i, j]
    step = max(1, len(t_curve) // 20)

    ax.plot(t_curve, saved_curves['A'][i, j], 'tab:green', lw=4)
    if not np.all(np.isnan(saved_curves['C'][i, j])):
        ax.plot(t_curve, saved_curves['C'][i, j], 'tab:orange', lw=4)
    if not np.all(np.isnan(saved_curves['B'][i, j])):
        ax.plot(t_curve, saved_curves['B'][i, j], 'tab:orange', lw=4)
    if best_model[i, j] == 4:
        ax.set_facecolor('mistyrose')
    elif best_model[i, j] == 3:
        ax.set_facecolor('w')

    ax.scatter(t_curve[::step], sim_curve[::step], c='k', marker='x', zorder=2, lw=2, s=50)

    ax.set_ylabel(r'$\rm{\mathbb{E}}[Y_t]$', fontsize=label_fontsize)
    ax.set_xlabel(r'$t$', fontsize=label_fontsize)

plot_trajectory('D', idx_D)
plot_trajectory('E', idx_E)
plot_trajectory('F', idx_F)

for key in ['A', 'B', 'C', 'D', 'E', 'F']:
    axes[key].tick_params(axis='both', which='both', labelsize=tick_fontsize)
    trans = mtransforms.offset_copy(axes[key].transAxes, fig=fig, x=-20, y=10, units='points')
    axes[key].text(0.0, 1.0, key, transform=trans, fontsize=annotation_fontsize,
                   fontweight='bold', va='bottom', ha='right')

plt.savefig('supp2_1r1s.pdf', bbox_inches='tight', dpi=600)