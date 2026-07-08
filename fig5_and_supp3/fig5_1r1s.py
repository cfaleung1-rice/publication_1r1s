import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from theory_stochastic_1r1s import simulate_trajectories, get_met
from theory_deterministic_1r1s import solve_numerical

gamma_base = 50.0
gammas_conv = [10**2, 10**3, 10**4, 10**5]
gamma_label = [r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$']
k_base = 1.0
log_min, log_max = -4.0, 4.0
f_min, f_max = 0.1, 0.9
grid_size = 9
n_traj_heatmap = 1000
n_traj_conv = 1000
n_time_pts_sim = 1000
n_time_pts_ode = 1000
seed_base = 42
n_scatter_points = 20
data_file = 'fig5_data.npz'

def get_heatmap_data(gamma):
    f_vals = np.linspace(f_min, f_max, grid_size)
    logr_vals = np.linspace(log_min, log_max, grid_size)
    rmse_grid = np.full((grid_size, grid_size), np.nan)

    for i, logr in enumerate(logr_vals):
        g = (10 ** logr) * k_base
        for j, f in enumerate(f_vals):
            alpha = int(round(gamma * f))
            alpha = max(1, min(gamma - 1, alpha))
            beta = gamma - alpha
            seed = seed_base + i * grid_size + j

            mean_tau, std_tau = get_met(alpha, beta, 0, k_base, g, n_traj=1000, seed=seed)
            t_max = max(mean_tau + 3 * std_tau, 1e-3)
            t_points = np.linspace(0.0, t_max, n_time_pts_sim)

            _, _, y_trajs, _ = simulate_trajectories(
                alpha, beta, 0, k_base, g, t_max,
                n_traj=n_traj_heatmap, t_points=t_points, seed=seed
            )

            mean_y_sim = np.mean(y_trajs, axis=0) / gamma
            _, R_ode, _ = solve_numerical(t_points, alpha, beta, k_base, g)
            R_ode_norm = R_ode / gamma

            rmse_grid[i, j] = np.sqrt(np.mean((mean_y_sim - R_ode_norm)**2))

    return f_vals, logr_vals, rmse_grid

def get_convergence_data(focus_f, focus_logg):
    g_base = (10 ** focus_logg) * k_base
    alpha_base = int(round(gamma_base * focus_f))
    beta_base = gamma_base - alpha_base

    mean_tau, std_tau = get_met(alpha_base, beta_base, 0, k_base, g_base, n_traj=1000, seed=seed_base)
    t_max = max(mean_tau + 3 * std_tau, 1e-3)

    t_points_sim = np.linspace(0.0, t_max, n_time_pts_sim)
    t_points_ode = np.linspace(0.0, t_max, n_time_pts_ode)

    S0_ode_base, R_ode_base, SR_ode_base = solve_numerical(t_points_ode, alpha_base, beta_base, k_base, g_base)
    _, R_ode_for_rmse, _ = solve_numerical(t_points_sim, alpha_base, beta_base, k_base, g_base)

    conv_data = {}
    for i, gamma in enumerate(gammas_conv):
        vol_scale = gamma / gamma_base
        alpha = int(round(gamma * focus_f))
        beta = gamma - alpha
        k_scaled = k_base / vol_scale
        g_scaled = g_base

        _, x_trajs, y_trajs, z_trajs = simulate_trajectories(
            alpha, beta, 0, k_scaled, g_scaled, t_max,
            n_traj=n_traj_conv, t_points=t_points_sim, seed=seed_base + i
        )

        conv_data[gamma] = {
            'S0': np.mean(x_trajs, axis=0) / gamma,
            'R': np.mean(y_trajs, axis=0) / gamma,
            'SR': np.mean(z_trajs, axis=0) / gamma,
            'RMSE_R': np.sqrt(np.mean(((np.mean(y_trajs, axis=0) / gamma) - (R_ode_for_rmse / gamma_base))**2))
        }

    return t_points_sim, t_points_ode, S0_ode_base / gamma_base, R_ode_base / gamma_base, SR_ode_base / gamma_base, conv_data

plt.rcParams['mathtext.default'] = 'it'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

fs_scaling = 1.5
label_fontsize = 16 * fs_scaling
tick_fontsize = 14 * fs_scaling
cbar_label_fontsize = 16 * fs_scaling
cbar_tick_fontsize = 14 * fs_scaling
annotation_fontsize = 20 * fs_scaling

lw_scaling = 2.25

if os.path.exists(data_file):
    data = np.load(data_file, allow_pickle=True)
    f_vals = data['f_vals']
    logr_vals = data['logr_vals']
    rmse_grid = data['rmse_grid']
    focus_f = float(data['focus_f'])
    focus_logg = float(data['focus_logg'])
    t_points_sim = data['t_points_sim']
    t_points_ode = data['t_points_ode']
    S0_ode = data['S0_ode']
    R_ode = data['R_ode']
    SR_ode = data['SR_ode']
    gammas_conv_arr = data['gammas_conv']
    R_curves = data['R_curves']
    S0_curves = data['S0_curves']
    SR_curves = data['SR_curves']
    rmse_list = data['rmse_list'].tolist()

    conv_data = {}
    for idx, g in enumerate(gammas_conv_arr):
        conv_data[int(g)] = {
            'R': R_curves[idx],
            'S0': S0_curves[idx],
            'SR': SR_curves[idx],
            'RMSE_R': rmse_list[idx]
        }
else:
    f_vals, logr_vals, rmse_grid = get_heatmap_data(gamma_base)

    i_max, j_max = np.unravel_index(np.nanargmax(rmse_grid), rmse_grid.shape)
    focus_logg = logr_vals[i_max]
    focus_f = f_vals[j_max]

    t_points_sim, t_points_ode, S0_ode, R_ode, SR_ode, conv_data = get_convergence_data(focus_f, focus_logg)

    rmse_list = [conv_data[g]['RMSE_R'] for g in gammas_conv]

    np.savez_compressed(
        data_file,
        f_vals=f_vals,
        logr_vals=logr_vals,
        rmse_grid=rmse_grid,
        focus_f=np.asarray(focus_f),
        focus_logg=np.asarray(focus_logg),
        t_points_sim=t_points_sim,
        t_points_ode=t_points_ode,
        S0_ode=S0_ode,
        R_ode=R_ode,
        SR_ode=SR_ode,
        gammas_conv=np.array(gammas_conv),
        R_curves=np.stack([conv_data[g]['R'] for g in gammas_conv]),
        S0_curves=np.stack([conv_data[g]['S0'] for g in gammas_conv]),
        SR_curves=np.stack([conv_data[g]['SR'] for g in gammas_conv]),
        rmse_list=np.array(rmse_list),
        allow_pickle=True
    )

marker_interval = max(1, len(t_points_sim) // n_scatter_points)

colors = plt.cm.Spectral(np.linspace(0.15, 0.925, len(gammas_conv)))
hsv = mcolors.rgb_to_hsv(colors[:, :3])
hsv[:, 1] = np.clip(hsv[:, 1] * 1.75, 0, 1)
hsv[:, 2] *= 0.95
colors = mcolors.hsv_to_rgb(hsv)

base_size_supp = 4.0
gap_x_supp = 2.0
cbar_h_supp = 0.15
cbar_gap_supp = 1.0
margin_l_supp = 0.9
margin_r_supp = 0.5
margin_t_supp = 0.6
margin_b_supp = 0.8

fig_w_supp = margin_l_supp + (2 * base_size_supp) + gap_x_supp + margin_r_supp
fig_h_supp = margin_b_supp + cbar_h_supp + cbar_gap_supp + base_size_supp + margin_t_supp

fig_supp = plt.figure(figsize=(fig_w_supp, fig_h_supp))

def get_rect_supp(x_inch, y_inch, w_inch, h_inch):
    return [x_inch / fig_w_supp, y_inch / fig_h_supp, w_inch / fig_w_supp, h_inch / fig_h_supp]

x_colA = margin_l_supp
x_colB = margin_l_supp + base_size_supp + gap_x_supp
y_cbar_supp = margin_b_supp
y_row_supp = margin_b_supp + cbar_h_supp + cbar_gap_supp

axes_supp = {
    'A': fig_supp.add_axes(get_rect_supp(x_colA, y_row_supp, base_size_supp, base_size_supp)),
    'B': fig_supp.add_axes(get_rect_supp(x_colB, y_row_supp, base_size_supp, base_size_supp)),
    'caxA': fig_supp.add_axes(get_rect_supp(x_colA, y_cbar_supp, base_size_supp, cbar_h_supp)),
    'caxB': fig_supp.add_axes(get_rect_supp(x_colB, y_cbar_supp, base_size_supp, cbar_h_supp)),
}

def add_subplot_label_supp(ax, label):
    trans = mtransforms.offset_copy(ax.transAxes, fig=fig_supp, x=-32, y=10, units='points')
    ax.text(0.0, 1.0, label, transform=trans,
            fontsize=annotation_fontsize, fontweight='bold', va='bottom', ha='right')

dw = (f_max - f_min) / (grid_size - 1)
dh = (log_max - log_min) / (grid_size - 1)
extent = [f_min - dw/2, f_max + dw/2, log_min - dh/2, log_max + dh/2]

valid_vals = rmse_grid[~np.isnan(rmse_grid) & (rmse_grid > 0)]
vmin = max(np.nanmin(valid_vals), 1e-6) if len(valid_vals) > 0 else 1e-6
vmax = np.nanmax(valid_vals) if len(valid_vals) > 0 else 1.0
imA = axes_supp['A'].imshow(
    rmse_grid, extent=extent, origin='lower', aspect='auto',
    norm=LogNorm(vmin=vmin, vmax=vmax), cmap='Reds'
)

axes_supp['A'].set_xlabel(r'$\alpha / (\alpha + \beta)$', fontsize=label_fontsize)
axes_supp['A'].set_ylabel(r'$\ln(g/k)$', fontsize=label_fontsize)
axes_supp['A'].set_xticks(f_vals)
axes_supp['A'].set_yticks(logr_vals)
axes_supp['A'].tick_params(axis='both', which='both', labelsize=tick_fontsize, direction='out')

rect_x, rect_y = focus_f - dw/2, focus_logg - dh/2
axes_supp['A'].add_patch(Rectangle((rect_x, rect_y), dw, dh, fill=False, edgecolor='white', linestyle=':', lw=3))
axes_supp['A'].text(focus_f, focus_logg, 'B', ha='center', va='center', color='white',
                    fontsize=label_fontsize, fontweight='bold')

cbarA = fig_supp.colorbar(imA, cax=axes_supp['caxA'], orientation='horizontal')
cbarA.set_label(r'$\varphi_R(t)$ RMSE', fontsize=cbar_label_fontsize)
cbarA.ax.tick_params(labelsize=cbar_tick_fontsize, direction='out')

add_subplot_label_supp(axes_supp['A'], 'A')

axB = axes_supp['B']
log_x = np.log10(np.array(gammas_conv))
log_y = np.log10(np.array(rmse_list))
slope, intercept = np.polyfit(log_x, log_y, 1)
r_squared = 1 - (np.sum((log_y - (slope * log_x + intercept))**2) / np.sum((log_y - np.mean(log_y))**2))

fit_x = np.logspace(np.log10(gammas_conv[0]), np.log10(gammas_conv[-1]), 100)
axB.plot(fit_x, 10**(slope * np.log10(fit_x) + intercept), 'k--', linewidth=3, alpha=0.85, zorder=5)

for i, (gamma, rmse_val) in enumerate(zip(gammas_conv, rmse_list)):
    axB.plot(gamma, rmse_val, marker='x', color=colors[i], markersize=11,
             markeredgewidth=2.5, linestyle='None', zorder=10)

axB.set_xscale('log')
axB.set_yscale('log')
axB.set_xlabel(r'$\alpha + \beta$', fontsize=label_fontsize)
axB.set_ylabel(r'$\varphi_R(t)$ RMSE', fontsize=label_fontsize)
axB.tick_params(axis='both', which='both', labelsize=tick_fontsize, direction='out')
axB.set_xlim(gammas_conv[0] * 0.6, gammas_conv[-1] * 1.6)
ymin_log = min(log_y) - 0.3
ymax_log = max(log_y) + 0.4
axB.set_ylim(10**ymin_log, 10**ymax_log)

axB.text(0.04, 0.96, rf'$m = {slope:.2f}$, $R^2 = {r_squared:.2f}$',
         transform=axB.transAxes, fontsize=annotation_fontsize * 0.65,
         va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='gray', alpha=0.9))

add_subplot_label_supp(axB, 'B')

cmap_discrete = mcolors.ListedColormap(colors)
bounds = np.arange(len(gammas_conv) + 1)
norm_discrete = mcolors.BoundaryNorm(bounds, cmap_discrete.N)
smB = plt.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
smB.set_array([])
tick_locs = bounds[:-1] + 0.5
cbarB = fig_supp.colorbar(smB, cax=axes_supp['caxB'], orientation='horizontal', ticks=tick_locs)
cbarB.ax.set_xticklabels(gamma_label)
cbarB.set_label(r'$\alpha + \beta$', fontsize=cbar_label_fontsize)
cbarB.ax.tick_params(labelsize=cbar_tick_fontsize)

plt.savefig('supp3_1r1s.pdf', dpi=600, bbox_inches='tight')

base_w = 5.0
base_h = 3.0
gap_y = 1.5
cbar_h = 0.2
cbar_gap = 1.2
margin_l = 1.0
margin_r = 0.5
margin_t = 0.6
margin_b = 0.8

fig_w = margin_l + base_w + margin_r
fig_h = margin_b + cbar_h + cbar_gap + (3 * base_h) + (2 * gap_y) + margin_t

fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x_inch, y_inch, w_inch, h_inch):
    return [x_inch / fig_w, y_inch / fig_h, w_inch / fig_w, h_inch / fig_h]

y_cbar = margin_b
y_C = margin_b + cbar_h + cbar_gap
y_B = y_C + base_h + gap_y
y_A = y_B + base_h + gap_y

x_panel = margin_l

axes = {
    'A': fig.add_axes(get_rect(x_panel, y_A, base_w, base_h)),
    'B': fig.add_axes(get_rect(x_panel, y_B, base_w, base_h)),
    'C': fig.add_axes(get_rect(x_panel, y_C, base_w, base_h)),
    'cax': fig.add_axes(get_rect(x_panel, y_cbar, base_w, cbar_h)),
}

def add_subplot_label(ax, label):
    trans = mtransforms.offset_copy(ax.transAxes, fig=fig, x=-35, y=12, units='points')
    ax.text(0.0, 1.0, label, transform=trans,
            fontsize=annotation_fontsize, fontweight='bold', va='bottom', ha='right')

colors = plt.cm.Spectral(np.linspace(0.15, 0.925, len(gammas_conv)))
hsv = mcolors.rgb_to_hsv(colors[:, :3])
hsv[:, 1] = np.clip(hsv[:, 1] * 1.75, 0, 1)
hsv[:, 2] *= 0.95
colors = mcolors.hsv_to_rgb(hsv)

axes['A'].plot(t_points_ode, S0_ode, 'k-', lw=3)
for i, gamma in enumerate(gammas_conv):
    axes['A'].plot(t_points_sim, conv_data[gamma]['S0'],
                   marker='x', linestyle='None', color=colors[i],
                   markevery=marker_interval, markersize=10, markeredgewidth=2, zorder=10)
axes['A'].set_ylabel(r'$\varphi_{S_0}(t)$', fontsize=label_fontsize)
axes['A'].set_xlabel(r'$t$', fontsize=label_fontsize)

axes['B'].plot(t_points_ode, R_ode, 'k-', lw=3)
for i, gamma in enumerate(gammas_conv):
    axes['B'].plot(t_points_sim, conv_data[gamma]['R'],
                   marker='x', linestyle='None', color=colors[i],
                   markevery=marker_interval, markersize=10, markeredgewidth=2, zorder=10)
axes['B'].set_ylabel(r'$\varphi_R(t)$', fontsize=label_fontsize)
axes['B'].set_xlabel(r'$t$', fontsize=label_fontsize)

axes['C'].plot(t_points_ode, SR_ode, 'k-', lw=3)
for i, gamma in enumerate(gammas_conv):
    axes['C'].plot(t_points_sim, conv_data[gamma]['SR'],
                   marker='x', linestyle='None', color=colors[i],
                   markevery=marker_interval, markersize=10, markeredgewidth=2, zorder=10)
axes['C'].set_ylabel(r'$\varphi_{S_R}(t)$', fontsize=label_fontsize)
axes['C'].set_xlabel(r'$t$', fontsize=label_fontsize)

cmap_discrete = mcolors.ListedColormap(colors)
bounds = np.arange(len(gammas_conv) + 1)
norm_discrete = mcolors.BoundaryNorm(bounds, cmap_discrete.N)
sm = plt.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
sm.set_array([])

tick_locs = bounds[:-1] + 0.5
cbar = fig.colorbar(sm, cax=axes['cax'], orientation='horizontal', ticks=tick_locs)
cbar.ax.set_xticklabels(gamma_label)
cbar.set_label(r'$\alpha + \beta$', fontsize=cbar_label_fontsize)
cbar.ax.tick_params(labelsize=cbar_tick_fontsize)

for key in ['A', 'B', 'C']:
    axes[key].set_xlim(0, t_points_sim[-1])
    axes[key].tick_params(axis='both', which='both', labelsize=tick_fontsize, direction='out')
    add_subplot_label(axes[key], key)

plt.savefig('fig5_1r1s.pdf', dpi=600, bbox_inches='tight')
plt.show()