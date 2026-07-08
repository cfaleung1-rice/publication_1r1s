import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def mu_fast(g, k, R):
    return g * R / (g / k + R)

def mu_slow(g, k, R):
    return g * R / (2 * g / k + R)

def load_and_process_data(filename):
    df = pd.read_csv(filename, header=None)
    x_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values
    x_data = np.round(x_data, 3)
    y_data = np.abs(y_data)
    return x_data, y_data

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

plt.rcParams['mathtext.default'] = 'it'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

fs_scaling = 1.5
label_fontsize = 16 * fs_scaling
tick_fontsize = 14 * fs_scaling
annotation_fontsize = 20 * fs_scaling

base_w, base_h = 6, 4
gap_y, margin_l, margin_r, margin_t, margin_b = 1.5, 1.0, 0.5, 0.6, 0.8

num_panels = 2
fig_w = margin_l + base_w + margin_r
fig_h = margin_b + (num_panels * base_h) + ((num_panels - 1) * gap_y) + margin_t
fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x, y, w, h):
    return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

axes = {}
panels = ['A', 'B']
row_y = [margin_b + base_h + gap_y, margin_b]

for i, p in enumerate(panels):
    axes[p] = fig.add_axes(get_rect(margin_l, row_y[i], base_w, base_h))

ax = axes['A']
R = np.linspace(0, 100, 500)
g = 1.0
k_fast = g / 10.0
k_slow = g * 10.0
mu_fast_vals = mu_fast(g, k_fast, R)
mu_slow_vals = mu_slow(g, k_slow, R)

ax.plot(R, mu_fast_vals, lw=5, color='tab:cyan')
ax.plot(R, mu_slow_vals, lw=5, color='tab:red')
ax.axhline(g, c='k', ls='--', lw=2.5)

ax.set_xlabel(r'$c_R(0^+)$', fontsize=label_fontsize)
ax.set_ylabel(r'$\mu(0^+)$', fontsize=label_fontsize)
ax.tick_params(labelsize=tick_fontsize)
ax.set_xlim(min(R) - 0.05, max(R))
ax.set_ylim(0, g * 1.1)

trans = mtransforms.offset_copy(ax.transAxes, fig=fig, x=-40, y=15, units='points')
ax.text(0.0, 1.0, 'A', transform=trans, fontsize=annotation_fontsize, fontweight='bold', va='bottom', ha='right')

ax = axes['B']

def growth_model(x, g, k):
    return mu_slow(g, k, x)

bounds = ([0.01, 0.01], [10.0, 100.0])
data_file = 'monod_fig4.csv'
x_data, y_data = load_and_process_data(data_file)
sort_idx = np.argsort(x_data)
x_data = x_data[sort_idx]
y_data = y_data[sort_idx]
x_fit = x_data
popt, _ = curve_fit(growth_model, x_fit, y_data, p0=[1.0, 5.0], bounds=bounds)

x_fine = np.linspace(0, max(x_fit), 200)

ax.plot(x_fine, growth_model(x_fine, *popt), c='tab:red', lw=5)
ax.scatter(x_data, y_data, marker='x', c='k', s=120, lw=2.5, zorder=10)

ax.set_xlabel(r'$c_R(0^+)$ ($\times10^{-4}$ M)', fontsize=label_fontsize)
ax.set_ylabel(r'$\mu(0^+)$ ($\mathrm{hr}^{-1}$)', fontsize=label_fontsize)
ax.tick_params(labelsize=tick_fontsize)
ax.set_xlim(0, max(x_data))
if len(y_data) > 0 and np.max(y_data) > 0:
    ax.set_ylim(0, np.max(y_data) * 1.1)
else:
    ax.set_ylim(0, 1.0)

trans = mtransforms.offset_copy(ax.transAxes, fig=fig, x=-40, y=15, units='points')
ax.text(0.0, 1.0, 'B', transform=trans, fontsize=annotation_fontsize, fontweight='bold', va='bottom', ha='right')

plt.savefig('fig7_1r1s.pdf', dpi=600, bbox_inches='tight')
plt.show()

y_pred = growth_model(x_data, *popt)
r2 = compute_r2(y_data, y_pred)
print(f"R^2 = {r2:.4f}")