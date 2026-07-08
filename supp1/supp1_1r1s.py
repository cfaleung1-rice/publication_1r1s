import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib.patches import Polygon


def add_arrowhead(ax, x1, y1, x2, y2, color, alpha,
                  tip_offset=0, size=0.2, zorder=10, centered=True):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    height = size * 3
    centroid_offset = 2 * size
    if centered:
        tip = np.array([mx + (tip_offset + centroid_offset) * ux,
                        my + (tip_offset + centroid_offset) * uy])
    else:
        tip = np.array([mx + tip_offset * ux,
                        my + tip_offset * uy])
    base_center = tip - height * np.array([ux, uy])
    left_corner = base_center + size * np.array([px, py])
    right_corner = base_center - size * np.array([px, py])
    triangle = Polygon([tip, left_corner, right_corner],
                       closed=True, facecolor=color,
                       edgecolor='none', alpha=alpha, zorder=zorder, clip_on=False)
    ax.add_patch(triangle)


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

base_size = 4.0
gap_x = 2
margin_l = 1.0
margin_r = 0.5
margin_t = 0.6
margin_b = 0.8

fig_w = margin_l + (2 * base_size) + gap_x + margin_r
fig_h = margin_b + base_size + margin_t

fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x_inch, y_inch, w_inch, h_inch):
    return [x_inch / fig_w, y_inch / fig_h, w_inch / fig_w, h_inch / fig_h]

ax1 = fig.add_axes(get_rect(margin_l, margin_b, base_size, base_size))
ax2 = fig.add_axes(get_rect(margin_l + base_size + gap_x, margin_b, base_size, base_size))

fs_scaling = 1.5
label_fontsize = 16 * fs_scaling
tick_fontsize = 14 * fs_scaling
annotation_fontsize = 20 * fs_scaling

lw_scaling = 2

locs = ['left', 'right', 'top', 'bottom']
c_g = 'tab:orange'
c_k = 'tab:green'

ax1.plot([0, 10], [0, 10], c='k', ls='--', lw=1 * lw_scaling)
ax1.plot([6, 10], [4, 0], c=c_g, alpha=0.8, zorder=5, lw=1 * lw_scaling)
add_arrowhead(ax1, 6, 4, 10, 0, c_g, 0.8)
ax1.plot([6, 2], [4, 0], c=c_k, alpha=0.8, lw=1 * lw_scaling)
add_arrowhead(ax1, 6, 4, 2, 0, c_k, 0.8)
ax1.plot([2, 10], [0, 0], c=c_k, alpha=0.8, clip_on=False, zorder=2, lw=1 * lw_scaling)
add_arrowhead(ax1, 2, 0, 10, 0, c_k, 0.8)

for loc in locs:
    ax1.spines[loc].set_zorder(-1)
    ax1.spines[loc].set_linewidth(1 * lw_scaling)

ax1.set_aspect('equal')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_xticks([0, 2, 6, 10], [0, r'$\alpha - \beta$', r'$\alpha$', r'$\alpha$ + $\beta$'])
ax1.set_yticks([0, 4, 10], [0, r'$\beta$', r'$\alpha$ + $\beta$'])
ax1.set_xlabel(r'Number of $S_\varnothing$', fontsize=label_fontsize)
ax1.set_ylabel(r'Number of $R$', fontsize=label_fontsize)
ax1.tick_params(labelsize=tick_fontsize)

trans = mtransforms.offset_copy(ax1.transAxes, fig=fig, x=-35, y=12, units='points')
ax1.text(0.0, 1.0, 'A', transform=trans, fontsize=annotation_fontsize,
         fontweight='bold', va='bottom', ha='right')

ax2.plot([0, 10], [0, 10], c='k', ls='--', lw=1 * lw_scaling)
ax2.plot([4, 10], [6, 0], c=c_g, alpha=0.8, zorder=5, lw=1 * lw_scaling)
add_arrowhead(ax2, 4, 6, 10, 0, c_g, 0.8)
ax2.plot([4, 0], [6, 2], c=c_k, alpha=0.8, lw=1 * lw_scaling)
add_arrowhead(ax2, 4, 6, 0, 2, c_k, 0.8)
ax2.plot([0, 0], [2, 0], c=c_k, alpha=0.8, clip_on=False, zorder=2, lw=1 * lw_scaling)
add_arrowhead(ax2, 0, 2, 0, 0, c_k, 0.8)
ax2.plot([0, 10], [0, 0], c=c_k, alpha=0.8, clip_on=False, zorder=2, lw=1 * lw_scaling)
add_arrowhead(ax2, 0, 0, 10, 0, c_k, 0.8)

for loc in locs:
    ax2.spines[loc].set_zorder(-1)
    ax2.spines[loc].set_linewidth(1 * lw_scaling)

ax2.set_aspect('equal')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_xticks([0, 4, 10], [0, r'$\alpha$', r'$\alpha$ + $\beta$'])
ax2.set_yticks([0, 2, 6, 10], [0, r'$\beta - \alpha$', r'$\beta$', r'$\alpha$ + $\beta$'])
ax2.set_xlabel(r'Number of $S_\varnothing$ $\quad$ $X_t$', fontsize=label_fontsize)
ax2.set_ylabel(r'Number of $R$ $\quad$ $Y_t$', fontsize=label_fontsize)
ax2.tick_params(labelsize=tick_fontsize)

trans = mtransforms.offset_copy(ax2.transAxes, fig=fig, x=-35, y=12, units='points')
ax2.text(0.0, 1.0, 'B', transform=trans, fontsize=annotation_fontsize,
         fontweight='bold', va='bottom', ha='right')

plt.savefig('supp1_1r1s.pdf', dpi=600, bbox_inches='tight')
plt.show()