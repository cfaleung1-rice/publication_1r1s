import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np

plt.rcParams['mathtext.default'] = 'it'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

scale_A = 1.125
fs_scaling = 1.25
annotation_fontsize = 24 * fs_scaling

base_w = 9.0
h_A = 4.5
h_B = 5.4
gap_y = 0.9
margin_l = 0.8
margin_r = 0.5
margin_t = 0.5
margin_b = 0.7

fig_w = margin_l + base_w + margin_r
fig_h = margin_b + h_A + gap_y + h_B + margin_t

fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x, y, w, h):
    return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

axes = {}
row_y_A = margin_b + h_B + gap_y
row_y_B = margin_b

axes['schematicA'] = fig.add_axes(get_rect(margin_l, row_y_A, base_w, h_A))
axes['schematicB'] = fig.add_axes(get_rect(margin_l, row_y_B, base_w, h_B))

def draw_schematic_A(ax, zoom=1.0):
    r_A = 0.40
    r_B = 0.20
    r_C = 0.70

    arrow_length = 0.8
    arrow_half = arrow_length / 2.0

    gap1 = 0.5
    gap2 = 0.5

    y_top = 0.85
    y_bottom = -0.85

    x_arrow_top = 0.0
    x_B = x_arrow_top - (arrow_half + gap1 + r_B)
    x_C_top = x_arrow_top + (arrow_half + gap1 + r_C)
    x_plus_top = x_B - (gap2 + r_B)
    x_A_top = x_plus_top - (gap2 + r_A)

    x_arrow_bottom = x_arrow_top
    x_C_bottom = x_arrow_bottom - (arrow_half + gap1 + r_C)
    x_A1_bottom = x_arrow_bottom + (arrow_half + gap1 + r_A)
    x_plus_bottom = x_A1_bottom + (gap2 + r_A)
    x_A2_bottom = x_plus_bottom + (gap2 + r_A)

    label_offset = 0.2

    circle_linewidth = 2
    arrow_linewidth = 2
    circle_label_fontsize = 20
    plus_fontsize = 25
    arrow_label_fontsize = 20

    x_margin_original = 4.0
    y_margin_original = 2.0

    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_xlim(-x_margin_original / zoom, x_margin_original / zoom)
    ax.set_ylim(-y_margin_original / zoom, y_margin_original / zoom)

    def draw_labeled_circle(ax, x, y, radius, label, fc='white'):
        circle = patches.Circle(
            (x, y),
            radius,
            facecolor=fc,
            edgecolor='black',
            linewidth=circle_linewidth
        )
        ax.add_patch(circle)
        ax.text(
            x, y, label,
            ha='center',
            va='center',
            fontsize=circle_label_fontsize,
            fontweight='bold'
        )

    draw_labeled_circle(ax, x_A_top, y_top, r_A, r'$\boldsymbol{S}_\boldsymbol{0}$', 'mistyrose')
    ax.text(x_plus_top, y_top, '+', ha='center', va='center',
            fontsize=plus_fontsize, fontweight='bold')
    draw_labeled_circle(ax, x_B, y_top, r_B, r'$\boldsymbol{R}$', 'cornsilk')

    arrow_start = x_arrow_top - arrow_half
    arrow_end = x_arrow_top + arrow_half
    arrow_top = patches.FancyArrowPatch(
        (arrow_start, y_top),
        (arrow_end, y_top),
        arrowstyle='->,head_length=0.18,head_width=0.12',
        mutation_scale=50,
        color='tab:green',
        linewidth=arrow_linewidth
    )
    ax.add_patch(arrow_top)
    ax.text(
        x_arrow_top, y_top + label_offset, r'$k$',
        ha='center', va='center', color='tab:green',
        fontsize=arrow_label_fontsize, fontweight='bold'
    )

    draw_labeled_circle(ax, x_C_top, y_top, r_C, r'$\boldsymbol{S}_\boldsymbol{R}$', 'honeydew')

    draw_labeled_circle(ax, x_C_bottom, y_bottom, r_C, r'$\boldsymbol{S}_\boldsymbol{R}$', 'honeydew')

    arrow_start2 = x_arrow_bottom - arrow_half
    arrow_end2 = x_arrow_bottom + arrow_half
    arrow_bottom = patches.FancyArrowPatch(
        (arrow_start2, y_bottom),
        (arrow_end2, y_bottom),
        arrowstyle='->,head_length=0.18,head_width=0.12',
        mutation_scale=50,
        color='r',
        linewidth=arrow_linewidth
    )
    ax.add_patch(arrow_bottom)
    ax.text(
        x_arrow_bottom, y_bottom + label_offset, r'$g$',
        ha='center', va='center', color='r',
        fontsize=arrow_label_fontsize, fontweight='bold'
    )

    draw_labeled_circle(ax, x_A1_bottom, y_bottom, r_A, r'$\boldsymbol{S}_\boldsymbol{0}$', 'mistyrose')
    ax.text(x_plus_bottom, y_bottom, '+', ha='center', va='center',
            fontsize=plus_fontsize, fontweight='bold')
    draw_labeled_circle(ax, x_A2_bottom, y_bottom, r_A, r'$\boldsymbol{S}_\boldsymbol{0}$', 'mistyrose')

def draw_schematic_B(ax):
    COLS = 9
    ROWS = 5
    dx_space = 4.0
    dy_space = 4.0
    R = 1.5
    center_col = 3
    center_row = 2

    xmin, xmax = -10, 10
    ymin, ymax = -10, 2.25

    for r in range(ROWS):
        for c in range(COLS):
            x = (c - center_col) * dx_space
            y = (r - center_row) * dy_space

            circle = patches.Circle((x, y), R, edgecolor='black', facecolor='white', zorder=4, linewidth=2)
            ax.add_patch(circle)

            dc = c - center_col
            dr = r - center_row

            c_str = 'x' if dc == 0 else f'x + {dc}' if dc > 0 else f'x - {abs(dc)}'
            y_origin_shift = 1
            eff_dr = dr + y_origin_shift
            if eff_dr == 0:
                r_str = 'y'
            elif eff_dr > 0:
                r_str = f'y + {eff_dr}'
            else:
                r_str = f'y - {abs(eff_dr)}'

            label = rf'$\boldsymbol{{{c_str}, {r_str}}}$'

            if xmin <= x <= xmax and ymin <= y <= ymax:
                ax.text(x, y, label, ha='center', va='center', fontsize=15*fs_scaling, zorder=5)

    for r in range(ROWS):
        for c in [-2, -1]:
            x = (c - center_col) * dx_space
            y = (r - center_row) * dy_space
            ghost = patches.Circle((x, y), R, edgecolor='gray', facecolor='white',
                                   linestyle='--', linewidth=2, zorder=3, alpha=0.6)
            ax.add_patch(ghost)

    def get_horiz_pt(c, r, is_start):
        x = (c - center_col) * dx_space
        y = (r - center_row) * dy_space
        angle = np.radians(65 if is_start else 115)
        return (x + R * np.cos(angle), y + R * np.sin(angle))

    def get_diag_pt(c, r, is_start):
        x = (c - center_col) * dx_space
        y = (r - center_row) * dy_space
        angle = np.radians(225 if is_start else 45)
        return (x + R * np.cos(angle), y + R * np.sin(angle))

    for r in range(ROWS):
        for c_start in range(-2, COLS):
            c_end = c_start + 2

            if (c_start < 0 and c_end < 0) or (c_start >= COLS and c_end >= COLS):
                continue

            offset = c_start - center_col
            is_even = (offset % 2 == 0)
            ls = '-' if is_even else '-'

            p1 = get_horiz_pt(c_start, r, is_start=True)
            p2 = get_horiz_pt(c_end, r, is_start=False)

            arrow = patches.FancyArrowPatch(
                p1, (list(p2)[0]-0.15, list(p2)[1]+0.175),
                arrowstyle='-',
                linestyle=ls,
                connectionstyle='arc3,rad=-0.175',
                color='tab:green',
                linewidth=3,
                zorder=5
            )
            ax.add_patch(arrow)

            arrow = patches.FancyArrowPatch(
                p1, p2,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=-0.3',
                mutation_scale=40,
                color='tab:green',
                linewidth=0,
                zorder=4
            )
            ax.add_patch(arrow)

    for r_start in range(-1, ROWS + 1):
        for c_start in range(-1, COLS + 1):
            r_end = r_start - 1
            c_end = c_start - 1

            start_visible = (0 <= r_start < ROWS and 0 <= c_start < COLS)
            end_visible = (0 <= r_end < ROWS and 0 <= c_end < COLS)

            if not (start_visible or end_visible):
                continue

            p1 = get_diag_pt(c_start, r_start, is_start=True)
            p2 = get_diag_pt(c_end, r_end, is_start=False)

            arrow = patches.FancyArrowPatch(
                p1, (list(p2)[0]+0.1, list(p2)[1]+0.1),
                arrowstyle='-',
                color='r',
                linewidth=3,
                zorder=3
            )
            ax.add_patch(arrow)

            arrow = patches.FancyArrowPatch(
                p1, p2,
                arrowstyle='-|>',
                mutation_scale=40,
                color='r',
                linewidth=0,
                zorder=2
            )
            ax.add_patch(arrow)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)

    ax.spines['bottom'].set_zorder(10)
    ax.spines['left'].set_zorder(10)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel(r'Number of $S_0$', fontsize=24, labelpad=20)
    ax.set_ylabel(r'Number of $R$', fontsize=24, labelpad=20)

    ax.scatter(xmax-0.1, ymin, marker='>', s=500, c='black', zorder=10, clip_on=False)
    ax.scatter(xmin, ymax-0.1, marker='^', s=500, c='black', zorder=10, clip_on=False)

    ax.set_aspect('equal')

draw_schematic_A(axes['schematicA'], zoom=scale_A)
draw_schematic_B(axes['schematicB'])

for idx, key in enumerate(['schematicA', 'schematicB']):
    ax = axes[key]
    panel_label = 'A' if idx == 0 else 'B'
    trans = mtransforms.offset_copy(ax.transAxes, fig=fig, x=-40, y=15, units='points')
    ax.text(0.0, 1.0, panel_label, transform=trans, fontsize=annotation_fontsize, fontweight='bold', va='bottom', ha='right')

plt.savefig('fig1_1r1s.pdf', dpi=600, bbox_inches='tight')
plt.show()