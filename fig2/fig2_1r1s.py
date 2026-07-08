import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['mathtext.default'] = 'it'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

LATTICE_DX = 4.0
LATTICE_DY = 4.0
LATTICE_RADIUS = 1.5
CIRCLE_LW = 2
ARROW_LW = 3
ARROW_MUTATION_SCALE = 40
SPINE_LW = 4

GLOBAL_XMIN = -10.00
GLOBAL_XMAX = 10.00
GLOBAL_YMIN = -10.00
GLOBAL_YMAX = 2.25

fs_scaling = 1.25
annotation_fontsize = 24 * fs_scaling
state_fontsize = 15 * fs_scaling

panel_h = 6.0
panel_w = 9.0
mid_w = 2.0
mid_gap = 0.8
right_group_w = panel_w + mid_w + mid_gap + panel_w

gap_y = 1.2
margin_l = 0.7
margin_r = 0.5
margin_t = 0.5
margin_b = 0.6

fig_w = margin_l + right_group_w + margin_r
fig_h = margin_b + panel_h + gap_y + panel_h + margin_t

fig = plt.figure(figsize=(fig_w, fig_h))

def get_rect(x, y, w, h):
    return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

axes = {}

right_x0 = margin_l

c_l_x = right_x0
c_l_y = margin_b + panel_h + gap_y
axes['A_l'] = fig.add_axes(get_rect(c_l_x, c_l_y, panel_w, panel_h))

c_m_x = c_l_x + panel_w
c_m_y = c_l_y
axes['A_m'] = fig.add_axes(get_rect(c_m_x, c_m_y, mid_w, panel_h), facecolor='none')

c_r_x = c_m_x + mid_w + mid_gap
c_r_y = c_l_y
axes['A_r'] = fig.add_axes(get_rect(c_r_x, c_r_y, panel_w, panel_h))

d_l_x = right_x0
d_l_y = margin_b
axes['B_l'] = fig.add_axes(get_rect(d_l_x, d_l_y, panel_w, panel_h))

d_m_x = d_l_x + panel_w
d_m_y = margin_b
axes['B_m'] = fig.add_axes(get_rect(d_m_x, d_m_y, mid_w, panel_h), facecolor='none')

d_r_x = d_m_x + mid_w + mid_gap
d_r_y = margin_b
axes['B_r'] = fig.add_axes(get_rect(d_r_x, d_r_y, panel_w, panel_h))

def draw_sim_scheme_A(ax_left, ax_mid, ax_right):
    num_levels = 4
    arrow_length = 3.0
    dx = LATTICE_DX
    dy = LATTICE_DY
    R = LATTICE_RADIUS
    xmin, xmax = GLOBAL_XMIN, GLOBAL_XMAX
    ymin, ymax = GLOBAL_YMIN, GLOBAL_YMAX

    def draw_lattice(ax):
        for k in range(num_levels):
            for i in range(-k, k+1, 2):
                x = i * dx
                y = -k * dy
                ax.add_patch(patches.Circle((x, y), R, edgecolor='black', facecolor='white',
                                            zorder=4, linewidth=CIRCLE_LW))
                alpha = (r'\alpha' if i == 0
                         else rf'\alpha+{i}' if i > 0
                         else rf'\alpha-{abs(i)}')
                beta = r'\beta' if k == 0 else rf'\beta-{k}'
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ax.text(x, y, rf'$\boldsymbol{{{alpha}, {beta}}}$',
                            ha='center', va='center', fontsize=state_fontsize, zorder=5)

    def get_horiz_pt(i, k, is_start):
        angle = np.radians(65 if is_start else 115)
        return (i*dx + R*np.cos(angle), -k*dy + R*np.sin(angle))

    def get_diag_left_pt(i, k, is_start):
        angle = np.radians(225 if is_start else 45)
        return (i*dx + R*np.cos(angle), -k*dy + R*np.sin(angle))

    def get_diag_right_pt(i, k, is_start):
        angle = np.radians(315 if is_start else 135)
        return (i*dx + R*np.cos(angle), -k*dy + R*np.sin(angle))

    draw_lattice(ax_left)
    for k in range(num_levels):
        for i in range(-k, k+1, 2):
            if i + 2 > k:
                continue
            p1 = get_horiz_pt(i, k, True)
            p2 = get_horiz_pt(i+2, k, False)
            ax_left.add_patch(patches.FancyArrowPatch(p1, (p2[0]-0.15, p2[1]+0.175),
                              arrowstyle='-', connectionstyle='arc3,rad=-0.175',
                              color='tab:green', linewidth=ARROW_LW, zorder=4))
            ax_left.add_patch(patches.FancyArrowPatch(p1, p2, arrowstyle='-|>',
                              connectionstyle='arc3,rad=-0.3', mutation_scale=ARROW_MUTATION_SCALE,
                              color='tab:green', linewidth=0, zorder=3))
    for k in range(num_levels-1):
        for i in range(-k, k+1, 2):
            p1 = get_diag_left_pt(i, k, True)
            p2 = get_diag_left_pt(i-1, k+1, False)
            ax_left.add_patch(patches.FancyArrowPatch(p1, (p2[0]+0.1, p2[1]+0.1),
                              arrowstyle='-', color='r', linewidth=ARROW_LW, zorder=2))
            ax_left.add_patch(patches.FancyArrowPatch(p1, p2, arrowstyle='-|>',
                              mutation_scale=ARROW_MUTATION_SCALE, color='r', linewidth=0, zorder=1))

    draw_lattice(ax_right)
    for k in range(num_levels-1):
        i = k
        p1 = get_diag_right_pt(i, k, True)
        p2 = get_diag_right_pt(i+1, k+1, False)
        ax_right.add_patch(patches.FancyArrowPatch(p1, (p2[0]-0.15, p2[1]+0.175),
                            arrowstyle='-', color='tab:blue', linewidth=ARROW_LW, zorder=2))
        ax_right.add_patch(patches.FancyArrowPatch(p1, p2, arrowstyle='-|>',
                            mutation_scale=ARROW_MUTATION_SCALE, color='tab:blue', linewidth=0, zorder=1))

    ax_mid.set_xlim(0, arrow_length)
    ax_mid.set_ylim(-1, 1)
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['right'].set_visible(False)
    ax_mid.spines['bottom'].set_visible(False)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.set_xticks([])
    ax_mid.set_yticks([])
    ax_mid.add_patch(patches.FancyArrowPatch((0,0), (arrow_length,0), arrowstyle='->',
                     mutation_scale=ARROW_MUTATION_SCALE, color='black', linewidth=ARROW_LW, zorder=10))
    ax_mid.text(arrow_length/2, 0.1, r'$g/k \nearrow \infty$', ha='center', va='bottom',
                fontsize=24, zorder=10)

    for ax in (ax_left, ax_right):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(SPINE_LW)
        ax.spines['left'].set_linewidth(SPINE_LW)
        ax.spines['bottom'].set_zorder(10)
        ax.spines['left'].set_zorder(10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'Number of $S_0$', fontsize=24, labelpad=20)
        ax.set_ylabel(r'Number of $R$', fontsize=24, labelpad=20)
        ax.scatter(xmax-0.1, ymin, marker='>', s=500, c='black', zorder=10, clip_on=False)
        ax.scatter(xmin, ymax-0.1, marker='^', s=500, c='black', zorder=10, clip_on=False)
        ax.set_aspect('equal')

def draw_sim_scheme_Cii(ax_left, ax_mid, ax_right):
    num_levels = 5
    max_x = 6
    arrow_length = 3.0
    dx = LATTICE_DX
    dy = LATTICE_DY
    R = LATTICE_RADIUS
    xmin, xmax = GLOBAL_XMIN + 2 * dx, GLOBAL_XMAX + 2 * dx
    ymin, ymax = GLOBAL_YMIN, GLOBAL_YMAX

    def draw_half_lattice(ax, show_ghost=False):
        for r in range(num_levels):
            start_x = 0 if r % 2 == 0 else 1
            for x_val in range(start_x, max_x + 1, 2):
                x = x_val * dx
                y = -r * dy
                ax.add_patch(patches.Circle((x, y), R, edgecolor='black', facecolor='white',
                                            zorder=4, linewidth=CIRCLE_LW))
                label = (rf'${{{x_val}, \beta-\alpha-{r}}}$'
                         if r != 0
                         else rf'${{{x_val}, \beta-\alpha}}$')
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ax.text(x, y, label, ha='center', va='center',
                            fontsize=state_fontsize, zorder=5)
        if show_ghost:
            r = -1
            for x_val in range(1, max_x + 1, 2):
                x = x_val * dx
                y = -r * dy
                ax.add_patch(patches.Circle((x, y), R, edgecolor='gray', facecolor='white',
                               linestyle='--', linewidth=CIRCLE_LW, zorder=4))
                label = rf'${{{x_val}, \beta-\alpha+1}}$'
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    ax.text(x, y, label, ha='center', va='center',
                            fontsize=state_fontsize, zorder=5)

    def get_horiz_pt(x_val, r, is_start):
        angle = np.radians(65 if is_start else 115)
        return (x_val*dx + R*np.cos(angle), -r*dy + R*np.sin(angle))

    def get_diag_left_pt(x_val, r, is_start):
        angle = np.radians(225 if is_start else 45)
        return (x_val*dx + R*np.cos(angle), -r*dy + R*np.sin(angle))

    def get_vert_pt(x_val, r, is_start):
        angle = np.radians(270 if is_start else 90)
        return (x_val*dx + R*np.cos(angle), -r*dy + R*np.sin(angle))

    draw_half_lattice(ax_left, show_ghost=True)
    for r in range(num_levels):
        start_x = 0 if r % 2 == 0 else 1
        for x_val in range(start_x, max_x - 1, 2):
            p1 = get_horiz_pt(x_val, r, True)
            p2 = get_horiz_pt(x_val + 2, r, False)
            ax_left.add_patch(patches.FancyArrowPatch(p1, (p2[0]-0.15, p2[1]+0.175),
                              arrowstyle='-', connectionstyle='arc3,rad=-0.175',
                              color='tab:green', linewidth=ARROW_LW, zorder=4))
            ax_left.add_patch(patches.FancyArrowPatch(p1, p2, arrowstyle='-|>',
                              connectionstyle='arc3,rad=-0.3', mutation_scale=ARROW_MUTATION_SCALE,
                              color='tab:green', linewidth=0, zorder=3))
    for r in range(-1, num_levels - 1):
        x_vals = (range(1, max_x, 2) if r == -1
                  else (range(0, max_x + 1, 2) if r % 2 == 0
                        else range(1, max_x + 1, 2)))
        for x_val in x_vals:
            if x_val - 1 < 0:
                continue
            p1 = get_diag_left_pt(x_val, r, True)
            p2 = get_diag_left_pt(x_val - 1, r + 1, False)
            ax_left.add_patch(patches.FancyArrowPatch(p1, (p2[0]+0.1, p2[1]+0.1),
                              arrowstyle='-', color='r', linewidth=ARROW_LW, zorder=2))
            ax_left.add_patch(patches.FancyArrowPatch(p1, p2, arrowstyle='-|>',
                              mutation_scale=ARROW_MUTATION_SCALE, color='r', linewidth=0, zorder=1))

    draw_half_lattice(ax_right, show_ghost=True)
    p1 = get_diag_left_pt(1, -1, True)
    p2 = get_diag_left_pt(0, 0, False)
    ax_right.add_patch(patches.FancyArrowPatch(p1, (p2[0]+0.1, p2[1]+0.1),
                        arrowstyle='-', color='tab:blue', linewidth=ARROW_LW, zorder=2))
    ax_right.add_patch(patches.FancyArrowPatch(p1, p2, arrowstyle='-|>',
                        mutation_scale=ARROW_MUTATION_SCALE, color='tab:blue', linewidth=0, zorder=1))
    for r in range(0, num_levels - 2, 2):
        p1 = get_vert_pt(0, r, True)
        p2 = get_vert_pt(0, r + 2, False)
        ax_right.add_patch(patches.FancyArrowPatch(p1, p2, arrowstyle='-|>',
                            mutation_scale=ARROW_MUTATION_SCALE, color='tab:blue', linewidth=ARROW_LW, zorder=1))

    ax_mid.set_xlim(0, arrow_length)
    ax_mid.set_ylim(-1, 1)
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['right'].set_visible(False)
    ax_mid.spines['bottom'].set_visible(False)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.set_xticks([])
    ax_mid.set_yticks([])
    ax_mid.add_patch(patches.FancyArrowPatch((0,0), (arrow_length,0), arrowstyle='->',
                     mutation_scale=ARROW_MUTATION_SCALE, color='black', linewidth=ARROW_LW, zorder=10))
    ax_mid.text(arrow_length/2, 0.1, r'$g/k \searrow 0$', ha='center', va='bottom', fontsize=24, zorder=10)
    ax_mid.text(arrow_length/2, -0.1, r'$\alpha < \beta$', ha='center', va='top', fontsize=24, zorder=10)

    for ax in (ax_left, ax_right):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(SPINE_LW)
        ax.spines['left'].set_linewidth(SPINE_LW)
        ax.spines['bottom'].set_zorder(10)
        ax.spines['left'].set_zorder(10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'Number of $S_0$', fontsize=24, labelpad=20)
        ax.set_ylabel(r'Number of $R$', fontsize=24, labelpad=20)
        ax.scatter(xmax-0.1, ymin, marker='>', s=500, c='black', zorder=10, clip_on=False)
        ax.scatter(xmin, ymax-0.1, marker='^', s=500, c='black', zorder=10, clip_on=False)
        ax.set_aspect('equal')

draw_sim_scheme_A(axes['A_l'], axes['A_m'], axes['A_r'])
draw_sim_scheme_Cii(axes['B_l'], axes['B_m'], axes['B_r'])

label_fontsize = annotation_fontsize
label_offset_x_inch = 0.35
label_offset_y_inch = 0.12

top_y = margin_b + panel_h + gap_y + panel_h
bot_y = margin_b + panel_h

fig.text((margin_l - label_offset_x_inch)/fig_w,
         (top_y + label_offset_y_inch)/fig_h,
         'A', fontsize=label_fontsize, fontweight='bold', ha='right', va='bottom')
fig.text((margin_l - label_offset_x_inch)/fig_w,
         (bot_y + label_offset_y_inch)/fig_h,
         'B', fontsize=label_fontsize, fontweight='bold', ha='right', va='bottom')

plt.savefig('fig2_1r1s.pdf', dpi=600, bbox_inches='tight')
plt.show()