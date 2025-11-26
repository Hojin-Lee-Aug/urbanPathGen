import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def setup_plot_style():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'

def plot_results(env, path1, path2, waypoints, scenario_name):
    fig, ax = plt.subplots(figsize=(12, 8))

    for b in env.building_hulls:
        rect = b["rect"]
        x_c = np.append(rect[:,0], rect[0,0])
        y_c = np.append(rect[:,1], rect[0,1])
        ax.plot(x_c, y_c, 'k-', linewidth=2)
        cx, cy = b["centroid"]
        ax.text(cx, cy, "B", ha='center', va='center', fontsize=16)

    if waypoints:
        ax.plot(waypoints[0][0], waypoints[0][1], 'go', markersize=10, label='SP')
        ax.plot(waypoints[-1][0], waypoints[-1][1], 'rx', markersize=10, mew=3, label='DE')
        for wp in waypoints[1:-1]:
            ax.plot(wp[0], wp[1], 'bs', markersize=8)

    if path1:
        p1 = np.array(path1)
        ax.plot(p1[:,0], p1[:,1], 'g-', linewidth=1.5, label='Opt 1')
    if path2:
        p2 = np.array(path2)
        ax.plot(p2[:,0], p2[:,1], 'r--', linewidth=1.5, label='Opt 2')

    ax.set_aspect('equal')
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_xlabel("$x/d_{ref}$")
    ax.set_ylabel("$y/d_{ref}$")
    ax.legend(loc='upper right', frameon=True)
    ax.set_title(f"Path Planning: {scenario_name}")

    plt.tight_layout()
    plt.savefig(f"{scenario_name.replace(' ', '_')}_result.png", dpi=300)
    plt.show()
