import argparse
import numpy as np
from src.environment import UrbanFlowEnv
from src.pathfinding import dijkstra_pathfinding, calculate_total_energy, compute_path_length
from src.visualization import setup_plot_style, plot_results

def main():
    info_file = "data/Info.Netcdf"
    pv_file = "data/PV.Netcdf"

    print(f"Processing with: {info_file}, {pv_file}")

    try:
        env = UrbanFlowEnv(info_file, pv_file)
    except Exception as e:
        print(f"[Error] data load fail: {e}")
        print("Please check that there are NetCDF files in data folder.")
        return

    # Coordinate SP, DE, WPs
    START_POS = np.array([0.4, -1.6])
    GOAL_POS = np.array([2.5, 0.5])
    STOPOVER_POS_1 = np.array([-1, 2.5])
    STOPOVER_POS_2 = np.array([-0.5, -2.5])
    STOPOVER_POS_3 = np.array([2.5, -2.5])

    scenarios = {
        'Scenario (1 Stopover)': [START_POS, STOPOVER_POS_1, GOAL_POS],
        'Scenario (2 Stopovers)': [START_POS, STOPOVER_POS_1, STOPOVER_POS_2, GOAL_POS],
        'Scenario (3 Stopovers)': [START_POS, STOPOVER_POS_1, STOPOVER_POS_2, STOPOVER_POS_3, GOAL_POS],
    }

    setup_plot_style()

    for name, waypoints in scenarios.items():
        print(f"\\n==================== {name} ====================")

        # --- Opt 1 ---
        print(f"--- {name} - Opt 1 (Distance) ---")
        full_path1 = []
        is_path1_found = True
        for i in range(len(waypoints)-1):
            seg = dijkstra_pathfinding(waypoints[i], waypoints[i+1], env, 'distance')
            if seg: full_path1.extend(seg if i == 0 else seg[1:])
            else:
                is_path1_found = False
                print(f"Path finding failed at segment {i}")
                break

        if is_path1_found:
            print(f"Path found. Steps: {len(full_path1)}")

        # --- Opt 2 ---
        print(f"--- {name} - Opt 2 (Work) ---")
        full_path2 = []
        is_path2_found = True
        for i in range(len(waypoints)-1):
            seg = dijkstra_pathfinding(waypoints[i], waypoints[i+1], env, 'energy')
            if seg: full_path2.extend(seg if i == 0 else seg[1:])
            else:
                is_path2_found = False
                print(f"Path finding failed at segment {i}")
                break

        if is_path2_found:
            print(f"Path found. Steps: {len(full_path2)}")

        dist1 = compute_path_length(full_path1) if is_path1_found else None
        dist2 = compute_path_length(full_path2) if is_path2_found else None

        e1 = calculate_total_energy(full_path1, env) if is_path1_found else None
        e2 = calculate_total_energy(full_path2, env) if is_path2_found else None

        e_no_wind = calculate_total_energy(full_path1, env, no_wind=True) if is_path1_found else None

        print(f"\\n[Results for {name}]")
        if dist1: print(f"  Opt 1 Distance: {dist1:.3f} m")
        if dist2: print(f"  Opt 2 Distance: {dist2:.3f} m")

        if e_no_wind: print(f"  No wind_Energy:   {e_no_wind:.2f}")
        if e1: print(f"  Opt 1 Energy:     {e1:.2f}")
        if e2: print(f"  Opt 2 Energy:     {e2:.2f}")

        if e1 and e2 and e_no_wind and e_no_wind > 0:
            print(f"  Opt 1 Non_dimensional_Energy: {e1/e_no_wind:.4f}")
            print(f"  Opt 2 Non_dimensional_Energy: {e2/e_no_wind:.4f}")
            savings = ((e1 - e2) / e1) * 100
            print(f"  Energy Savings: {savings:.2f}%")

        plot_results(env, full_path1, full_path2, waypoints, name)

if __name__ == "__main__":
    main()
