import heapq
import numpy as np
import math
from .physics import *

GRID_CELL_SIZE = 0.04
BASE_DRONE_SPEED_FOR_PATHFINDING = 7.5

DIRECTIONS = [(0, GRID_CELL_SIZE), (0, -GRID_CELL_SIZE), (GRID_CELL_SIZE, 0), (-GRID_CELL_SIZE, 0),
              (GRID_CELL_SIZE, GRID_CELL_SIZE), (GRID_CELL_SIZE, -GRID_CELL_SIZE),
              (-GRID_CELL_SIZE, GRID_CELL_SIZE), (-GRID_CELL_SIZE, -GRID_CELL_SIZE)]

def calculate_step_cost(pos_a, pos_b, env, cost_type):
    pos_a = np.array(pos_a); pos_b = np.array(pos_b)
    seg_vec = pos_b - pos_a
    seg_len = np.linalg.norm(seg_vec)
    if seg_len == 0: return 0.0

    if cost_type == 'distance':
        return seg_len
    elif cost_type == 'energy':
        v_ground_vec = (seg_vec / seg_len) * BASE_DRONE_SPEED_FOR_PATHFINDING
        wind_A = env.get_wind(*pos_a)
        wind_B = env.get_wind(*pos_b)

        v_air_A = np.linalg.norm(v_ground_vec - wind_A)
        v_air_B = np.linalg.norm(v_ground_vec - wind_B)

        Re_A = reynolds_uav_from_v_air(v_air_A)
        Re_B = reynolds_uav_from_v_air(v_air_B)
        Cd_A = calculate_cd_for_circular_cylinder(Re_A) if v_air_A > 0 else 0.0
        Cd_B = calculate_cd_for_circular_cylinder(Re_B) if v_air_B > 0 else 0.0

        v_air_avg = (v_air_A + v_air_B) / 2.0
        Cd_avg = (Cd_A + Cd_B) / 2.0

        if v_air_avg <= 0: return 1e-9 * seg_len

        drag_force = 0.5 * RHO_AIR * CYLINDER_DIAMETER * Cd_avg * (v_air_avg**2)
        return max(drag_force * seg_len, 1e-9 * seg_len)
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")

def dijkstra_pathfinding(start, goal, env, cost_type):
    def snap(val): return np.round(val / GRID_CELL_SIZE) * GRID_CELL_SIZE
    start_grid = (snap(start[0]), snap(start[1]))
    goal_grid = (snap(goal[0]), snap(goal[1]))

    if env.is_obstacle(start_grid) or env.is_obstacle(goal_grid):
        print("Error: Start or Goal is inside an obstacle.")
        return None

    open_set = []
    heapq.heappush(open_set, (0.0, start_grid))
    came_from = {}
    g_score = {start_grid: 0.0}

    while open_set:
        current_g, current = heapq.heappop(open_set)
        if np.linalg.norm(np.array(current) - np.array(goal_grid)) < GRID_CELL_SIZE:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_grid)
            return path[::-1]

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)
            if not env.is_valid_pos(neighbor) or env.is_obstacle(neighbor): continue

            step_cost = calculate_step_cost(current, neighbor, env, cost_type)
            tentative_g = g_score.get(current, float('inf')) + step_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))
    return None

def calculate_total_energy(path, env, no_wind=False):
    if not path or len(path) < 2: return 0.0
    total_energy = 0.0
    for i in range(len(path)-1):
        a = np.array(path[i])
        b = np.array(path[i+1])
        seg_vec = b - a
        seg_len = np.linalg.norm(seg_vec)
        if seg_len <= 0: continue

        mid = (a + b) * 0.5
        v_ground_vec = (seg_vec / seg_len) * BASE_DRONE_SPEED_FOR_PATHFINDING

        if no_wind:
            wind_mid = np.array([0.0, 0.0])
        else:
            wind_mid = env.get_wind(mid[0], mid[1])

        v_air_vec = v_ground_vec - wind_mid
        v_air = np.linalg.norm(v_air_vec)

        Re_UAV = reynolds_uav_from_v_air(v_air)
        C_D = calculate_cd_for_circular_cylinder(Re_UAV)

        D_force = 0.5 * RHO_AIR * CYLINDER_DIAMETER * C_D * (v_air**2)
        total_energy += D_force * seg_len

    return total_energy

def compute_path_length(path):
    if not path or len(path) < 2: return 0.0
    total = 0.0
    for i in range(len(path)-1):
        a, b = np.array(path[i]), np.array(path[i+1])
        total += np.linalg.norm(b - a)
    return total
