# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import math
import netCDF4 as nc
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from matplotlib.path import Path

def _rotation_matrix(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[c, -s], [s,  c]])

def oriented_bounding_box(points, theta_ref=None):
    pts = np.asarray(points)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    if theta_ref is None:
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        main_axis = eigvecs[:, np.argmax(eigvals)]
        theta_used = math.atan2(main_axis[1], main_axis[0])
    else:
        theta_used = theta_ref
    R = _rotation_matrix(-theta_used)
    pts_local = (R @ pts_centered.T).T
    min_x, max_x = pts_local[:,0].min(), pts_local[:,0].max()
    min_y, max_y = pts_local[:,1].min(), pts_local[:,1].max()
    rect_local = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    R_inv = _rotation_matrix(theta_used)
    rect_world = (R_inv @ rect_local.T).T + centroid
    return rect_world, centroid, theta_used

class UrbanFlowEnv:
    def __init__(self, info_file, pv_file, obstacle_padding=0.1):
        self.obstacle_padding = obstacle_padding
        self.building_hulls = []
        self.load_data(info_file, pv_file)
        self.process_buildings()

    def load_data(self, info_file, pv_file):
        print(f"Loading GCNN Data...\\n INFO: {info_file}\\n PV: {pv_file}")
        try:
            with nc.Dataset(info_file, 'r') as ds_info:
                self.x_coords = np.array(ds_info.variables['coord_0'][:])
                self.y_coords = np.array(ds_info.variables['coord_1'][:])
                self.bnd_data = np.array(ds_info.variables['bndCut'][:])
            with nc.Dataset(pv_file, 'r') as ds_pv:
                self.u_data = np.array(ds_pv.variables['variables0'][:])
                self.v_data = np.array(ds_pv.variables['variables1'][:])
            self.x_min, self.x_max = self.x_coords.min(), self.x_coords.max()
            self.y_min, self.y_max = self.y_coords.min(), self.y_coords.max()
            self.coords_array = np.vstack((self.x_coords, self.y_coords)).T
            self.tree = cKDTree(self.coords_array)
            print("Data loaded successfully.")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load NetCDF files: {e}")

    def process_buildings(self):
        boundary_indices = np.where(self.bnd_data == 1)[0]
        if len(boundary_indices) == 0: return
        is_on_outer_x = np.isclose(self.x_coords, self.x_min) | np.isclose(self.x_coords, self.x_max)
        is_on_outer_y = np.isclose(self.y_coords, self.y_min) | np.isclose(self.y_coords, self.y_max)
        is_not_outer = ~(is_on_outer_x | is_on_outer_y)
        final_indices = boundary_indices[is_not_outer[boundary_indices]]
        if len(final_indices) == 0: return
        building_coords = self.coords_array[final_indices]
        dbscan = DBSCAN(eps=0.2, min_samples=5)
        clusters = dbscan.fit_predict(building_coords)
        cluster_point_sets = []
        for cluster_id in np.unique(clusters):
            if cluster_id == -1: continue
            points = building_coords[clusters == cluster_id]
            if len(points) < 3: continue
            cluster_point_sets.append(points)
        if not cluster_point_sets: return
        _, _, theta_ref = oriented_bounding_box(cluster_point_sets[0], theta_ref=None)
        for points in cluster_point_sets:
            try:
                rect_world, centroid, _ = oriented_bounding_box(points, theta_ref=theta_ref)
                path = Path(rect_world)
                self.building_hulls.append({"path": path, "rect": rect_world, "centroid": centroid})
            except: continue
        print(f"Identified {len(self.building_hulls)} buildings.")

    def is_valid_pos(self, pos):
        x, y = pos
        return (self.x_min <= x < self.x_max) and (self.y_min <= y < self.y_max)

    def is_obstacle(self, pos):
        x, y = pos
        for b in self.building_hulls:
            if b["path"].contains_point((x, y), radius=self.obstacle_padding):
                return True
        return False

    def get_wind(self, x, y):
        if self.is_obstacle((x, y)): return np.array([0.0, 0.0])
        dist, idx = self.tree.query([x, y], k=1)
        return np.array([self.u_data[idx], self.v_data[idx]])
