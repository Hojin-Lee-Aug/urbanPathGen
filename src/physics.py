import numpy as np
from scipy.interpolate import interp1d

RHO_AIR = 1.225     # kg/m^3
NU_AIR  = 1.5e-5    # m^2/s

Re_FLOW = 5000.0
D_REF   = 1.0
CYLINDER_DIAMETER = 0.1 * D_REF  # d_UAV

# $C_d$ table
_RE_CD_COORDS = np.array([
    (1.448312088,  9.089759777), (1.696578405,  8.246488576),
    (2.090228755,  7.198769046), (2.583101459,  6.336485910),
    (3.349665131,  5.407413844), (4.294199386,  4.615250507),
    (5.370008058,  4.130357275), (6.871094129,  3.582841589),
    (8.774997601,  3.156776034), (11.51028424,  2.865821088),
    (14.64358633,  2.550032233), (18.62982844,  2.286172051),
    (23.97448978,  2.063884317), (30.85246664,  1.888448491),
    (39.70364777,  1.729927116), (51.09412043,  1.640841945),
    (65.75237515,  1.552348837), (84.61589711,  1.479572654),
    (108.8911241,  1.431954455), (140.1306057,  1.393957523),
    (180.3322981,  1.360669424), (232.0673458,  1.324757304),
    (298.6445221,  1.279173711), (384.3218453,  1.232758218),
    (494.5789053,  1.194399321), (636.4673165,  1.143064269),
    (819.0617122,  1.083874119), (1054.040123,  1.029750154),
    (1356.430857,  0.999650825), (1745.573655,  0.989012561),
    (2246.356584,  1.007459548), (2890.807779,  1.041332906),
    (3720.143843,  1.077916659), (4787.405899,  1.114158964),
    (6160.851895,  1.141582847), (7928.322115,  1.141397805),
    (9634.434583,  1.157452893),
], dtype=float)

RE_POINTS = _RE_CD_COORDS[:, 0]
CD_POINTS = _RE_CD_COORDS[:, 1]
MIN_RE_DATASET = float(RE_POINTS.min())
MAX_RE_DATASET = float(RE_POINTS.max())

# log10(Re) linear interpolation
_f_cd_loglin = interp1d(
    np.log10(RE_POINTS), CD_POINTS, kind="linear",
    bounds_error=False, fill_value=(CD_POINTS[0], CD_POINTS[-1]), assume_sorted=True
)

def reynolds_uav_from_v_air(v_air):
    v_air = float(v_air)
    if v_air <= 0.0: return 0.0
    return 500.0 * v_air

def calculate_cd_for_circular_cylinder(reynolds_number):
    Re = np.asarray(reynolds_number, dtype=float)
    Re = np.where(Re <= 0.0, MIN_RE_DATASET, Re)
    cd = _f_cd_loglin(np.log10(Re))
    return float(cd) if np.isscalar(reynolds_number) else cd
