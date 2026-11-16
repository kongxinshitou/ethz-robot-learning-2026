import numpy as np
from scipy.optimize import minimize
import ik6_planar as m
b = [(0, np.deg2rad(170))]*6
for tgt in [(30,14),(28,18),(26,20),(30,20)]:
    m.P_DES = np.array(tgt, float)
    for s in ([45]*6, [60]*6, [70,50,45,40,35,30]):
        q0 = np.deg2rad(s)
        r = minimize(lambda q: np.sum((q-q0)**2), q0,
                     constraints=[{"type":"eq","fun":lambda q: m.fk(q)[-1]-m.P_DES}],
                     bounds=b, method="SLSQP", options={"maxiter":300,"ftol":1e-10})
        q = r.x
        err = np.linalg.norm(m.fk(q)[-1]-m.P_DES)*1000
        print(tgt, s, r.success, f"err={err:.1e}mm dist={np.linalg.norm(q-q0):.2f}", np.round(np.rad2deg(q),1))
