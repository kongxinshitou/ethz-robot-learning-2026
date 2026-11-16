import numpy as np
from scipy.optimize import minimize
import ik6_planar as m
q0 = np.deg2rad([45.]*6)
cons = [{"type":"eq","fun":lambda q: m.fk(q)[-1]-m.P_DES}]
r = minimize(lambda q: np.sum((q-q0)**2), q0, constraints=cons, method="SLSQP",
             options={"maxiter":500,"ftol":1e-12})
print(r.success, r.message)
print("dist", np.linalg.norm(r.x-q0), "err_mm", np.linalg.norm(m.fk(r.x)[-1]-m.P_DES)*1000)
print(np.round(np.rad2deg(r.x),1))
