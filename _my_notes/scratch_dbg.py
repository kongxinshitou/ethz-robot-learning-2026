import numpy as np, ik6_planar as m
q0 = np.deg2rad([45.]*6)
q = q0.copy()
for k in range(12):
    e = m.P_DES - m.fk(q)[-1]
    J = m.jac(q)
    Jp = J.T @ np.linalg.inv(J @ J.T + 1e-6*np.eye(2))
    g = -0.4*min(1.0, np.linalg.norm(e)*1000)*(q-q0)
    dq = Jp@e + (np.eye(6)-Jp@J)@g
    n = np.max(np.abs(dq))
    if n > 0.15: dq *= 0.15/n
    q = np.clip(q+dq, 0, np.deg2rad(170))
    print(k, f"err={np.linalg.norm(m.fk(q)[-1]-m.P_DES)*1000:10.3f}mm", f"dist2q0={np.linalg.norm(q-q0):.3f}", np.round(np.rad2deg(q),1))
