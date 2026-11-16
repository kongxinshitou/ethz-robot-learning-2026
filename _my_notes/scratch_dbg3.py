import numpy as np, importlib
import ik6_planar as m
q0 = np.deg2rad([45.]*6)
for tgt in [(30,14),(30,20),(28,22),(25,20),(32,18)]:
    m.P_DES = np.array(tgt, float)
    q = q0.copy()
    for k in range(400):
        e = m.P_DES - m.fk(q)[-1]
        if np.linalg.norm(e)<1e-12: break
        J = m.jac(q); Jp = J.T@np.linalg.inv(J@J.T+1e-6*np.eye(2))
        dq = Jp@e + (np.eye(6)-Jp@J)@(-0.4*(q-q0))
        n=np.max(np.abs(dq))
        if n>0.15: dq*=0.15/n
        q = np.clip(q+dq, 0, np.deg2rad(170))
    print(tgt, f"dist={np.linalg.norm(q-q0):.3f}", f"err={np.linalg.norm(m.fk(q)[-1]-m.P_DES)*1000:.2e}", np.round(np.rad2deg(q),1))
