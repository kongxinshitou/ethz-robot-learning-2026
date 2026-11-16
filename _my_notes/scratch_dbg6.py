import numpy as np, importlib, ik6_planar as m
tgt = np.array([30.,20.])
for qf in ([50,30,15,25],[55,35,20,30],[45,25,15,20],[50,30,15,40]):
    q_fix = np.deg2rad(qf+[0,0])
    p4 = m.fk(q_fix)[4]; v = tgt-p4
    print(qf, np.round(v,2), np.linalg.norm(v))
    for t5,t6 in m.solve_2r(v, m.L[4], m.L[5]):
        print("   ", round(np.rad2deg(t5),1), round(np.rad2deg(t6),1))
