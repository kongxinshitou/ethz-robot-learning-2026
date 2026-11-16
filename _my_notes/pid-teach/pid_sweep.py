# -*- coding: utf-8 -*-
import numpy as np
from pid_demo import simulate

for kp, ki, umax in [(3, 2, 1.6), (3, 4, 1.6), (3, 6, 1.6),
                     (2, 4, 1.6), (2, 6, 1.6), (2, 8, 1.6)]:
    for aw in (False, True):
        t, r, y, u = simulate(kp=kp, ki=ki, sat=(0, umax), anti_windup=aw, t_end=40)
        peak = y.max()
        tpk = t[y.argmax()]
        tail = y[t > 35]
        print(f"kp={kp} ki={ki} umax={umax} aw={aw!s:<5} peak={peak:.2f}@t={tpk:.1f} "
              f"y(10)={np.interp(10,t,y):.2f} y(20)={np.interp(20,t,y):.2f} "
              f"settle_tail_pp={tail.max()-tail.min():.3f}")
    print()
