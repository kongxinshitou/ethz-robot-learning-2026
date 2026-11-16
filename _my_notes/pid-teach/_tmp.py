import numpy as np
from pid_demo import simulate
for noise in [0.004, 0.006]:
    t,r,_,u = simulate(kp=3.0, ki=1.0, kd=0.1, noise=noise, t_end=25)
    m = (t>12)&(t<20)
    print(noise, "u range", u[m].min(), u[m].max())
