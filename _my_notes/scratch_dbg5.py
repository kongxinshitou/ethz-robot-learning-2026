import numpy as np
import ik6_planar as m
QMIN, QMAX = 0, np.deg2rad(170)
def fam_count(q_base_deg, tgt):
    q_base = np.deg2rad(q_base_deg)
    p3 = m.fk(np.concatenate([q_base,[0,0,0]]))[3]
    n=0; q4s=[]
    for q6 in np.deg2rad(np.arange(-120,150,4)):
        w = tgt - p3 - m.L[5]*np.array([np.cos(q6),np.sin(q6)])
        for t4,t5 in m.solve_2r(w, m.L[3], m.L[4]):
            q = np.concatenate([q_base,[t4,t5,q6]])
            if np.all(q>=QMIN-1e-9) and np.all(q<=QMAX):
                n+=1; q4s.append(np.rad2deg(t4))
    return n, (min(q4s),max(q4s)) if q4s else None
for base in ([60,40,25],[55,35,20],[50,30,15],[45,25,10],[40,20,10],[35,15,10]):
    for tgt in ([28,15],[30,18],[26,16],[30,20]):
        print(base,tgt,fam_count(base,np.array(tgt,float)))
