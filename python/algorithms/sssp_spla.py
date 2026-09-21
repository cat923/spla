import math
from pyspla import FLOAT, Matrix, Scalar, Vector
INF = math.inf


def make_vector(n):
    v = Vector(n, FLOAT)
    v.set_fill_value(Scalar(FLOAT, INF))
    return v

def sssp_spla(start: int, A: Matrix):
    n = A.n_rows
    dist = make_vector(n)
    dist.set(start, 0.0)
    mask = Vector.dense(n, FLOAT, 1.0)
    init_inf = Scalar(FLOAT, INF)
    while True:
        _prev_idx, prev_vals = dist.to_lists()
        new = make_vector(n)
        dist.vxm(mask, 
                 A,
                 op_mult=FLOAT.PLUS,
                 op_add=FLOAT.MIN,
                 op_select=FLOAT.ALWAYS,
                 init=init_inf,
                 out=new)
        out = make_vector(n)
        dist.eadd(FLOAT.MIN, new, out=out)
        dist = out
        _new_idx, new_vals = dist.to_lists()
        if prev_vals == new_vals:
            break
    return dist