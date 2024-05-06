from dolfin import *
import scipy.sparse as sparse
import numpy        as np
from petsc4py     import PETSc
import os


def check_if_file_exists(file_path):

    if not os.path.exists(file_path):        
        print(f"The file '{file_path}' does not exist.")
        exit()

def norm_2(vec):
    return sqrt(dot(vec,vec))


def dump(thing, path):
    if isinstance(thing, PETSc.Vec):
        assert np.all(np.isfinite(thing.array))
        return np.save(path, thing.array)
    m = sparse.csr_matrix(thing.getValuesCSR()[::-1]).tocoo()
    assert np.all(np.isfinite(m.data))
    return np.save(path, np.c_[m.row, m.col, m.data])


def get_adaptive_dt(dt_old, iterations):
        
    its_high = 4
    its_low  = 2

    # adaptivity parameters
    if iterations >= its_high:
        dt_new = dt_old / 1.5
    elif iterations <= its_low:
        dt_new = dt_old * 1.1
    else:
        dt_new = dt_old    

    # correct in case of extreme values
    max_dt = 0.0002
    min_dt = 0.00005        
    
    if dt_new > max_dt:
        dt_new = max_dt
    elif dt_new < min_dt:
        dt_new = min_dt

    return dt_new
