# lammps-USER-CONP2
updated constant potential plugin for LAMMPS

fixes contained in this package:

conp/v0 -- the original lammps-conp (https://github.com/zhenxingwang/lammps-conp),
    maintained for comparison purposes

conp/v1 -- (formerly conp4 in internal codes) lammps-conp, but now:
    - electroneutrality guaranteed via
        - constrained cg (see T. Gingrich thesis for pseudocode)
        - inv-a-matrix projected onto e-null space
        (using matrix "S" in Salanne 2020)
    - left- and right-voltages take equal-style variables for time-dependent change

conp/v2 + conq --
    pre-calculates the "pure electrode" charge vector q_c such that Aq_c = 1V x d
    (1V = "evscale" in LAMMPS units), then solves for the charge vector as:
    q*(DV) = q*(0) + DV x q_c, where Aq*(0) = b
    
    in conp mode, DV is Vleft - Vright (no second evscale conversion needed)
    in conq mode, DV is chosen such that q*.d = Q (as input, including equal-style)
    
    in either mode the fix now outputs the appropriate scalar (Q for conp, V for conq)
    also consolidates b-vector communication into b_comm() routine
    
    conq can be derived from the latest conp/v2+ class

conp/v3 -- uses LAMMPS utilities to manage these per-atom arrays:
    -- int* i2eleall: -1 if not electrode atom, else "eleall" numbering from v0
    -- double* arrelesetq: 0 if not electrode atom, else "elesetq" holding q_c from v2
    
    v0-2 b is allgathered, inside b_comm() routine in v2, and every proc simultaneously
    solves inv_A.b for the full b-vector.
    in v3, for inv solving, b_comm() keeps only local elements of the b-vector
    (while ordering them in bbb_all -- which we can do since i2eleall is known to all procs)
    
    inv_A.b is blocked column-wise (equivalent to row-wise, since inv_A is symmetric)
    and then the per-processor vectors are allreduced. This should parallelize the
    matrix multiplication relative to v2

conp/v4 -- uses BLAS calls to further optimize matrix-vector calculations
