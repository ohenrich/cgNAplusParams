#!/bin/env python3

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from ..cgnaplus_params import constructSeqParms, constructSeqParms_original
from .assignment_utils import cgnaplus_name_assignment, nonphosphate_dof_map
from .assignment_utils import crick_phosphate_dof_indices


def apply_crick_flip(
    gs: np.ndarray,
    stiff: csc_matrix | None,
    param_names: list[str],
) -> tuple[np.ndarray, csc_matrix | None]:
    """Flip the orientation of Crick base-to-phosphate junction coordinates.

    Within each C-type 6-DOF block the sign of coordinates 0, 1, 3, 4 is
    negated (coordinates 2 and 5 are kept).  For the ground state this is a
    direct negation; for the stiffness the congruence K' = S K S
    (S = diag(sign_vector)) is applied via an O(nnz) elementwise multiply on
    the CSC data array – no index copies required.

    Parameters
    ----------
    gs:
        Raw Cayley ground-state vector of length N.
    stiff:
        CSC sparse stiffness matrix, or None when stiffness is not needed.
    param_names:
        List of parameter names.

    Returns
    -------
    gs_flipped, stiff_flipped
    """
    crick_idx = crick_phosphate_dof_indices(param_names)

    flip_within = np.array([0, 1, 3, 4])  # offsets inside each 6-DOF block
    raw_flip = (crick_idx[:, None] * 6 + flip_within).ravel()

    gs_flipped = gs.copy()
    gs_flipped[raw_flip] *= -1.0

    if stiff is None:
        return gs_flipped, None

    N = len(gs)
    s = np.ones(N, dtype=np.float64)
    s[raw_flip] = -1.0

    # Congruence S K S: multiply each stored value by s[row] * s[col]
    stiff_data = stiff.data * s[stiff.indices] * np.repeat(s, np.diff(stiff.indptr))
    stiff_flipped = csc_matrix(
        (stiff_data, stiff.indices, stiff.indptr),
        shape=stiff.shape,
        copy=False,
    )
    return gs_flipped, stiff_flipped