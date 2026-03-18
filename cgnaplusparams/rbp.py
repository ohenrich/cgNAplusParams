#!/bin/env python3

from __future__ import annotations

import numpy as np
import scipy as sp
from ._so3 import so3

from .cgnaplus_params import constructSeqParms, constructSeqParms_original
from .utils.assignment_utils import cgnaplus_name_assignment

from .utils.assignment_utils import INTER_BP_PARAM_NAME

def cgnaplus2rbp(
    sequence: str, 
    translations_in_nm: bool = True,
    euler_definition: bool = True,
    group_split: bool = True,
    parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
    remove_factor_five: bool = True,
    rotations_only: bool = False,
    include_stiffness: bool = True,
    ) -> dict[str, np.ndarray | bool | str]:
    
    gs,stiff = constructSeqParms(sequence,parameter_set_name)
    # gs,stiff = constructSeqParms_original(sequence,parameter_set_name)
    names = cgnaplus_name_assignment(sequence)
    select_names = [INTER_BP_PARAM_NAME+"*"]
    if include_stiffness:
        stiff = so3.matrix_marginal_assignment(stiff,select_names,names,block_dim=6)
        if sp.sparse.issparse(stiff):
            stiff = stiff.toarray()
    gs    = so3.vector_marginal_assignment(gs,select_names,names,block_dim=6)

    if remove_factor_five:
        factor = 5
        gs   = so3.array_conversion(gs,1./factor,block_dim=6,dofs=[0,1,2])
        if include_stiffness:
            stiff = so3.array_conversion(stiff,factor,block_dim=6,dofs=[0,1,2])
    
    if translations_in_nm:
        factor = 10
        gs   = so3.array_conversion(gs,1./factor,block_dim=6,dofs=[3,4,5])
        if include_stiffness:
            stiff = so3.array_conversion(stiff,factor,block_dim=6,dofs=[3,4,5])
    gs = so3.statevec2vecs(gs,vdim=6) 

    if euler_definition:
        # cayley2euler_stiffmat requires gs in cayley definition
        if include_stiffness:
            stiff = so3.se3_cayley2euler_stiffmat(gs,stiff,rotation_first=True)
        gs = so3.se3_cayley2euler(gs)

    if group_split:
        if not euler_definition:
            raise ValueError('The group_split option requires euler_definition to be set!')
        if include_stiffness:
            gs,stiff = so3.algebra2group_params(gs, stiff, rotation_first=True, translation_as_midstep=True, optimized=True) 
        else:
            gs = so3.midstep2triad(gs)

    if rotations_only:
        gs = so3.vector_rotmarginal(so3.vecs2statevec(gs))
        if include_stiffness:
            stiff = so3.matrix_rotmarginal(stiff)

    result = {
        "gs": gs,
        "sequence": sequence,
        "translations_in_nm": translations_in_nm,
        "euler_definition": euler_definition,
        "group_split": group_split,
        "remove_factor_five": remove_factor_five,
        "rotations_only": rotations_only,
    }
    if include_stiffness:
        result["stiffmat"] = stiff

    return result
