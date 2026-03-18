#!/bin/env python3

from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix
from ._so3 import so3

from .cgnaplus_params import constructSeqParms, constructSeqParms_original
from .utils.assignment_utils import cgnaplus_name_assignment, nonphosphate_dof_map
from .utils.assignment_utils import crick_phosphate_dof_indices
from .utils.crick_flip import apply_crick_flip



def cgnaplusparams(
    sequence: str, 
    parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
    euler_definition: bool = True,
    group_split: bool = True,
    remove_factor_five: bool = True,
    translations_in_nm: bool = True,
    include_stiffness: bool = True,
    aligned_strands: bool = False
    ) -> dict[str, np.ndarray | bool | str]:
    
    gs, stiff = constructSeqParms(sequence, parameter_set_name)

    param_names = cgnaplus_name_assignment(sequence)
    nonphosphate_map = nonphosphate_dof_map(sequence, param_names=param_names)

    if aligned_strands:
        gs, stiff = apply_crick_flip(
            gs,
            stiff if include_stiffness else None,
            param_names,
        )

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
            
            gs,stiff = so3.algebra2group_params(gs, stiff, rotation_first=True, translation_as_midstep=nonphosphate_map, optimized=True) 
        else:
            gs = so3.midstep2triad(gs)

    result = {
        "gs": gs,
        "sequence": sequence,
        "translations_in_nm": translations_in_nm,
        "euler_definition": euler_definition,
        "group_split": group_split,
        "remove_factor_five": remove_factor_five,
        "param_names": param_names,
        "aligned_strands": aligned_strands
    }
    if include_stiffness:
        result["stiffmat"] = stiff

    return result



class CGNAPlus:

    def __init__(
            self, 
            sequence: str, 
            parameter_set_name: str = "Prmset_cgDNA+_CGF_10mus_int_12mus_ends",
            euler_definition: bool = True,
            group_split: bool = True,
            translations_in_nm: bool = True,
            aligned_strands: bool = False
            ):

        self.params = cgnaplusparams(
            sequence=sequence,
            parameter_set_name=parameter_set_name,
            euler_definition=euler_definition,
            group_split=group_split,
            remove_factor_five=True,
            translations_in_nm=translations_in_nm,
            include_stiffness=True,
            aligned_strands=aligned_strands
        )    

        self.parameter_set_name = parameter_set_name
        self.euler_definition = euler_definition
        self.group_split = group_split
        self.translations_in_nm = translations_in_nm

        self.stiffmat = self.params["stiffmat"]
        self.gs = self.params["gs"]
        self.sequence = sequence 
        self.param_names = self.params["param_names"]


    



