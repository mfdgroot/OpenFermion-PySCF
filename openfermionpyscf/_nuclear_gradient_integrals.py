#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Helper functions for AO gradient tensors (core, eri, s) """

import numpy as np
from pyscf import gto


def hcore_generator(mol: gto.Mole):
    """Generator for the core deriv function

    int1e_ipkin and int1e_ipnuc take the grad with respect to each
    basis function's atomic position x, y, z and place in a matrix.
    To get the gradient with respect to a particular atom we must
    add the columns of basis functions associated
    """
    aoslices = mol.aoslice_by_atom()
    h1 = mol.intor('int1e_ipkin', comp=3)  # (0.5 \nabla | p dot p | \)
    h1 += mol.intor('int1e_ipnuc', comp=3)  # (\nabla | nuc | \)
    h1 *= -1

    def hcore_deriv(atm_id):
        _, _, p0, p1 = aoslices[atm_id]
        """this part gets the derivative with respect to the electron-nuc
        operator. See pyscf docs for more info.
        (p|Grad_{Ra}Sum(M) 1/r_{e} - R_{M}|q)
        """
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(atm_id)
        vrinv[:, p0:p1] += h1[:, p0:p1]  # add the row's that aren't zero
        return vrinv + vrinv.transpose(0, 2, 1)

    return hcore_deriv


def overlap_generator(mol: gto.Mole):
    """Generator for the overlap derivfunction

    int1e_ipovlp takes the grad of the overlap
    with respect to each basis function's positions
    """
    aoslices = mol.aoslice_by_atom()
    s1 = mol.intor('int1e_ipovlp', comp=3)  # (\nabla \| \)

    def ovlp_deriv(atm_id):
        s_r = np.zeros_like(s1)
        _, _, p0, p1 = aoslices[atm_id]
        # row-idx indexes basis function.  All basis functions not on
        # a specific atom is zero.
        s_r[:, p0:p1] = s1[:, p0:p1]
        # (\nabla \| \ ) +  (\| \nabla)
        return s_r + s_r.transpose((0, 2, 1))

    return ovlp_deriv


def eri_generator(mol: gto.Mole):
    """Using int2e_ip1 = (nabla, | , )

    Remeber: chem notation (1*,1|2*,2) -> (ij|kl)

    NOTE: Prove the following is true through integral recursions

    (nabla i,j|kl) = (j,nablai|k,l) = (k,l|nabla i,j) = (k,l|j,nabla i)
    """
    aoslices = mol.aoslice_by_atom()
    eri_3 = mol.intor("int2e_ip1", comp=3)

    def eri_deriv(atm_id):
        eri_r = np.zeros_like(eri_3)
        _, _, p0, p1 = aoslices[atm_id]
        # take only the p0:p1 rows of the first index.
        # note we leverage numpy taking over all remaining jkl indices.
        # (p1 - p0, N, N, N) are non-zero
        eri_r[:, p0:p1] = eri_3[:, p0:p1]
        eri_r[:, :, p0:p1, :, :] += np.einsum('xijkl->xjikl', eri_3[:, p0:p1])
        eri_r[:, :, :, p0:p1, :] += np.einsum('xijkl->xklij', eri_3[:, p0:p1])
        eri_r[:, :, :, :, p0:p1] += np.einsum('xijkl->xklji', eri_3[:, p0:p1])
        return eri_r

    return eri_deriv


def grad_nuc(mol: gto.Mole, atmlst=None):
    '''
    Derivatives of nuclear repulsion energy wrt nuclear coordinates

    courtesy of pyscf and Szabo
    '''
    gs = np.zeros((mol.natm, 3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = np.sqrt(np.dot(r1-r2, r1-r2))
                gs[j] -= q1 * q2 * (r2-r1) / r**3
    if atmlst is not None:
        gs = gs[atmlst]
    return gs
