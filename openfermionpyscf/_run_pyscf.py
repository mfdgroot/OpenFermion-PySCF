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

"""Driver to initialize molecular object from pyscf program."""

from __future__ import absolute_import

from functools import reduce

import numpy
from pyscf import gto, scf, ao2mo, ci, cc, fci, mp

from openfermion import MolecularData, general_basis_change
from openfermionpyscf import PyscfMolecularData
from ._nuclear_gradient_integrals import hcore_generator, \
    eri_generator, overlap_generator


def prepare_pyscf_molecule(molecule):
    """
    This function creates and saves a pyscf input file.

    Args:
        molecule: An instance of the MolecularData class.

    Returns:
        pyscf_molecule: A pyscf molecule instance.
    """
    pyscf_molecule = gto.Mole()
    pyscf_molecule.atom = molecule.geometry
    pyscf_molecule.basis = molecule.basis
    pyscf_molecule.spin = molecule.multiplicity - 1
    pyscf_molecule.charge = molecule.charge
    pyscf_molecule.symmetry = False
    pyscf_molecule.build()

    return pyscf_molecule


def compute_scf(pyscf_molecule):
    """
    Perform a Hartree-Fock calculation.

    Args:
        pyscf_molecule: A pyscf molecule instance.

    Returns:
        pyscf_scf: A PySCF "SCF" calculation object.
    """
    if pyscf_molecule.spin:
        pyscf_scf = scf.ROHF(pyscf_molecule)
    else:
        pyscf_scf = scf.RHF(pyscf_molecule)
    return pyscf_scf


def compute_integrals(pyscf_molecule, pyscf_scf):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(numpy.dot, (pyscf_scf.mo_coeff.T,
                                                 pyscf_scf.get_hcore(),
                                                 pyscf_scf.mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           pyscf_scf.mo_coeff)

    # No permutation symmetry
    two_electron_integrals = ao2mo.restore(
        1, two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = numpy.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals


def run_pyscf(molecule,
              run_scf=True,
              run_mp2=False,
              run_cisd=False,
              run_ccsd=False,
              run_fci=False,
              forces=False,
              verbose=False):
    """
    This function runs a pyscf calculation.

    Args:
        molecule: An instance of the MolecularData or PyscfMolecularData class.
        run_scf: Optional boolean to run SCF calculation.
        run_mp2: Optional boolean to run MP2 calculation.
        run_cisd: Optional boolean to run CISD calculation.
        run_ccsd: Optional boolean to run CCSD calculation.
        run_fci: Optional boolean to FCI calculation.
        forces: Optional boolean to get nuclear gradient matrix elements.
        verbose: Boolean whether to print calculation results to screen.

    Returns:
        molecule: The updated PyscfMolecularData object. Note the attributes
        of the input molecule are also updated in this function.
    """
    # Prepare pyscf molecule.
    pyscf_molecule = prepare_pyscf_molecule(molecule)
    molecule.n_orbitals = int(pyscf_molecule.nao_nr())
    molecule.n_qubits = 2 * molecule.n_orbitals
    molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

    # Run SCF.
    pyscf_scf = compute_scf(pyscf_molecule)
    pyscf_scf.verbose = 0
    pyscf_scf.run()
    molecule.hf_energy = float(pyscf_scf.e_tot)
    if verbose:
        print('Hartree-Fock energy for {} ({} electrons) is {}.'.format(
            molecule.name, molecule.n_electrons, molecule.hf_energy))

    # Hold pyscf data in molecule. They are required to compute density
    # matrices and other quantities.
    molecule._pyscf_data = pyscf_data = {}
    pyscf_data['mol'] = pyscf_molecule
    pyscf_data['scf'] = pyscf_scf

    # Populate fields.
    molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)

    # Get integrals.
    one_body_integrals, two_body_integrals = compute_integrals(
        pyscf_molecule, pyscf_scf)
    molecule.one_body_integrals = one_body_integrals
    molecule.two_body_integrals = two_body_integrals
    molecule.overlap_integrals = pyscf_scf.get_ovlp()

    # Run MP2.
    if run_mp2:
        if molecule.multiplicity != 1:
            print("WARNING: RO-MP2 is not available in PySCF.")
        else:
            pyscf_mp2 = mp.MP2(pyscf_scf)
            pyscf_mp2.verbose = 0
            pyscf_mp2.run()
            # molecule.mp2_energy = pyscf_mp2.e_tot  # pyscf-1.4.4 or higher
            molecule.mp2_energy = pyscf_scf.e_tot + pyscf_mp2.e_corr
            pyscf_data['mp2'] = pyscf_mp2
            if verbose:
                print('MP2 energy for {} ({} electrons) is {}.'.format(
                    molecule.name, molecule.n_electrons, molecule.mp2_energy))

    # Run CISD.
    if run_cisd:
        pyscf_cisd = ci.CISD(pyscf_scf)
        pyscf_cisd.verbose = 0
        pyscf_cisd.run()
        molecule.cisd_energy = pyscf_cisd.e_tot
        pyscf_data['cisd'] = pyscf_cisd
        if verbose:
            print('CISD energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.cisd_energy))

    # Run CCSD.
    if run_ccsd:
        pyscf_ccsd = cc.CCSD(pyscf_scf)
        pyscf_ccsd.verbose = 0
        pyscf_ccsd.run()
        molecule.ccsd_energy = pyscf_ccsd.e_tot
        pyscf_data['ccsd'] = pyscf_ccsd
        if verbose:
            print('CCSD energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.ccsd_energy))

    # Run FCI.
    if run_fci:
        pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
        pyscf_fci.verbose = 0
        molecule.fci_energy = pyscf_fci.kernel()[0]
        pyscf_data['fci'] = pyscf_fci
        if verbose:
            print('FCI energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.fci_energy))

    # Get gradients
    if forces:
        one_body_force_integrals, \
            two_body_force_integrals = calculate_force_integrals(
                                        pyscf_molecule, pyscf_scf)
        pyscf_data['forces'] = {'f1': one_body_force_integrals,
                                'f2': two_body_force_integrals}
        molecule._one_body_force_integrals = one_body_force_integrals
        molecule._two_body_force_integrals = two_body_force_integrals

    # Return updated molecule instance.
    pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    pyscf_molecular_data.__dict__.update(molecule.__dict__)
    pyscf_molecular_data.save()

    return pyscf_molecular_data


def generate_molecular_hamiltonian(
        geometry,
        basis,
        multiplicity,
        charge=0,
        n_active_electrons=None,
        n_active_orbitals=None,
        data_directory=None):
    """Generate a molecular Hamiltonian with the given properties.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to
            specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
            Only optional if loading from file.
        multiplicity: An integer giving the spin multiplicity.
        charge: An integer giving the charge.
        n_active_electrons: An optional integer specifying the number of
            electrons desired in the active space.
        n_active_orbitals: An optional integer specifying the number of
            spatial orbitals desired in the active space.

    Returns:
        The Hamiltonian as an InteractionOperator.
    """

    # Run electronic structure calculations
    molecule = run_pyscf(
            MolecularData(
                geometry, basis, multiplicity, charge,
                data_directory=data_directory)
    )

    # Freeze core orbitals and truncate to active space
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(range(n_core_orbitals,
                                    n_core_orbitals + n_active_orbitals))

    return molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices)


def calculate_force_integrals(
        pyscf_mol, pyscf_scf, atmlst=None, hcore_mo=None, tei_mo=None):
    r"""
        Obtain gradient operator integrals in physicist ordering

    Args:
        pyscf_mol (pyscf.Mole): Object holding AO integrals.
        pyscf_scf (pyscf.scf): SCF object holding the molecular orbital
            coefficients.

    Returns:
        f1mos (ndarray): An N_a by 3 by N by N array storing the one-body
            part of the force operators.
        f2mos (ndarray): An N_a by 3 by N by N by N by N array storing the
            two-body part of the force operators.
    """
    norbs = pyscf_mol.nao

    hcore_deriv = hcore_generator(pyscf_mol)
    ovrlp_deriv = overlap_generator(pyscf_mol)
    eri_deriv = eri_generator(pyscf_mol)

    if atmlst is None:
        atmlst = range(pyscf_mol.natm)

    if hcore_mo is None:
        hcore_mo = general_basis_change(
                    pyscf_mol.get_hcore(), pyscf_scf.mo_coeff, key=(1, 0)
                    )

    if tei_mo is None:
        tei_mo_compressed = ao2mo.kernel(pyscf_mol, pyscf_scf.mo_coeff)
        tei_mo = ao2mo.restore(1, tei_mo_compressed, norbs)

    f1mos = numpy.zeros((len(atmlst), 3, norbs, norbs))
    f2mos = numpy.zeros((len(atmlst), 3, norbs, norbs, norbs, norbs))

    for k, ia in enumerate(atmlst):
        h1ao = hcore_deriv(ia)
        s1ao = ovrlp_deriv(ia)
        eriao = eri_deriv(ia)
        s1mo = numpy.zeros_like(s1ao)

        for xyz in range(3):
            # Core-MO - Hellmann-Feynman term
            f1mos[k, xyz] = general_basis_change(
                            h1ao[xyz], pyscf_scf.mo_coeff, key=(1, 0)
                            )

            # X-S-MO
            s1mo[xyz] = general_basis_change(
                            s1ao[xyz], pyscf_scf.mo_coeff, key=(1, 0)
                            )

            # one-body part of wavefunction force
            f1mos[k, xyz] += 0.5 * (
                            numpy.einsum('pj,ip->ij', hcore_mo, s1mo[xyz]) +
                            numpy.einsum('ip,jp->ij', hcore_mo, s1mo[xyz])
                            )

            # eriao in openfermion ordering Hellmann-Feynman term
            f2mos[k, xyz] -= general_basis_change(
                                eriao[xyz], pyscf_scf.mo_coeff,
                                key=(1, 0, 1, 0)
                                ).transpose((0, 2, 3, 1))

            # two-body part of wavefunction force
            f2mos[k, xyz] += 0.5 * (
                                numpy.einsum('px,xqrs', s1mo[xyz], tei_mo) +
                                numpy.einsum('qx,pxrs', s1mo[xyz], tei_mo) +
                                numpy.einsum('rx,pqxs', s1mo[xyz], tei_mo) +
                                numpy.einsum('sx,pqrx', s1mo[xyz], tei_mo)
                                )

    return f1mos, f2mos


def generate_nuclear_forces(
        geometry,
        basis,
        multiplicity,
        charge=0,
        data_directory=None):
    """Generate nuclear gradient operators.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to
            specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
            Only optional if loading from file.
        multiplicity: An integer giving the spin multiplicity.
        charge: An integer giving the charge.

    Returns:
        A list of lists of forces as InteractionOperator.
    """

    # Run electronic structure calculations
    molecule = run_pyscf(
                        MolecularData(
                            geometry, basis, multiplicity, charge,
                            data_directory=data_directory
                            ),
                        forces=True
                    )

    return molecule.get_nuclear_forces()
