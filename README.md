# LAMMPS-USER-CONP2
The updated constant potential plugin for LAMMPS

# Summary

The USER-CONP2 package allows users to perform LAMMPS MD simulations with constant potential electrodes. It updates the original LAMMPS-CONP (https://github.com/zhenxingwang/lammps-conp) with three main sets of performance optimizations:

1. Overall electroneutrality enforced via projection matrix precomputation
2. New PPPM KSpace styles and CONP interfaces, including with the Intel package
3. Smart neighborlisting (including `newton on` compatibility)
4. Extensive vectorization speedups in the old Ewald KSpace code

This is version 1.0 of the code. Upgrade priorities for version 1.1 are listed throughout this document in **boldface**; I welcome all feature suggestions!

# Installation instructions

Currently the easiest way to install USER-CONP2 is to copy all relevant .cpp/.h files into the LAMMPS src folder, and compile. Note that ``pppm_conp_intel.cpp/h`` require the USER-INTEL package to be compiled at the same time. **Work is in progress to wrap this all up into a ``USER-CONP2''-style folder together with CMake patches to streamline the process.**

# Usage instructions

The fix command is mostly identical to the previous version:

```
fix [ID] [all] conp [Nevery] [η] [Molecule-ID 1] [Molecule-ID 2] [Potential 1] [Potential 2] [Method] [Log] [optional keyword1] [optional value1] ...
```

`ID` = ID of FIX command

`[all]` = LAMMPS format requires you to list a valid group name here but it doesn't matter to the code

`Nevery` = Compute charge every this many steps

`η` = Parameter for Gaussian charge. The unit is is angstrom<sup>-1</sup>. Usually this is 1.979 for graphene.

`Molecule-ID 1` = Molecule ID of first electrode (the second column in data file)

`Molecule-ID 2` = Molecule ID of second electrode

`Potential 1` = Potential on first electrode in V (can be v_ style variable)

`Potential 2` = Potential on second electrode in V (can be v_ style variable)

`Method` = Method for solving linear equations. "inv" for inverse matrix and "cg" for conjugate gradient.

*The `cg` code branches have not been debugged and optimized as extensively as the `inv` branches, and their use is highly discouraged.*

`Log` = Name of log file recording time usage of different parts

## The optional keywords and values allowed are as follows:

### added in v1.0

`pppm`

Does what it says on the box!

To use `pppm`, the `pppm/conp` KSpace style must be used. This style is interchangeable with the LAMMPS `pppm` style in every other way but can interface with `fix conp` to enable PPPM calculations of the electrode-electrolyte interactions in KSpace. (Note that there is a `pppm/conp/intel` KSpace implementation, which works with `package intel` and will automatically be used when `suffix intel` is specified).

If the `pppm` flag is not specified, `fix conp` will use the default Ewald KSpace solver; if the `pppm` flag is specified but `pppm/conp` is not the KSpace style being used, LAMMPS will crash with a generic segfault message. **The eventual design goal is for the pppm flag to be removed and for `fix conp` to silently interface with `pppm/conp` when detected.**

### added in v0.9

`etypes [Ntypes] [type1] [type2] ... [typeN]`

This tells fix conp that the electrode _only_ contains the specified particle types, and the electrolyte _does not_ contain the specified particle types. For example, "etypes 2 5 3" is a promise to fix conp that the electrode only contains types 3 and 5. This allows fix conp to request specialized neighbor lists from LAMMPS: in the construction of the electrode "A" matrix the neighbor list will contain _only_ electrode-electrode interactions, and in the calculation of the electrolyte "b" vector the neighbor list will contain _only_ electrode-electrolyte interactions. This will speed up your calculation by 10-20% if correctly specified, and will **silently return incorrect results if the electrode and electrolyte particle types overlap in any way.**

`ffield [no values]`

This tells fix conp to run in finite-field mode as per Dufils (2019) [1], where (1) slab corrections are disabled and (2) the electrode preset charge vector is calculated using the electrodes' z coordinates. Configurations can be switched seamlessly between slab and finite-field mode, as long as the usual changes to the script are made ("boundary p p p" instead of "boundary p p f" and no "kspace_modify slab").

**_However_, you must add in the electric field separately: if fix conp is run in ffield mode, you _must_ include lines like these in your script:**

```
variable v_zfield equal (v_L-v_R)/lz
fix efield all efield 0.0 0.0 v_zfield
```

where v_L and v_R are the (possibly variable, or numerical) values of the left and right voltages. Note that fix efield already takes its argument in the correct units for _units real_ (V/angstroms) so no unit conversion should usually be needed. Fix conp in ffield mode will **silently return incorrect results without the electric field**; upgrading this is an urgent feature addition priority.

*Note: In [1], electrode atoms are mentioned as being "set at 0V", but the effect of this is already automatically done inside the conp ffield code. You do not specify electrode voltages in any different way whatsoever in the input script when using ffield.

`noslab [no values]`

This tells fix conp to run in no-slab-corrections mode, without further modifications. Run this with _zneutr_ (see below) unless you know what you are doing.

`zneutr [no values]`

This tells fix conp to enforce an additional electroneutrality constraint: the total sum of electrode charges in the right half of the box (_z_ > 0) will be zero. Thus, `noslab zneutr` enables fix conp to run a "doubled cell" simulation, in which two cells are positioned back-to-back with reversed polarities to create a zero dipole supercell -- see Raiteri (2020) [2] for a version of this simulation geometry, but with fixed charges.

`matout [no values]`

Causes fix conp to print out the A-matrix and the inverse A-matrix (if `inv` is used) to `amatrix` and `inv_a_matrix` respectively. The inverse matrix will already be projected into the space of electroneutral charge vectors (see [4], eq (21) for the expression, which they label as "S"). Note that the former conp would print out this matrix by default -- I have decided to make this capability opt-in because the A matrix reading code still has bugs lurking in it and the alternative of just computing the A matrix afresh is now very fast with the latest optimizations in v1.0.

`inv [inv_matrix_file]`

`org [org_matrix_file]`

This allows fix conp to read in a pre-existing electrode matrix for its calculations, either the A matrix ("org") or its inverse ("inv"). Option "org" will result in an electroneutral final matrix (since fix conp calculates the electroneutrality projection when inverting the A matrix), while option "inv" has no such guarantee. Use of these options are maintained for compatibility but is discouraged: there are still plenty of possible bugs lurking, and the code here does not use LAMMPS's latest file-reading functions, and optimization (notably calculating only half the A-matrix and then mirroring by symmetry) means that the A-matrix calculation is very short for all but the largest electrode configurations.

# Development: Other fixes included

The main branch contains two experimental fixes -- conp/dyn and conp/dyn2 -- which dynamically track the changes in the b-vector between timesteps and attempt to use a quadratic extrapolation to avoid having to recalculate everything. Conp/dyn tracks the overall b-vector discrepancies, while conp/dyn2 tracks the pair and kspace contributions separately. If you would like to try these out, let me know.

The _dev-historical_ branch contains versions frozen in various stages of optimization -- including the original conp/v0, with only semantic changes to bring it in line with the latest LAMMPS code -- to prove correctness of optimizations.

# Citations
    
[1] Dufils et al, PRL **123** (19): 195501 (2019)
https://doi.org/10.1103/PhysRevLett.123.195501

[2] Raiteri et al, J Chem Phys **153**: 164714 (2020)
https://doi.org/10.1063/5.0027876

[3] Zhang et al, J Phys Energy **2**: 032005 (2020)
https://doi.org/10.1088/2515-7655/ab9d8c

[4] Scalfi et al, Phys Chem Chem Phys **22**: 10480-10489 (2020)
https://doi.org/10.1039/C9CP06285H
