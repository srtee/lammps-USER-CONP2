/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Version: May/2021
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(potential/atom,ComputePotentialAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_POTENTIAL_ATOM_H
#define LMP_COMPUTE_POTENTIAL_ATOM_H

#include "compute.h"
#include "kspacemodule.h"

namespace LAMMPS_NS {

class ComputePotentialAtom : public Compute {
 public:
  ComputePotentialAtom(class LAMMPS *, int, char **);
  ~ComputePotentialAtom();
  void init() {}
  void setup();
  void compute_peratom();
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();
  void compute_pair_potential();

 private:
  int eta_check(int);
  void slabcorr();
  class KSpaceModule* kspmod;
  bool pairflag,kspaceflag,etaflag;
  int slabflag;
  int nmax;
  double evscale;
  double *potential;
  double eta;
  double volume;
  int molidL,molidR;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Per-atom energy was not tallied on needed timestep

You are using a thermo keyword that requires potentials to
have tallied energy, but they didn't on this timestep.  See the
variable doc page for ideas on how to make this work.

*/
