/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Version: Mar/2021
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#ifndef LMP_FIXCONP_KSPACEMODULE_H
#define LMP_FIXCONP_KSPACEMODULE_H

#include "fix_conp.h"

namespace LAMMPS_NS{

class KSpaceModule {
 public:
  KSpaceModule() {fixconp = nullptr;}
  virtual ~KSpaceModule() {}
  void register_fix(class FixConp* infix) {fixconp = infix;}
  virtual void conp_setup(bool) {}
  virtual void conp_post_neighbor(bool, bool) {}
  virtual void conp_pre_force() {}
  virtual void a_cal(double *) {}
  virtual void a_read() {}
  virtual void b_cal(double *) {}
  virtual void update_charge() {}
  virtual double compute_particle_potential(int) {return 0. ;}
  virtual void compute_group_potential(int, double* ) {}
  virtual double return_qsum() {return 0.;}

  class FixConp* fixconp;
 protected:
  bool lowmemflag;
};
}

#endif
