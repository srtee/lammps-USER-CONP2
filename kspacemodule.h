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

#include "lammps.h"
#include "fix_conp.h"

namespace LAMMPS_NS{

class KspaceModule {
 public:
  KspaceModule(class LAMMPS *lmp, class FixConp *fix){
   fixconp = fix;
  }
  ~KspaceModule(){}
  virtual void setup(){}
  virtual void setup_allocate(){}
  virtual void elyte_allocate(int){}
  virtual void ele_allocate(int){}
  virtual void setup_deallocate(){}
  virtual void elyte_deallocate(){}
  virtual void ele_deallocate(){}
  virtual void sincos_a(){}
  virtual void aaa_from_sincos_a(double *){}
  virtual void sincos_b(){}
  virtual void bbb_from_sincos_b(double *){}
 protected:
  class FixConp * fixconp;
  char *kspmod_name;
};
}

#endif