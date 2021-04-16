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

#include "pointers.h"
#include "fix_conp.h"

namespace LAMMPS_NS{

class KSpaceModule: public Pointers {
 friend class FixConp;
 protected:
  KSpaceModule(class LAMMPS * lmp) : Pointers(lmp) {}
  virtual ~KSpaceModule() {}
  void register_fix(class FixConp* infix) {fixconp = infix;}
  virtual void setup() {}
  virtual void post_neighbor(bool, bool) {}
  virtual void a_cal(double *) {}
  virtual void a_read() {}
  virtual void b_cal(double *) {}

  class FixConp* fixconp;
};
}

#endif
