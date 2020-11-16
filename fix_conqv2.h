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
   Version: Nov/2020
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(conq,FixConqV3)

#else

#ifndef LMP_FIX_CONQV3_H
#define LMP_FIX_CONQV3_H

#include "fix.h"
#include "fix_conpv3.h"

namespace LAMMPS_NS {

class FixConqV3 : public FixConpV3 {
 public:
  FixConqV3(class LAMMPS *lmp, int narg, char **arg):FixConpV3(lmp,narg,arg) {};
  ~FixConqV3() {}
  void update_charge();
 };

}

#endif
#endif
