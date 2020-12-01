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

FixStyle(conp/dyn2,FixConpDyn2)

#else

#ifndef LMP_FIX_CONP_DYN2_H
#define LMP_FIX_CONP_DYN2_H

#include "fix.h"
#include "pair.h"
#include "fix_conpv3.h"

namespace LAMMPS_NS {

class FixConpDyn2 : public FixConpV3 {
 public:
  FixConpDyn2(class LAMMPS *lmp, int narg, char **arg):FixConpV3(lmp,narg,arg) {};
  ~FixConpDyn2() {}
  void setup(int);
  void b_cal();
 private:
  double *bk,*bkold,*vbk,*abk;
  double *bp,*bpold,*vbp,*abp;
  int bk_step, bk_interval, bk_status, bk_fails;
  int bp_step, bp_interval, bp_status, bp_fails;
};

}

#endif
#endif
