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
#include "fix_conp.h"

namespace LAMMPS_NS {

class FixConpDyn2 : public FixConp {
 public:
  FixConpDyn2(class LAMMPS *lmp, int narg, char **arg):FixConp(lmp,narg,arg) {};
  ~FixConpDyn2();
  void dyn_setup();
  void b_cal();
  void update_bp();
  double update_dynv(double*, double*, int*, int);
  void update_from_dynv(double*, double*);
 private:
  double *bk,*bkvec;
  double *bp,*bpvec;
  int bk_step, bk_interval, bk_status, bk_fails;
  int bp_step, bp_interval, bp_status, bp_fails;
  int bk_cap, bk_report, bk_reportevery;
  int bp_cap, bp_report, bp_reportevery;
  int bk_maxfails, bp_maxfails;
  double bk_runerr, bp_runerr;
};

}

#endif
#endif
