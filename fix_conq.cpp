/* ---------------------------------------------------------------------
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

#include "fix_conq.h"
#include "atom.h"
#include "input.h"
#include "variable.h"

using namespace LAMMPS_NS;

enum{CONSTANT,EQUAL,ATOM};
enum{CG,INV};
enum{NORMAL,FFIELD,NOSLAB};
extern "C" {
  double ddot_(const int *N, const double *SX, const int *INCX, const double *SY, const int *INCY);
}

FixConq::FixConq(LAMMPS *lmp, int narg, char **arg):
  FixConp(lmp, narg, arg)
{
  rightchargevar = potdiffvar;
  rightchargestyle = potdiffstyle;
  rightcharge = potdiff;
}

void FixConq::update_charge()
{
  int leftchargevar = potdiffvar;
  int i,j,idx1d,iall,jall,iloc;
  int elealli,tagi;
  double eleallq_i;
  int *tag = atom->tag;
  int const nlocal = atom->nlocal;
  int const nall = nlocal+atom->nghost;
  double netcharge_right = 0;
  double *q = atom->q;    
  int const elenum_c = elenum;
  int const elenum_all_c = elenum_all;
  if (minimizer == 1) {
    idx1d = 0;
    int one = 1;
    for (iloc = 0; iloc < elenum_c; ++iloc) {
      iall = ele2eleall[iloc];
      idx1d = iall*elenum_all;
      bbb[iloc] = ddot_(&elenum_all,&aaa_all[idx1d],&one,bbb_all,&one);
    }
    b_comm(bbb,eleallq);
  } // if minimizer == 0 then we already have eleallq ready;

  if (rightchargestyle == EQUAL) rightcharge = input->variable->compute_equal(rightchargevar);
  //  now qL and qR are left and right *voltages*
  //  evscale was included in the precalculation of eleallq

  //  update charges including additional charge needed
  //  this fragment is the only difference from fix_conp
  for (iall = 0; iall < elenum_all_c; ++iall) {
    if (elecheck_eleall[iall] == 1) netcharge_right -= eleallq[iall];
  }
  
  potdiff = scalar_output = (rightcharge - rightcharge_left)/totsetq;
  
  for (iall = 0; iall < elenum_all_c; ++iall) {
    i = atom->map(eleall2tag[iall]);
    if (i != -1) {
      q[i] = eleallq[iall] + potdiff*elesetq[iall];
      if (qinitflag) q[i] += eleinitq[iall];
    }
  } // we need to loop like this to correctly charge ghost atoms

  kspmod->update_charge();
}