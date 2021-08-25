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

#include "fix_cond.h"
#include "atom.h"
#include "input.h"
#include "variable.h"
#include "force.h"
#include "domain.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

enum{CONSTANT,EQUAL,ATOM};
enum{CG,INV};
enum{NORMAL,FFIELD,NOSLAB};
extern "C" {
  double ddot_(const int *N, const double *SX, const int *INCX, const double *SY, const int *INCY);
}

FixCond::FixCond(LAMMPS *lmp, int narg, char **arg):
  FixConp(lmp, narg, arg)
{
  rightchargevar = potdiffvar;
  rightchargestyle = potdiffstyle;
  rightcharge = potdiff;
  setup2_done = false;
}

void FixCond::cond_setup()
{
  setzvec = new double[elenum_all];
  memcpy(setzvec,bbb_all,elenum_all*sizeof(double));
  for (int i = 0; i < elenum_all; ++i) {
    setzvec[i] /= evscale;
  }
  // now setzvec holds z-hat vector
  // = -z_i / Lz
}

void FixCond::cond_setup2()
{
  double zOAz = 0.;
  double lz = domain->zprd;
  double Axy = domain->xprd*domain->yprd;
  for (int i=0; i < elenum_all; ++i) zOAz += elesetq[i]*setzvec[i];
  vmult = 4*MY_PI*zOAz*lz/(evscale*Axy);
  vmult /= 1+vmult;
  vmult /= zOAz;

  setup2_done = true;
}

void FixCond::update_charge()
{
  if (!setup2_done) cond_setup2();

  const double lz = domain->zprd;
  int i,j,idx1d,iall,jall,iloc;
  int elealli,tagi;
  double eleallq_i;
  int *tag = atom->tag;
  int const nlocal = atom->nlocal;
  int const nall = nlocal+atom->nghost;
  double netcharge_right = 0;
  double *q = atom->q;
  double **x = atom->x;
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
  //  now qL and qR are left and right *charges*
  //  evscale was included in the precalculation of eleallq

  double dipole = 0.;
  double dipole_all;
  for (int i = 0; i < nlocal; ++i) {
    if(!electrode_check(i)) dipole -= q[i]*x[i][2];
  }
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  potdiff = rightcharge - dipole_all/lz;

  for (iall = 0; iall < elenum_all_c; ++iall) {
    potdiff -= setzvec[iall]*eleallq[iall];
  }

  potdiff *= vmult;
  scalar_output = potdiff; 
  
  for (iall = 0; iall < elenum_all_c; ++iall) {
    i = atom->map(eleall2tag[iall]);
    if (i != -1) {
      q[i] = eleallq[iall] + potdiff*elesetq[iall];
      if (qinitflag) q[i] += eleinitq[iall];
    }
  } // we need to loop like this to correctly charge ghost atoms

  kspmod->update_charge();
}
