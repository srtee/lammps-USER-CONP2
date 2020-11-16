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
   Version: Nov/2020
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#include "fix_conqv3.h"
#include "force.h"
#include "atom.h"
#include "input.h"
#include "variable.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CONSTANT,EQUAL,ATOM};

/* --------------------------------------------------------------------- */

void FixConqV3::update_charge()
{
  int i,j,idx1d;
  int elealli,tagi;
  double eleallq_i;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nall = nlocal+atom->nghost;
  double netcharge_left_local = 0;
  double netcharge_left = 0;
  double *q = atom->q;    
  if (minimizer == 0) {
    for (i = 0; i < nall; ++i) {
      if (electrode_check(i)) {
        tagi = tag[i];
        elealli = tag2eleall[tagi];
        q[i] = eleallq[elealli];
      }
    }
  } else if (minimizer == 1) {
    for (i = 0; i < nall; ++i) {
      if (electrode_check(i)) {
        tagi = tag[i];
        elealli = tag2eleall[tagi];
	idx1d = elealli*elenum_all;
        eleallq_i = 0.0;
        for (j = 0; j < elenum_all; j++) {
          eleallq_i += aaa_all[idx1d]*bbb_all[j];
	  idx1d++;
        }
        q[i] = eleallq_i;
      } 
    }
  }

  //  now we need to get total left charge
  for (i = 0; i < nlocal; ++i) {
    if (electrode_check(i) == 1) netcharge_left_local += q[i];
  }
  MPI_Allreduce(&netcharge_left_local,&netcharge_left,1,MPI_DOUBLE,MPI_SUM,world);

  //  calculate additional charge needed
  if (qlstyle == EQUAL) qL = input->variable->compute_equal(qlvar);
  addv = (qL - netcharge_left)/totsetq;
  //if (me == 0) fprintf(outf,"%g  %g  %g  %g\n",qL,netcharge_left,totsetq,addv);
  // TO-DO: figure out how to output addv
  for (i = 0; i < nall; ++i) {
    if (electrode_check(i)) {
      tagi = tag[i];
      elealli = tag2eleall[tagi];
      q[i] += addv*elesetq[elealli];
    }
  }
}
