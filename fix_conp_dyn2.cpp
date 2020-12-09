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

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "stddef.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "force.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "compute.h"
#include "fix_conp_dyn2.h"
#include "pair_hybrid.h"

#include "pair.h"
#include "kspace.h"
#include "comm.h"
#include "mpi.h"
#include "math_const.h"
#include "neigh_list.h"
#include "domain.h"
#include "utils.h"
#include "iostream"

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{CONSTANT,EQUAL,ATOM};
enum{CG,INV};
enum{DYN_INIT,DYN_INCR,DYN_MAINTAIN,DYN_DECR};
enum{NO_BOLD,NO_VB,NO_AB,DYN_READY};
enum{NORMAL,FFIELD,NOSLAB};
extern "C" {
  void daxpy_(const int *N, const double *alpha, const double *X, const size_t *incX, double *Y, const size_t *incY);
  void dgetrf_(const int *M,const int *N,double *A,const int *lda,int *ipiv,int *info);
  void dgetri_(const int *N,double *A,const int *lda,const int *ipiv,double *work,const int *lwork,int *info);
}

/* ---------------------------------------------------------------------- */
// TODO: make sure destructor destroys these arrays!!
void FixConpDyn2::dyn_setup()
{
  bk = new double[elenum_all];
  bkvec = new double[elenum_all*3];
  bp = new double[elenum_all];
  bpvec = new double[elenum_all*3];
  bp_step = bp_fails = bk_step = bk_fails = 0;
  bp_interval = bk_interval = 1;
  bp_status = bk_status = NO_BOLD;
  int iall, j;
  for (iall = 0; iall < 3*elenum_all; ++iall) {
    bkvec[iall] = bpvec[iall] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixConpDyn2::b_cal()
{
  //printf("%d\t%d\t%d\t%d\n",bk_step,bk_interval,bk_fails,bk_status);
  //printf("%d\t%d\t%d\t%d\n",bp_step,bp_interval,bp_fails,bp_status);
  double bk_uerr = 4e-3;
  double bk_lerr = 1e-3;
  if (bk_step % bk_interval == 0 || bk_fails > 10) {
    bool do_bp = false;
    update_bk(do_bp,bk); 
    if (bk_fails <= 10) {
      double bk_err = update_dynv(bk,bkvec,&bk_status,bk_interval);
      if (bk_err > 0.0 && bk_err <= bk_lerr) {
        ++bk_interval;
        if (me == 0) fprintf(outf,"Step %d, bk_err was %g: now checking every %d steps\n",update->ntimestep,bk_err,bk_interval);
      }
      else if (bk_err >= bk_uerr && bk_interval > 1) {
        bk_interval = bk_interval % 2 + bk_interval / 2;
        if (me == 0) fprintf(outf,"Step %d, bk_err was %g: now checking every %d steps\n",update->ntimestep,bk_err,bk_interval);
      }
      else if (bk_err >= bk_lerr && bk_interval == 1) {
        ++bk_fails;
        if (me == 0) fprintf(outf,"Step %d, bk_err was %g: have failed to change check interval %d times\n",update->ntimestep,bk_err,bk_fails);
      }
      bk_step = 0; 
    }
  }
  else update_from_dynv(bk,bkvec);
  if (bk_fails <= 10) ++bk_step;
  // after this bk[elenum_all] holds kspace

  double bp_uerr = 4e-3;
  double bp_lerr = 1e-3;
  if (bp_step % bp_interval == 0 || bp_fails > 10) {
    update_bp(); 
    if (bp_fails <= 10) {
      double bp_err = update_dynv(bp,bpvec,&bp_status,bp_interval);
      if (bp_err > 0.0 && bp_err <= bp_lerr) {
        ++bp_interval;
        if (me == 0) fprintf(outf,"Step %d, bp_err was %g: now checking every %d steps\n",update->ntimestep,bp_err,bp_interval);
      }
      else if (bp_err >= bp_uerr && bp_interval > 1) {
        bp_interval = bp_interval % 2 + bp_interval / 2;
        if (me == 0) fprintf(outf,"Step %d, bp_err was %g: now checking every %d steps\n",update->ntimestep,bp_err,bp_interval);
      }
      else if (bp_err >= bp_lerr && bp_interval == 1) {
        ++bp_fails;
        if (me == 0) fprintf(outf,"Step %d, bp_err was %g: have failed to change check interval %d times\n",update->ntimestep,bp_err,bp_fails);
      }
      bp_step = 0; 
    }
  }
  else update_from_dynv(bp,bpvec);
  if (bp_fails <= 10) ++bp_step;
  // after this bp[elenum_all] holds rspace

  int iall;
  for (iall = 0; iall < elenum_all; ++iall) {
    bbb_all[iall] = bk[iall] + bp[iall];
  }
}

/* ---------------------------------------------------------------------- */

void FixConpDyn2::update_bp() {
  int iloc;
  for (iloc = 0; iloc < elenum; ++iloc) bbb[iloc] = 0;
  if (smartlist) blist_coul_cal(bbb);
  else coul_cal(1,bbb);
  b_comm(bbb,bp);
}

/* ---------------------------------------------------------------------- */

double FixConpDyn2::update_dynv(double *v, 
        double *vec, int* vstatus, int vinterval) {
  int iall;
  if (*vstatus == DYN_READY) {
    double vid = static_cast<double>(vinterval);
    double adenom = 1.0/(vid*vid);
    double vdenom = 0.5*(3*vid-1)*adenom;
    double vecsq = 0.0;
    double err = 0.0;
    double errsq = 0.0;
    double upp_tol = 9e-4;
    double low_tol = 4e-4;
    for (iall = 0; iall < elenum_all; ++iall) {
      vec[elenum_all+iall] += vec[2*elenum_all+iall];
      vec[iall] += vec[elenum_all+iall];
      err = v[iall] - vec[iall];
      vecsq += v[iall]*v[iall];
      errsq += err*err;
      vec[2*elenum_all+iall] += err*adenom;
      vec[elenum_all+iall] += err*vdenom;
      vec[iall] = v[iall];
    }
    double tolcheck = errsq/vecsq;
    return tolcheck;
  }
  else if (*vstatus == NO_BOLD) {
    for (iall = 0; iall < elenum_all; ++iall) {
      vec[iall] = v[iall];
    }
    *vstatus = NO_VB;
    return -1.;
  }
  else if (*vstatus == NO_VB) {
    double vid = static_cast<double>(vinterval);
    for (iall = 0; iall < elenum_all; ++iall) {
      vec[elenum_all+iall] = (v[iall] - vec[iall])/vid;
      vec[iall] = v[iall];
    }
    *vstatus = NO_AB;
    return -1.;
  }
  else if (*vstatus == NO_AB) {
    double vid = static_cast<double>(vinterval);
    for (iall = 0; iall < elenum_all; ++iall) {
      vec[2*elenum_all+iall] = (v[iall] - vec[iall] - vec[elenum_all+iall])/vid;
      vec[elenum_all+iall] = (v[iall] - vec[iall])/vid;
      vec[iall] = v[iall]; 
    }
    *vstatus = DYN_READY;
    return -1.;
  }
}

/* ---------------------------------------------------------------------- */

void FixConpDyn2::update_from_dynv(double *v, double *vec) {
  int iall;
  for (iall = 0; iall < elenum_all; ++iall) {
    vec[elenum_all+iall] += vec[2*elenum_all+iall];
    vec[iall] += vec[elenum_all+iall];
    v[iall] = vec[iall];
  }
}