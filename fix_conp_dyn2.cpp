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
    update_bk(); 
    if (bk_fails <= 10) {
      double bk_err = update_dynv(bk,bkvec,&bk_status,bk_interval);
      if (bk_err > 0.0 && bk_err <= bk_lerr) ++bk_interval;
      else if (bk_err >= bk_uerr && bk_interval > 1) {
        bk_interval = bk_interval % 2 + bk_interval / 2;
      }
      else if (bk_err >= bk_lerr && bk_interval == 1) ++bk_fails;
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
      if (bp_err > 0.0 && bp_err <= bp_lerr) ++bp_interval;
      else if (bp_err >= bp_uerr && bp_interval > 1) {
        bp_interval = bp_interval % 2 + bp_interval / 2;
      }
      else if (bp_err >= bp_lerr && bp_interval == 1) ++bp_fails;
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

void FixConpDyn2::update_bk() {
  Ktime1 = MPI_Wtime();
  int i,j,k;
  int nmax = atom->nmax;
  if (atom->nlocal > nmax) {
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    kmax_created = kmax;
  }
  sincos_b();
  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  double **x = atom->x;
  double *q = atom->q;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int kx,ky,kz;
  double cypz,sypz,exprl,expim,kspacetmp;
  int elenum = 0;
  for (i = 0; i < nlocal; i++) {
    if(electrode_check(i)) elenum++;
  }
  int *ele2i = new int[elenum];
  int *ele2eleall = new int[elenum];
  // initialize bbb and create ele tag list in current time step
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      ele2i[j] = i;
      ele2eleall[j] = tag2eleall[tag[i]];
      j++;
    }
  }
  int iele,iall;
  for (iall = 0; iall < elenum_all; ++iall) {
    bk[iall] = 0.0;
  }
  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    j = 0;
    for (iele = 0; iele < elenum; ++iele) {
      i = ele2i[iele];
      iall = ele2eleall[iele];
      cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
      sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
      exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
      expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;
      bk[iall] -= 2.0*ug[k]*(exprl*sfacrl_all[k]+expim*sfacim_all[k]);
    }
  }

  //slabcorrection in current timestep -- skip if ff / noslab
  if (ff_flag == NORMAL) {
    double slabcorrtmp = 0.0;
    double slabcorrtmp_all = 0.0;
    for (i = 0; i < nlocal; i++) {
      if (electrode_check(i) == 0) {
        slabcorrtmp += 4*q[i]*MY_PI*x[i][2]/volume;
      }
    }
    MPI_Allreduce(&slabcorrtmp,&slabcorrtmp_all,1,MPI_DOUBLE,MPI_SUM,world);
    for (iele = 0; iele < elenum; ++iele) {
      i = ele2i[iele];
      iall = ele2eleall[iele];
      bk[iall] -= x[i][2]*slabcorrtmp_all;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,bk,elenum_all,MPI_DOUBLE,MPI_SUM,world);
  Ktime2 = MPI_Wtime();
  Ktime += Ktime2-Ktime1;
  delete [] ele2i;
  delete [] ele2eleall;
}

/* ---------------------------------------------------------------------- */

void FixConpDyn2::update_bp() {
  int iall;
  for (iall = 0; iall < elenum_all; ++iall) {
    bp[iall] = 0.0;
  }
  coul_cal(1,bp);
  MPI_Allreduce(MPI_IN_PLACE,bp,elenum_all,MPI_DOUBLE,MPI_SUM,world);
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