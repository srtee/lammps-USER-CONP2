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
extern "C" {
  void daxpy_(const int *N, const double *alpha, const double *X, const size_t *incX, double *Y, const size_t *incY);
  void dgetrf_(const int *M,const int *N,double *A,const int *lda,int *ipiv,int *info);
  void dgetri_(const int *N,double *A,const int *lda,const int *ipiv,double *work,const int *lwork,int *info);
}

/* ---------------------------------------------------------------------- */

void FixConpDyn2::dyn_setup()
{
  bk = new double[elenum_all];
  bkold = new double[elenum_all];
  vbk = new double[elenum_all];
  abk = new double[elenum_all];
  bp = new double[elenum_all];
  bpold = new double[elenum_all];
  vbp = new double[elenum_all];
  abp = new double[elenum_all];
  bp_step = bp_fails = bk_step = bk_fails = 0;
  bp_interval = bk_interval = 1;
  bp_status = bk_status = NO_BOLD;
  //if (me == 0) printf("%d\t%d\t%d\t%d\n",dyn_status,dyn_interval,dyn_step,dyn_fails);
}

/* ---------------------------------------------------------------------- */

void FixConpDyn2::b_cal()
{
  //fprintf(outf,"i   id    Bvec\n");
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
  double bbb[elenum];
  j = 0;
  for (j = 0; j < elenum; j++) {
    bbb[j] = 0;
  }
  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    j = 0;
    for (i = 0; i < nlocal; i++) {
      if (electrode_check(i)) {
        cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
        sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
        exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
        expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;
        bbb[j] -= 2.0*ug[k]*(exprl*sfacrl_all[k]+expim*sfacim_all[k]);
        j++;
      }
    }
  }

  //slabcorrection and create ele tag list in current timestep
  double slabcorrtmp = 0.0;
  double slabcorrtmp_all = 0.0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i) == 0) {
      slabcorrtmp += 4*q[i]*MY_PI*x[i][2]/volume;
    }
  }
  MPI_Allreduce(&slabcorrtmp,&slabcorrtmp_all,1,MPI_DOUBLE,MPI_SUM,world);
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      bbb[j] -= x[i][2]*slabcorrtmp_all;
      ele2tag[j] = tag[i];
      j++;
    }
  }
  Ktime2 = MPI_Wtime();
  Ktime += Ktime2-Ktime1;
  
  coul_cal(1,bbb,ele2tag);
  b_comm(elenum,ele2tag,bbb,bbb_all);
}