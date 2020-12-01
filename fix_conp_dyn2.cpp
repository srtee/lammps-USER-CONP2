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

void FixConpDyn2::setup(int vflag)
{
  //Pair *coulpair;

  // assign coulpair to either the existing pair style if it matches 'coul'
  // or, if hybrid, the pair style matching 'coul'
  // and if neither are true then something has gone horribly wrong!
  coulpair = NULL;
  coulpair = (Pair *) force->pair_match("coul",0);
  if (coulpair == NULL) {
    // return 1st hybrid substyle matching coul (inexactly)
    coulpair = (Pair *) force->pair_match("coul",0,1);
    }
  if (coulpair == NULL) error->all(FLERR,"Must use conp with coul pair style");
  //PairHybrid *pairhybrid = dynamic_cast<PairHybrid*>(force->pair);
  //Pair *coulpair = pairhybrid->styles[0];



  g_ewald = force->kspace->g_ewald;
  slab_volfactor = force->kspace->slab_volfactor;
  double accuracy = force->kspace->accuracy;
  
  int i;
  double qsqsum = 0.0;
  for (i = 0; i < atom->nlocal; i++) {
    qsqsum += atom->q[i]*atom->q[i];
  }
  double tmp,q2;
  MPI_Allreduce(&qsqsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsqsum = tmp;
  q2 = qsqsum * force->qqrd2e / force->dielectric;

// Copied from ewald.cpp
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd_slab;

  bigint natoms = atom->natoms;
  double err;
  kxmax = 1;
  kymax = 1;
  kzmax = 1;

  err = rms(kxmax,xprd,natoms,q2);
  while (err > accuracy) {
    kxmax++;
    err = rms(kxmax,xprd,natoms,q2);
  }

  err = rms(kymax,yprd,natoms,q2);
  while (err > accuracy) {
    kymax++;
    err = rms(kymax,yprd,natoms,q2);
  }

  err = rms(kzmax,zprd_slab,natoms,q2);
  while (err > accuracy) {
    kzmax++;
    err = rms(kzmax,zprd_slab,natoms,q2);
  }

  kmax = MAX(kxmax,kymax);
  kmax = MAX(kmax,kzmax);
  kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];
  ug = new double[kmax3d];

  double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
  double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
  double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
  gsqmx = MAX(gsqxmx,gsqymx);
  gsqmx = MAX(gsqmx,gsqzmx);

  gsqmx *= 1.00001;

  coeffs();
  kmax_created = kmax;

//copied from ewald.cpp end

  int nmax = atom->nmax;
  //double evscale = 0.069447;
  //vL *= evscale;
  //vR *= evscale;
  
  memory->create3d_offset(cs,-kmax,kmax,3,nmax,"fixconpv3:cs");
  memory->create3d_offset(sn,-kmax,kmax,3,nmax,"fixconpv3:sn");
  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];
  sfacrl_all = new double[kmax3d];
  sfacim_all = new double[kmax3d];
  tag2eleall = new int[natoms+1];
  curr_tag2eleall = new int[natoms+1];
  if (runstage == 0) {
    int i;
    int nlocal = atom->nlocal;
    for ( i = 0; i < nlocal; i++) {
      if (electrode_check(i)) ++elenum;
    }
    MPI_Allreduce(&elenum,&elenum_all,1,MPI_INT,MPI_SUM,world);
    
    eleall2tag = new int[elenum_all];
    elecheck_eleall = new int[elenum_all];
    // elecheck_eleall[tag2eleall[tag[i]]] = electrode_check(i)
    for (i = 0; i < elenum_all; i++) elecheck_eleall[i] = 0;
    aaa_all = new double[elenum_all*elenum_all];
    bbb_all = new double[elenum_all];
    ele2tag = new int[elenum];
    for (i = 0; i < natoms+1; i++) tag2eleall[i] = -1;
    for (i = 0; i < natoms+1; i++) curr_tag2eleall[i] = -1;
    eleallq = new double[elenum_all];
    if (a_matrix_f == 0) {
      if (me == 0) outa = fopen("amatrix","w");
      a_cal();
    } else {
      a_read();
    }
    runstage = 1;
    
    int gotsetq = 0;
    double totsetq = 0;
    b_setq_cal();
    equation_solve();
    double addv = 0;
    elesetq = new double[elenum_all]; 
    get_setq();
    gotsetq = 1;
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