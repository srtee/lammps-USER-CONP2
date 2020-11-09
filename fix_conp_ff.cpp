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
   Version: Sep/22/2014
   Zhenxing Wang(KU)
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
#include "fix_conp_ff.h"
#include "pair_hybrid.h"

#include "pair.h"
#include "kspace.h"
#include "comm.h"
#include "mpi.h"
#include "math_const.h"
#include "neigh_list.h"
#include "domain.h"
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
extern "C" {
  void daxpy_(const int *N, const double *alpha, const double *X, const size_t *incX, double *Y, const size_t *incY);
  void dgetrf_(const int *M,const int *N,double *A,const int *lda,int *ipiv,int *info);
  void dgetri_(const int *N,double *A,const int *lda,const int *ipiv,double *work,const int *lwork,int *info);
}

/* ---------------------------------------------------------------------- */
void FixConpFF::b_setq_cal()
{
  int i,j;
  int *tag = atom->tag;
  int **x = atom->x;
  int nlocal = atom->nlocal;
  double evscale = force->qe2f/force->qqr2e;
  double zprd = domain->zprd;
  //fprintf(outf,"%g \n",evscale);
  double bbb[elenum]; // we know elenum because things haven't changed since a_cal/read
  j = 0;
  double zfield = evscale / zprd;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      bbb[j] = -(x[i][2]+0.5*electrode_check(i)*zprd)*efield;
      elecheck_eleall[tag2eleall[tag[i]]] = electrode_check(i);
      j++;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,elecheck_eleall,elenum_all,MPI_INT,MPI_SUM,world);
  b_comm(elenum, ele2tag, bbb);
  if (runstage == 1) runstage = 2;
}
/* ---------------------------------------------------------------------- */


void FixConpFF::b_cal()
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
        ele2tag[j] = tag[i];
        j++;
      }
    }
  }

  Ktime2 = MPI_Wtime();
  Ktime += Ktime2-Ktime1;
  
  coul_cal(1,bbb,ele2tag);
  b_comm(elenum,ele2tag,bbb);
}

/*----------------------------------------------------------------------- */
void FixConpFF::a_cal()
{
  double t1,t2;
  t1 = MPI_Wtime();
  Ktime1 = MPI_Wtime();
  if (me == 0) {
    fprintf(outf,"A matrix calculating ...\n");
  }

  double **eleallx = NULL;
  memory->create(eleallx,elenum_all,3,"fixconpv3:eleallx");

  int nprocs = comm->nprocs;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  int i,j,k;
  int elenum_list[nprocs];
  MPI_Allgather(&elenum,1,MPI_INT,elenum_list,1,MPI_INT,world);
  int displs[nprocs];
  displs[0] = 0;
  int displssum = 0;
  for (i = 1; i < nprocs; ++i) {
    displssum += elenum_list[i-1];
    displs[i] = displssum;
  }
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      ele2tag[j] = tag[i];
      j++;
    }
  }

  //gather tag,x and q
  double **x = atom->x;
  
  double *elexyzlist = new double[3*elenum];
  double *elexyzlist_all = new double[3*elenum_all];
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      elexyzlist[j] = x[i][0];
      j++;
      elexyzlist[j] = x[i][1];
      j++;
      elexyzlist[j] = x[i][2];
      j++;
    }
  }
  MPI_Allgatherv(ele2tag,elenum,MPI_INT,eleall2tag,elenum_list,displs,MPI_INT,world);
  int displs2[nprocs];
  int elenum_list2[nprocs];
  for (i = 0; i < nprocs; i++) {
    elenum_list2[i] = elenum_list[i]*3;
    displs2[i] = displs[i]*3;
  }
  MPI_Allgatherv(elexyzlist,elenum*3,MPI_DOUBLE,elexyzlist_all,elenum_list2,displs2,MPI_DOUBLE,world);

  double *aaa = new double[elenum*elenum_all];
  for (i = 0; i < elenum*elenum_all; i++) {
    aaa[i] = 0.0;
  }
  j = 0;
  for (i = 0; i < elenum_all; i++) {
    if (i == 0 && me == 0) fprintf(outa," ");
    if (me == 0) fprintf (outa,"%20d",eleall2tag[i]);
    tag2eleall[eleall2tag[i]] = i;
    eleallx[i][0] = elexyzlist_all[j];
    j++;
    eleallx[i][1] = elexyzlist_all[j];
    j++;
    eleallx[i][2] = elexyzlist_all[j];
    j++;
  }
  if (me == 0) fprintf (outa,"\n");


  memory->create(csk,kcount,elenum_all,"fixconpv3:csk");
  memory->create(snk,kcount,elenum_all,"fixconpv3:snk");
  sincos_a(eleallx);
  delete [] elexyzlist;
  delete [] elexyzlist_all;

  int elealli,elei,idx1d;
  double CON_s2overPIS = sqrt(2.0)/MY_PIS;
  double CON_2overPIS = 2.0/MY_PIS;
  for (i = 0; i < nlocal; ++i) {
    if (electrode_check(i)) {
      elealli = tag2eleall[tag[i]];
      for (k = 0; k < elenum; ++k) {
        if (ele2tag[k] == tag[i]) {
          elei = k;
          break;
        }
      }
      for (j = 0; j < elenum_all; ++j) {
        idx1d = elei*elenum_all+j;
        for (k = 0; k < kcount; ++k) {
          aaa[idx1d] += 2.0*ug[k]*(csk[k][elealli]*csk[k][j]+snk[k][elealli]*snk[k][j]);
        }
      }
      idx1d = elei*elenum_all+elealli;
      aaa[idx1d] += CON_s2overPIS*eta-CON_2overPIS*g_ewald; //gaussian self correction
    }
  }
  
  memory->destroy(eleallx);
  memory->destroy(csk);
  memory->destroy(snk);

  coul_cal(2,aaa,ele2tag);
  
  int elenum_list3[nprocs];
  int displs3[nprocs];
  for (i = 0; i < nprocs; i++) {
    elenum_list3[i] = elenum_list[i]*elenum_all;
    displs3[i] = displs[i]*elenum_all;
  }
  MPI_Allgatherv(aaa,elenum*elenum_all,MPI_DOUBLE,aaa_all,elenum_list3,displs3,MPI_DOUBLE,world);
  delete [] aaa;
  aaa = NULL;
  for (i = 0; i < elenum_all; ++i) {
    for (j = 0; j < elenum_all; ++j) {
      idx1d = i*elenum_all+j;
      if (j != 0 && me == 0) fprintf(outa," ");
      if (me == 0) fprintf (outa,"%20.12f",aaa_all[idx1d]);
    }
    if (me == 0) fprintf (outa,"\n");
  }
  if(me == 0) fclose(outa);

  t2 = MPI_Wtime();
  double tsum = t2 - t1;
  double tsum_all;
  MPI_Allreduce(&tsum,&tsum_all,1,MPI_DOUBLE,MPI_SUM,world);
  if (me == 0) {
    tsum = tsum_all/nprocs;
    fprintf(outf,"A matrix calculation time  = %g\n",tsum);
  }
  Ktime2 = MPI_Wtime();
  Ktime += Ktime2-Ktime1;
}


