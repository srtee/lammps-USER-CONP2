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
#include "fix_conp_dyn.h"
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
enum{NO_QOLD,NO_VQ,NO_AQ,DYN_READY};
extern "C" {
  void daxpy_(const int *N, const double *alpha, const double *X, const size_t *incX, double *Y, const size_t *incY);
  void dgetrf_(const int *M,const int *N,double *A,const int *lda,int *ipiv,int *info);
  void dgetri_(const int *N,double *A,const int *lda,const int *ipiv,double *work,const int *lwork,int *info);
}

/* ---------------------------------------------------------------------- */
// TODO: make sure destructor destroys these arrays!!
void FixConpDyn::dyn_setup()
{
  qold = new double[elenum_all];
  vq = new double[elenum_all];
  aq = new double[elenum_all];
  dyn_step = 0;
  dyn_interval = 1;
  dyn_fails = 0;
  dyn_status = NO_QOLD;
  //if (me == 0) printf("%d\t%d\t%d\t%d\n",dyn_status,dyn_interval,dyn_step,dyn_fails);
}

/* ---------------------------------------------------------------------- */

void FixConpDyn::post_integrate()
{
  if(update->ntimestep % everynum == 0) {
    if (strstr(update->integrate_style,"verlet")) { //not respa
      if (dyn_step % dyn_interval == 0 || dyn_fails > 10) {
        dyn_step = 0;
        Btime1 = MPI_Wtime();
        b_cal();
        Btime2 = MPI_Wtime();
        Btime += Btime2-Btime1;
        if (update->laststep == update->ntimestep) {
          double Btime_all;
          MPI_Reduce(&Btime,&Btime_all,1,MPI_DOUBLE,MPI_SUM,0,world);
          double Ctime_all;
          MPI_Reduce(&Ctime,&Ctime_all,1,MPI_DOUBLE,MPI_SUM,0,world);
          double Ktime_all;
          MPI_Reduce(&Ktime,&Ktime_all,1,MPI_DOUBLE,MPI_SUM,0,world);
          if (me == 0) {
            Btime = Btime_all/comm->nprocs;
            Ctime = Ctime_all/comm->nprocs;
            Ktime = Ktime_all/comm->nprocs;
            fprintf(outf,"B vector calculation time = %g\n",Btime);
            fprintf(outf,"Coulomb calculation time = %g\n",Ctime);
            fprintf(outf,"Kspace calculation time = %g\n",Ktime);
          }
        }
        equation_solve();
        update_charge();
        if (dyn_fails <= 10) {
          int dyn_interval_action = update_diffvecs_from_q();
          if (dyn_interval_action == DYN_MAINTAIN && dyn_interval == 1) {
            dyn_fails += 1;
            if (me == 0) fprintf(outf,"At step %d, couldn't increase or decrease check interval\n",update->ntimestep);
          }
          else if (dyn_interval_action == DYN_INCR) {
            dyn_interval += 1;
            if (me == 0) fprintf(outf,"At step %d, increasing explicit check interval to every %d steps\n",update->ntimestep,dyn_interval);
          }
          else if (dyn_interval_action == DYN_DECR) {
            dyn_interval = dyn_interval % 2 + dyn_interval / 2;
            if (me == 0) fprintf(outf,"At step %d, decreasing explicit check interval to every %d steps\n",update->ntimestep,dyn_interval);
          }
          else if (dyn_interval_action == DYN_INIT) {
            if (me == 0) fprintf(outf,"At step %d, initializing dynamic update, stage %d\n",update->ntimestep,dyn_status);
          }
        }
      }
      else update_q_from_diffvecs();
    }
    if (dyn_fails <= 10) ++dyn_step;
    // if (me == 0) printf("%d\t%d\t%d\t%d\n",dyn_status,dyn_interval,dyn_step,dyn_fails);
  }
}

/* ---------------------------------------------------------------------- */

int FixConpDyn::update_diffvecs_from_q()
{
  // now we know qR, qL, eleallq, and ainvd
  int i;
  double qi,dqi;
  if (dyn_status == DYN_READY) {
    double adenom = 1.0/static_cast<double>(dyn_interval*dyn_interval);
    double vdenom = 0.5*(3*static_cast<double>(dyn_interval)-1)*adenom;
    double qsq = 0;
    double sqerr = 0;
    double upp_tol = 2e-4;
    double low_tol = 1e-4;
    for (i = 0; i < elenum_all; ++i) {
      vq[i] += aq[i];
      qold[i] += vq[i];
      qi = eleallq[i] + (qR - qL)*ainvd[i];
      dqi = qi - qold[i];
      qsq += qi*qi;
      sqerr += dqi * dqi;
      aq[i] += dqi*adenom;
      vq[i] += dqi*vdenom;
      qold[i] = qi;
    }
    double tolcheck = sqerr/static_cast<double>(elenum_all);
    if (me == 0) fprintf(outf,"%g\t%g\t%g\n",sqerr,qsq,tolcheck);
    if (tolcheck >= upp_tol) return DYN_DECR;
    else if (tolcheck >= low_tol) return DYN_MAINTAIN;
    else return DYN_INCR;
  }
  else if (dyn_status == NO_QOLD) {
    for (i = 0; i < elenum_all; ++i) {
      qold[i] = eleallq[i] + (qR - qL)*ainvd[i];
    }
    dyn_status = NO_VQ;
    return DYN_INIT;
  }
  else if (dyn_status == NO_VQ) {
    for (i = 0; i < elenum_all; ++i) {
      qi = eleallq[i] + (qR - qL)*ainvd[i];
      vq[i] = (qi - qold[i])/static_cast<double>(dyn_interval);
      qold[i] = qi;
    }
    dyn_status = NO_AQ;
    return DYN_INIT;
  }
  else if (dyn_status == NO_AQ) {
    for (i = 0; i < elenum_all; ++i) {
      qi = eleallq[i] + (qR - qL)*ainvd[i];
      dqi = qi - qold[i];
      aq[i] = (dqi - vq[i])/static_cast<double>(dyn_interval);
      vq[i] = dqi;
      qold[i] = qi;
    }
    dyn_status = DYN_READY;
    return DYN_INIT;
  }
}

/* ---------------------------------------------------------------------- */

void FixConpDyn::update_q_from_diffvecs()
{
  int i,tagi,elealli,iall;
  int* tag = atom->tag;
  double* q = atom->q;
  int nlocal = atom->nlocal;
  int nall = nlocal+atom->nghost;
  double netcharge_left = 0;
  for (iall = 0; iall < elenum_all; ++iall) {
    vq[iall] += aq[iall];
    qold[iall] += vq[iall];
    if (elecheck_eleall[iall] == 1) netcharge_left += qold[iall];
    i = atom->map(eleall2tag[iall]);
    if (i != -1) q[i] = qold[iall];
  }
  //for (i = 0; i < nall; ++i) {
  //  if (electrode_check(i)) {
  //    tagi = tag[i];
  //    elealli = tag2eleall[tagi];
  //    q[i] = qold[elealli];
  //  }
  //}
  addv = netcharge_left;
}
