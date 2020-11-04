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
   Version Nov/2020
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(conp/v3,FixConpV3)

#else

#ifndef LMP_FIX_CONPV3_H
#define LMP_FIX_CONPV3_H

#include "fix.h"
#include "pair.h"

namespace LAMMPS_NS {

class FixConpV3 : public Fix {
 public:
  FixConpV3(class LAMMPS *, int, char **);
  ~FixConpV3();
  int setmask();
  void init();
  void setup(int);
  void pre_force(int);
  void pre_force_respa(int,int,int);
  void force_cal(int);
  void a_cal();
  void a_read();
  void b_setq_cal();
  void b_comm(int, double *);
  void b_cal();
  void equation_solve();
  virtual void update_charge();
  int electrode_check(int);
  void sincos_a(double **);
  void sincos_b();
  void cg();
  void inv();
  void get_setq();
  void coul_cal(int, double *,int *);
  virtual double compute_scalar();

 protected:
  int minimizer;
  double qL,qR;
  int qlstyle,qrstyle,qlvar,qrvar;
  int elenum,elenum_old,elenum_all;
  double *eleallq;
  double *elesetq;
  double *aaa_all,*bbb_all;
  int *tag2eleall,*eleall2tag,*curr_tag2eleall,*ele2tag;
  double totsetq,addv;

 private:
  int me,runstage,gotsetq;
  int ilevel_respa;
  double Btime,Btime1,Btime2;
  double Ctime,Ctime1,Ctime2;
  double Ktime,Ktime1,Ktime2;
  double cgtime,cgtime1,cgtime2;
  FILE *outf,*outa,*a_matrix_fp;
  int a_matrix_f;
  int molidL,molidR;
  int maxiter;
  double tolerance;

  double rms(int,double,bigint,double);
  void coeffs();

  char *qlstr,*qrstr;

  double unitk[3];
  double *ug;
  double g_ewald,eta,gsqmx,volume,slab_volfactor;
  int *kxvecs,*kyvecs,*kzvecs;
  double ***cs,***sn,**csk,**snk;
  int kmax,kmax3d,kmax_created,kcount;
  int kxmax,kymax,kzmax;
  double *sfacrl,*sfacrl_all,*sfacim,*sfacim_all;
  int everynum;
  Pair *coulpair;
};

}

#endif
#endif
