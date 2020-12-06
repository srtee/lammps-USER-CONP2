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
   Version: Nov/2020
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(conp,FixConpV3)

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
  void pre_neighbor();
  void force_cal(int);
  void a_cal();
  void a_read();
  void b_setq_cal();
  virtual void b_cal();
  void update_bk(bool, double *);
  void equation_solve();
  virtual void update_charge();
  int electrode_check(int);
  void sincos_a(double **);
  void sincos_b();
  void cg();
  void inv();
  void get_setq();
  void b_comm(double *, double *);
  void coul_cal(int, double *);
  virtual double compute_scalar();
  virtual void dyn_setup() {}

 protected:
  int ff_flag; 
  int minimizer;
  double qL,qR;
  int qlstyle,qrstyle,qlvar,qrvar;
  int elenum,elenum_old,elenum_all;
  double *eleallq;
  double *elesetq;
  double *aaa_all,*bbb_all;
  int *tag2eleall,*eleall2tag,*ele2tag;
  int *elecheck_eleall;
  int *elenum_list,*displs,*i2ele;
  int *ele2i,*elebuf2eleall,*ele2eleall;
  double totsetq,addv;
  double *bbb,*bbuf;

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

// crosslist naming conventions:
// the array 'A2B' holds values such that A2B[A] == B
// for example, 'ele2eleall[elei]' returns the eleall index
// of atom with ele index i
//
// Important lists:
// eleall: global permanent numbering of electrode atoms
// from 0 to elenum_all-1
// ele: local volatile numbering of electrode atoms
// from 0 to elenum-1
// i: local volatile numbering of all atoms
// from 0 to nlocal-1 for locals 
// and nlocal to nlocal+nghost-1 for ghosts
// tag: global permanent numbering of all atoms
// from *1* to natoms
//
// Important cross-lists:
// ele2eleall: length elenum     list holding eleall idx
// ele2i:      length elenum     list holding i      idx
// ele2tag:    length elenum     list holding tag    idx
// eleall2tag: length elenum_all list holding tag    idx
// i2ele:      length nlocal     list holding ele    idx
// tag2eleall: length natoms+1   list holding eleall idx
