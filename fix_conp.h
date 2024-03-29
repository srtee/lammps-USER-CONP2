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
   Version: Mar/2021
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(conp,FixConp)

#else

#ifndef LMP_FIX_CONP_H
#define LMP_FIX_CONP_H

#include "fix.h"
#include "pair.h"
#include "kspacemodule.h"

namespace LAMMPS_NS {

class FixConp : public Fix {
 friend class FixConq;
 public:
  FixConp(class LAMMPS *, int, char **);
  virtual ~FixConp();
  bool intelflag;
  int setmask();
  void init();
  int modify_param(int, char **);
  void pre_force(int);
  void post_neighbor();
  void post_force(int);
  void force_cal(int);
  void setup_post_neighbor();
  void setup_pre_force(int);
  void linalg_init();
  void linalg_setup();
  void a_cal();
  void a_read();
  void b_setq_cal();
  virtual void b_cal();
  void update_bk(bool, double *);
  void equation_solve();
  virtual void update_charge();
  int electrode_check(int);
  void cg();
  void inv();
  void inv_project();
  void get_setq();
  void b_comm(double *, double *);
  void b_comm_int(int *, int *);
  void b_bcast(int, int, int*, double*);
  void alist_coul_cal(double *);
  void blist_coul_cal(double *);
  void blist_coul_cal_post_force();
  void request_smartlist();
  virtual double compute_scalar();
  virtual void dyn_setup() {}
  virtual void cond_setup() {}
  void init_list(int, class NeighList*);
  void end_of_step();
  double erfcr_sqrt(double);
  double ferfcr_sqrt(double);
  int elenum,elenum_all;
  int elytenum;
  int *ele2tag,*ele2eleall;
  int *tag2eleall,*eleall2tag;
  class KSpaceModule *kspmod;
  double eta;
  int *elenum_list,*displs,*eleall2ele;
  bool splitflag;
  tagint maxtag_all;


 protected:
  /* pair type polymorphism      */
  int pairmode;
  using rsq_ij_func = double(FixConp::*)(double, int, int);
  // rsq_ij_func self_potential;
  rsq_ij_func pair_potential;
  rsq_ij_func pair_force;

  double eta_potential_A(double, int, int);
  double eta_potential(double, int, int);
  double eta_force(double, int, int);
  double evscale;
  
  double kappa;
  double* eta_i,*u0_i;
  double** eta_ij,**fo_ij;
  void ehgo_setup_tables();
  bool ehgo_allocated;
  void ehgo_allocate();
  void ehgo_deallocate();
  double ehgo_potential(double, int, int);
  double ehgo_force(double, int, int);
  /* ---------------------------- */
  
  class NeighList *list;
  int ilevel_respa, maxiter;
  double g_ewald;
  int ff_flag; 
  int minimizer;
  double potdiff;
  int potdiffstyle,potdiffvar;
  char *potdiffstr,*group2;
  int jgroup,jgroupbit;
  
  int elenum_old;
  double *eleallq;
  double *elesetq;
  double *eleinitq;
  double *aaa_all,*bbb_all;
  int *elecheck_eleall;
  int *elebuf2eleall;
  double totsetq,addv;
  double *bbb,*bbuf,*newtonbuf;

  bool smartlist,newton;
  int eletypenum,arequest,brequest;
  int *eletypes;
  class NeighList *alist,*blist;

  int me,runstage,gotsetq,nprocs;
  double Btime,Btime1,Btime2;
  double Ctime,Ctime1,Ctime2;
  double Ktime,Ktime1,Ktime2;
  double cgtime,cgtime1,cgtime2;
  FILE *outf,*a_matrix_fp;
  int a_matrix_f;
  int molidL,molidR;
  double tolerance;


  int everynum;
  Pair *coulpair;

  bool zneutrflag,initflag,matoutflag,pppmflag,qinitflag;
  bool lowmemflag,nullneutralflag;
  bool preforceflag,postforceflag;
  bool one_electrode_flag;

  double scalar_output;
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
// eleall2ele: length elenum_all list holding ele    idx
// ele2tag:    length elenum     list holding tag    idx
// eleall2tag: length elenum_all list holding tag    idx
// tag2eleall: length natoms+1   list holding eleall idx
// for conversions involving i, always use tag!
// i2ele[i] = eleall2ele[tag2eleall[tag[i]]]
// ele2i[ele] = atom->map(eleall2tag[ele2eleall[ele]])
