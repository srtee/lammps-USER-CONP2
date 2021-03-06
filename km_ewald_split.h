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

#ifndef LMP_FIXCONP_KM_EWALD_SPLIT_H
#define LMP_FIXCONP_KM_EWALD_SPLIT_H

#include "kspacemodule.h"
#include "pointers.h"

namespace LAMMPS_NS{

class KSpaceModuleEwaldSplit : public KSpaceModule, public Pointers {
 public:
  KSpaceModuleEwaldSplit(class LAMMPS *);
  virtual ~KSpaceModuleEwaldSplit();
  
  void conp_setup(bool);
  void conp_post_neighbor(bool, bool);
  void a_cal(double *);
  void a_read();
  void b_cal(double *);

  protected:
  bool aread_sincos_a;
  void setup_allocate();
  void elyte_allocate(int);
  void ele_allocate(int);
  void setup_deallocate();
  void elyte_deallocate();
  void ele_deallocate();
  void sincos_a_ele(double **, double **);
  void sincos_a_comm_eleall(double **, double **);
  void aaa_from_sincos_a(double *);
  void sincos_b();
  void bbb_from_sincos_b(double *);
  
  int slabflag;

  double rms(int,double,bigint,double);
  void make_kvecs_ewald();
  void make_kvecs_brick();
  void make_ug_from_kvecs();
  void make_kxy_list_from_kvecs();
  void sincos_a_transpose(double **, double **);
  void kz_expand(double*, double*, double*, double*, double*);
  double ewald_dot_ij(double*, double*, double*, double*, double*, double*, double*, double*); 
  double ewald_dot_ii(double*, double*, double*, double*);
  double ewald_dot_ib(int, double*, double*, double*);
  double unitk[3];
  double *ug;
  double g_ewald,gsqmx,volume,slab_volfactor;
  int *kxvecs,*kyvecs,*kzvecs;
  double **cs,**sn,**csk,**snk;
  double *qj_global;
  int kmax,kmax3d,kmax_created,kcount,kcount_flat,kcount_expand;
  int *kcount_dims;
  int *kxy_list,*kz_list;
  int kxmax,kymax,kzmax;
  double *sfacrl,*sfacrl_all,*sfacim,*sfacim_all;
};
}

#endif
