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

#ifndef LMP_FIXCONP_KSPACEMODULE_EWALD2_H
#define LMP_FIXCONP_KSPACEMODULE_EWALD2_H

#include "pointers.h"
#include "kspacemodule.h"

namespace LAMMPS_NS{

class KspaceModule_Ewald2 : public KspaceModule, Pointers {
 public:
  KspaceModule_Ewald2(class LAMMPS *, class FixConp *);
  ~KspaceModule_Ewald2();
  void setup();
  void setup_allocate();
  void elyte_allocate(int);
  void ele_allocate(int);
  void setup_deallocate();
  void elyte_deallocate();
  void ele_deallocate();
  void sincos_a(double **, double **);
  void sincos_b();
  void bbb_from_sincos_b(double *);
 protected:
  double rms(int,double,bigint,double);
  void coeffs();
  double unitk[3];
  double *ug;
  double g_ewald,gsqmx,volume,slab_volfactor;
  int *kxvecs,*kyvecs,*kzvecs;
  double **cs,**sn,**csk,**snk;
  double *qj_global;
  int kmax,kmax3d,kmax_created,kcount,kcount_flat;
  int *kcount_dims;
  int *kxy_list;
  int kxmax,kymax,kzmax;
  double *sfacrl,*sfacrl_all,*sfacim,*sfacim_all;
};
}

#endif