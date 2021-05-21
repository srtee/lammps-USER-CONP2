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

#ifdef KSPACE_CLASS

KSpaceStyle(pppm/conp,PPPMCONP)

#else 

#ifndef LMP_PPPM_CONP_H
#define LMP_PPPM_CONP_H

#include "pppm.h"
#include "kspacemodule.h"

namespace LAMMPS_NS {

class PPPMCONP : public PPPM, public KSpaceModule {
 public:
  PPPMCONP(class LAMMPS *);
  ~PPPMCONP();
  void conp_setup() {}
  void conp_post_neighbor(bool, bool);
  void a_cal(double*);
  void a_read();
  void b_cal(double*);
  void conp_pre_force() {elyte_mapped = false;}
  void update_charge() {ele_make_rho();}
  double compute_particle_potential(int);
 protected:
  void aaa_map_rho();
  void elyte_map_rho_pois();

  void setup_allocate();
  void ele_allocate(int);
  void elyte_allocate(int);

  void setup_deallocate();
  void ele_deallocate();
  void elyte_deallocate();

  class KSpaceModuleEwald* my_ewald;
  bool first_postneighbor;
  bool first_bcal;
  bool reuseflag;
  void elyte_particle_map();
  void elyte_make_rho();
  void elyte_poisson();

  FFT_SCALAR ***ele2rho;
  FFT_SCALAR ***elyte_density_brick;
  FFT_SCALAR ***ele_density_brick;
  int jmax;
  int* j2i;
  bool elyte_mapped;

  void ele_make_rho();
  void particle_map();
  void make_rho();
};
}

#endif
#endif