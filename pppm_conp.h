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

 protected:
  void aaa_make_grid_rho();
  void pppm_b();

  void setup_allocate();
  void ele_allocate(int);
  void elyte_allocate(int);

  void setup_deallocate();
  void ele_deallocate();
  void elyte_deallocate();

  class KSpaceModuleEwald* my_ewald;
  bool first_postneighbor;
  void elyte_particle_map();
  void elyte_make_rho();
  void elyte_brick2fft();
  void elyte_poisson_u();

  void pack_reverse_grid(int, void*, int, int*); 
  void unpack_reverse_grid(int, void*, int, int*); 

  int** eleall_grid;
  FFT_SCALAR ***eleall_rho;

  int jmax;
  int* j2i;
  int** elyte_grid;
  FFT_SCALAR ***elyte_density_brick;
  FFT_SCALAR ***elyte_u_brick;
  FFT_SCALAR *elyte_density_fft;

};
}

#endif
#endif