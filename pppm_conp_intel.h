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

KSpaceStyle(pppm/conp/intel,PPPMCONPIntel)

#else 

#ifndef LMP_PPPM_CONP_INTEL_H
#define LMP_PPPM_CONP_INTEL_H

#include "pppm_intel.h"
#include "kspacemodule.h"
#include "intel_buffers.h"

namespace LAMMPS_NS {

class PPPMCONPIntel : public PPPMIntel, public KSpaceModule {
 public:
  PPPMCONPIntel(class LAMMPS *);
  ~PPPMCONPIntel();
  void compute(int, int);
  void conp_setup() {}
  void conp_post_neighbor(bool, bool);
  void a_cal(double*);
  void a_read();
  void b_cal(double*);
  void conp_pre_force() {elyte_mapped = false;particles_mapped = false;}
  void update_charge();
 protected:
  class KSpaceModuleEwald * my_ewald;
  template<class flt_t, class acc_t, int use_table>
  void aaa_map_rho(IntelBuffers<flt_t,acc_t> *buffers);
  template<class flt_t, class acc_t>
  void aaa_map_rho(IntelBuffers<flt_t,acc_t> *buffers) {
    if (_use_table == 1) {
      aaa_map_rho<flt_t,acc_t,1>(buffers);
    } else {
      aaa_map_rho<flt_t,acc_t,0>(buffers);
    }
  }

  template<class flt_t, class acc_t, int use_table>
  void elyte_make_rho(IntelBuffers<flt_t,acc_t> *buffers);
  template<class flt_t, class acc_t>
  void elyte_make_rho(IntelBuffers<flt_t,acc_t> *buffers) {
    if (_use_table == 1) {
      elyte_make_rho<flt_t,acc_t,1>(buffers);
    } else {
      elyte_make_rho<flt_t,acc_t,0>(buffers);
    }
  }
  
  template<class flt_t, class acc_t>
  void ele_make_rho(IntelBuffers<flt_t,acc_t> *buffers);

  void conp_make_rho();
  void ele_make_rho();
  void conp_compute_first(int,int);
  bool first_bcal,first_postneighbor;
  bool particles_mapped;
  
  void setup_allocate();
  void ele_allocate(int);
  void elyte_allocate(int);
  void setup_deallocate();
  void ele_deallocate();
  void elyte_deallocate();

  int* j2i;
  bool elyte_mapped;
  FFT_SCALAR ***ele2rho;
  FFT_SCALAR ***ele_density_brick;
  FFT_SCALAR ***elyte_density_brick;

  int jmax;

  void elyte_map_rho_pois();
  void elyte_poisson();
  void fill_j2i();
};
}

#endif
#endif
