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

#ifndef LMP_FIXCONP_KSPACEMODULE_PPPM_H
#define LMP_FIXCONP_KSPACEMODULE_PPPM_H

#include "fix_conp.h"

#ifdef FFT_SINGLE
typedef float FFT_SCALAR;
#define LMP_FFT_PREC "single"
#define MPI_FFT_SCALAR MPI_FLOAT
#else

typedef double FFT_SCALAR;
#define LMP_FFT_PREC "double"
#define MPI_FFT_SCALAR MPI_DOUBLE
#endif

namespace LAMMPS_NS{

class KSpaceModulePPPM : public Pointers {
 public:
  KSpaceModulePPPM(class LAMMPS *);
  ~KSpaceModulePPPM();
  void register_fix(class FixConp * infix) {fixconp = infix;}
  void aaa_setup();
  void aaa_make_grid_rho();
  void pppm_b();
  void bbb_from_pppm_b(double *);

  void setup_allocate();
  void ele_allocate(int);
  void elyte_allocate(int);

  void setup_deallocate();
  void ele_deallocate();
  void elyte_deallocate();
  

 protected:
  class FixConp* fixconp;
  
  void set_grid_global();
  void set_grid_local();
  void elyte_particle_map();
  void elyte_make_rho();
  void elyte_brick2fft();
  void elyte_poisson_u();
  void compute_gf_ik();
  void compute_gf_denom();
  void compute_rho1d(const FFT_SCALAR &, const FFT_SCALAR &,
		     const FFT_SCALAR &);

  void procs2grid2d(int, int, int, int*, int*);

  void pack_reverse_grid(int, void*, int, int*); 
  void unpack_reverse_grid(int, void*, int, int*); 

  int me,nprocs;
  int nx_pppm,ny_pppm,nz_pppm;
  int slabflag,triclinic;
  double slab_volfactor,qdist;
  double g_ewald;
  double *boxlo;
  double *gf_b;

  int nxlo_fft,nylo_fft,nzlo_fft;
  int nxhi_fft,nyhi_fft,nzhi_fft;
  int nxlo_in,nylo_in,nzlo_in;
  int nxhi_in,nyhi_in,nzhi_in;
  int nxlo_out,nylo_out,nzlo_out;
  int nxhi_out,nyhi_out,nzhi_out;
  int ngrid,nfft,nfft_both;

  class FFT3d *fft1,*fft2;
  class Remap *remap;
  class GridComm *gc;

  FFT_SCALAR *gc_buf1,*gc_buf2;
  int order;
  int nlower,nupper;
  int ngc_buf1,ngc_buf2;

  double delxinv,delyinv,delzinv,delvolinv;
  double volume;
  double shift,shiftone;
  int** eleall_grid;
  FFT_SCALAR **rho_coeff, **rho1d;
  FFT_SCALAR ***eleall_rho;

  int collective_flag,stagger_flag;
  int jmax;
  int* j2i;
  int** elyte_grid;
  FFT_SCALAR ***elyte_density_brick;
  FFT_SCALAR ***u_brick;
  FFT_SCALAR *elyte_density_fft;
  FFT_SCALAR *work1, *work2;
  double *greensfn;

/* ----------------------------------------------------------------------
   denominator for Hockney-Eastwood Green's function
     of x,y,z = sin(kx*deltax/2), etc

            inf                 n-1
   S(n,k) = Sum  W(k+pi*j)**2 = Sum b(l)*(z*z)**l
           j=-inf               l=0

          = -(z*z)**n /(2n-1)! * (d/dx)**(2n-1) cot(x)  at z = sin(x)
   gf_b = denominator expansion coeffs
------------------------------------------------------------------------- */

  inline double gf_denom(const double &x, const double &y,
                         const double &z) const {
    double sx,sy,sz;
    sz = sy = sx = 0.0;
    for (int l = order-1; l >= 0; l--) {
      sx = gf_b[l] + sx*x;
      sy = gf_b[l] + sy*y;
      sz = gf_b[l] + sz*z;
    }
    double s = sx*sy*sz;
    return s*s;
  };

};
}

#endif
