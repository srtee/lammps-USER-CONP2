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

#include "pppm_conp_intel.h"
#include "km_ewald.h"
#include "km_ewald_split.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "gridcomm.h"
#include "math_const.h"
#include "memory.h"
#include "fft3d_wrap.h"

#include "omp_compat.h"

using namespace LAMMPS_NS;
using namespace MathConst;

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

PPPMCONPIntel::PPPMCONPIntel(class LAMMPS* lmp) :
  PPPMIntel(lmp),KSpaceModule(),
  j2i(nullptr),ele2rho(nullptr)
{
  first_postneighbor = true;
  first_bcal = true;
  particles_mapped = false;
  jmax = 0;
  u_brick = nullptr;
}

PPPMCONPIntel::~PPPMCONPIntel()
{
  setup_deallocate();
  ele_deallocate();
  elyte_deallocate();
}

void PPPMCONPIntel::conp_post_neighbor(
  bool do_elyte_alloc, bool do_ele_alloc
) 
{
  if (first_postneighbor) {
    first_postneighbor = false;
    return;
  }
  else {
    // need to allocate part2grid here
    // main PPPM doesn't do it until compute()

    if (atom->nmax > nmax) {
      memory->destroy(part2grid);
      nmax = atom->nmax;
      memory->create(part2grid,nmax,3,"pppm:part2grid");
    }

    if (do_elyte_alloc) {
      int elytenum = fixconp->elytenum;
      elyte_allocate(elytenum);
    }
    if (do_ele_alloc) {
      int elenum = fixconp->elenum;
      ele_allocate(elenum);
    }
    if (fix->precision() == FixIntel::PREC_MODE_MIXED) {
      particle_map<float,double>(fix->get_mixed_buffers());
      aaa_map_rho<float,double>(fix->get_mixed_buffers());
    } else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE) {
      particle_map<double,double>(fix->get_double_buffers());
      aaa_map_rho<double,double>(fix->get_double_buffers());
    } else if (fix->precision() == FixIntel::PREC_MODE_SINGLE) {
      particle_map<float,float>(fix->get_single_buffers());
      aaa_map_rho<float,float>(fix->get_single_buffers());
    }
    particles_mapped = true;
    fill_j2i();
  }
}

void PPPMCONPIntel::fill_j2i()
{
  double *q = atom->q;
  int nlocal = atom->nlocal;
  int j = 0;
  for (int i = 0; i < nlocal; i++) {
    if (fixconp->electrode_check(i) == 0 && q[i] != 0) {
      j2i[j] = i;
      ++j;
    }
  }
  jmax = j;
}

void PPPMCONPIntel::a_cal(double * aaa)
{
  if (fixconp->splitflag) my_ewald = new KSpaceModuleEwaldSplit(lmp);
  else my_ewald = new KSpaceModuleEwald(lmp);
  my_ewald->register_fix(fixconp);
  my_ewald->conp_setup();
  my_ewald->conp_post_neighbor(false,true); // ele_allocate
  my_ewald->a_cal(aaa);
  delete my_ewald;
  a_read();
}

void PPPMCONPIntel::a_read()
{
  setup_allocate();
  conp_post_neighbor(true,true);
}

void PPPMCONPIntel::b_cal(double *bbb)
{
  elyte_map_rho_pois();
  int i,iele,iall;
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR u;
  int const elenum_c = fixconp->elenum;
  int* ele2tag = fixconp->ele2tag;
  for (iele = 0; iele < elenum_c; ++iele) {
    double bbbtmp = 0;
    i = atom->map(ele2tag[iele]);
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];

    #if defined(LMP_SIMD_COMPILER)
    #pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
    #endif
    for (n = 0; n < order; ++n) {
      mz = n + nlower + nz;
      z0 = ele2rho[iele][2][n];
      #if defined(LMP_SIMD_COMPILER)
      #pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
      #endif
      for (m = 0; m < order; ++m) {
        my = m + nlower + ny;
        y0 = z0*ele2rho[iele][1][m];
        #if defined(LMP_SIMD_COMPILER)
        #pragma simd
        #endif
        for (l = 0; l < order; ++l) {
          mx = l + nlower + nx;
          x0 = y0*ele2rho[iele][0][l];
          bbbtmp -= x0*u_brick[mz][my][mx];
        }
      }
    }
    bbb[iele] = bbbtmp;
  }
  
  if (slabflag == 1) {
    double **x = atom->x;
    double *q = atom->q;
    double slabcorr = 0.0;
    #pragma ivdep
    for (int j = 0; j < jmax; ++j) {
      slabcorr += 4*q[j2i[j]]*MY_PI*x[j2i[j]][2]/volume;
    }
    MPI_Allreduce(MPI_IN_PLACE,&slabcorr,1,MPI_DOUBLE,MPI_SUM,world);
    for (iele = 0; iele < elenum_c; ++iele) {
      i = atom->map(ele2tag[iele]);
      bbb[iele] -= x[i][2]*slabcorr;
    }
  }
  first_bcal = false;
}

void PPPMCONPIntel::elyte_map_rho_pois()
{
  if (!particles_mapped) {
    if (fix->precision() == FixIntel::PREC_MODE_MIXED) {
      particle_map<float,double>(fix->get_mixed_buffers());
    } else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE) {
      particle_map<double,double>(fix->get_double_buffers());
    } else if (fix->precision() == FixIntel::PREC_MODE_SINGLE) {
      particle_map<float,float>(fix->get_single_buffers());
    }
    particles_mapped = true;
  }
  if (fix->precision() == FixIntel::PREC_MODE_MIXED) {
    elyte_make_rho<float,double>(fix->get_mixed_buffers());
  } else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE) {
    elyte_make_rho<double,double>(fix->get_double_buffers());
  } else if (fix->precision() == FixIntel::PREC_MODE_SINGLE) {
    elyte_make_rho<float,float>(fix->get_single_buffers());
  }
  elyte_mapped = true;
  // communicate my ghosts to others' grid
  gc->reverse_comm_kspace(dynamic_cast<KSpace*>(this),1,sizeof(FFT_SCALAR),REVERSE_RHO,
      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  //printf("comm:  %e\n",density_brick[9][10][10]);

  brick2fft();
  elyte_poisson();
  gc->forward_comm_kspace(dynamic_cast<KSpace*>(this),1,sizeof(FFT_SCALAR),FORWARD_AD,
      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
}

template<class flt_t, class acc_t, int use_table>
void PPPMCONPIntel::aaa_map_rho(IntelBuffers<flt_t,acc_t> *buffers)
{
  int * ele2tag = fixconp->ele2tag;
  ATOM_T * _noalias const x = buffers->get_x(0);
  const int elenum_c = fixconp->elenum;
  int nthr;
  if (_use_lrt)
    nthr = 1;
  else
    nthr = comm->nthreads;
  
  {
    const int nix = nxhi_out - nxlo_out + 1;
    const int niy = nyhi_out - nylo_out + 1;

    const flt_t lo0 = boxlo[0];
    const flt_t lo1 = boxlo[1];
    const flt_t lo2 = boxlo[2];
    const flt_t xi = delxinv;
    const flt_t yi = delyinv;
    const flt_t zi = delzinv;
    const flt_t fshift = shift;
    const flt_t fshiftone = shiftone;
    const flt_t fdelvolinv = delvolinv;

    int ifrom,ito,tid;
    IP_PRE_omp_range_id(ifrom,ito,tid,elenum_c,nthr);
    //printf("ifrom %d \t ito %d \t tid %d \t elenum %d \t nthr %d \n",ifrom,ito,tid,elenum_c,nthr);
    for (int iele = ifrom; iele < ito; ++iele) {

      int i = atom->map(ele2tag[iele]);

      int nx = part2grid[i][0];
      int ny = part2grid[i][1];
      int nz = part2grid[i][2];

      FFT_SCALAR dx = nx+fshiftone - (x[i].x-lo0)*xi;
      FFT_SCALAR dy = ny+fshiftone - (x[i].y-lo1)*yi;
      FFT_SCALAR dz = nz+fshiftone - (x[i].z-lo2)*zi;

      if (use_table) {
        dx = dx*half_rho_scale + half_rho_scale_plus;
        int idx = dx;
        dy = dy*half_rho_scale + half_rho_scale_plus;
        int idy = dy;
        dz = dz*half_rho_scale + half_rho_scale_plus;
        int idz = dz;
        #if defined(LMP_SIMD_COMPILER)
        #pragma simd
        #endif
        for (int k = 0; k < INTEL_P3M_ALIGNED_MAXORDER; ++k) {
          //printf("iele %d \t k %d \t idx %d \t idy %d \t idz %d \n",iele,k,idx,idy,idz);
          ele2rho[iele][0][k] = rho_lookup[idx][k];
          ele2rho[iele][1][k] = rho_lookup[idy][k];
          ele2rho[iele][2][k] = rho_lookup[idz][k];
        }
      } else {
        #if defined(LMP_SIMD_COMPILER)
        #pragma simd
        #endif
        for (int k = nlower; k <= nupper; ++k) {
          FFT_SCALAR r1,r2,r3;
          r1 = r2 = r3 = ZEROF;
          for (int l = order-1; l >= 0; --l) {
            r1 = rho_coeff[l][k] + r1*dx;
            r2 = rho_coeff[l][k] + r2*dy;
            r3 = rho_coeff[l][k] + r3*dz;
          }
          ele2rho[iele][0][k-nlower] = r1;
          ele2rho[iele][1][k-nlower] = r2;
          ele2rho[iele][2][k-nlower] = r3;
        }
      }
    }
  }
}

template<class flt_t, class acc_t, int use_table>
void PPPMCONPIntel::elyte_make_rho(IntelBuffers<flt_t,acc_t> *buffers)
{
  FFT_SCALAR * _noalias global_elyte_density = 
    &(elyte_density_brick[nzlo_out][nylo_out][nxlo_out]);

  ATOM_T * _noalias const x = buffers->get_x(0);
  flt_t * _noalias const q = buffers->get_q(0);
  int nthr;
  if (_use_lrt)
    nthr = 1;
  else
    nthr = comm->nthreads;
  
  int nlocal = atom->nlocal;

  #if defined(_OPENMP)
  #pragma omp parallel LMP_DEFAULT_NONE \
    shared(nthr, nlocal, global_elyte_density) if(!_use_lrt)
  #endif
  {
    const int nix = nxhi_out - nxlo_out + 1;
    const int niy = nyhi_out - nylo_out + 1;

    const flt_t lo0 = boxlo[0];
    const flt_t lo1 = boxlo[1];
    const flt_t lo2 = boxlo[2];
    const flt_t xi = delxinv;
    const flt_t yi = delyinv;
    const flt_t zi = delzinv;
    const flt_t fshift = shift;
    const flt_t fshiftone = shiftone;
    const flt_t fdelvolinv = delvolinv;

    int ifrom,ito,tid;
    IP_PRE_omp_range_id(ifrom,ito,tid,jmax,nthr);

    FFT_SCALAR * _noalias my_density = tid == 0 ?
      global_elyte_density : perthread_density[tid-1];
    memset(my_density, 0, ngrid * sizeof(FFT_SCALAR));

    for (int j = ifrom; j < ito; ++j) {

      int i = j2i[j];

      int nx = part2grid[i][0];
      int ny = part2grid[i][1];
      int nz = part2grid[i][2];

      int nysum = nlower + ny - nylo_out;
      int nxsum = nlower + nx - nxlo_out;
      int nzsum = (nlower + nz - nzlo_out)*nix*niy + nysum*nix + nxsum;
      
      FFT_SCALAR dx = nx+fshiftone - (x[i].x-lo0)*xi;
      FFT_SCALAR dy = ny+fshiftone - (x[i].y-lo1)*yi;
      FFT_SCALAR dz = nz+fshiftone - (x[i].z-lo2)*zi;

      _alignvar(flt_t rho[3][INTEL_P3M_ALIGNED_MAXORDER], 64) = {0};
      if (use_table) {
        dx = dx*half_rho_scale + half_rho_scale_plus;
        int idx = dx;
        dy = dy*half_rho_scale + half_rho_scale_plus;
        int idy = dy;
        dz = dz*half_rho_scale + half_rho_scale_plus;
        int idz = dz;
        #if defined(LMP_SIMD_COMPILER)
        #pragma simd
        #endif
        for (int k = 0; k < INTEL_P3M_ALIGNED_MAXORDER; ++k) {
          rho[0][k] = rho_lookup[idx][k];
          rho[1][k] = rho_lookup[idy][k];
          rho[2][k] = rho_lookup[idz][k];
        }
      } else {
        #if defined(LMP_SIMD_COMPILER)
        #pragma simd
        #endif
        for (int k = nlower; k <= nupper; ++k) {
          FFT_SCALAR r1,r2,r3;
          r1 = r2 = r3 = ZEROF;
          for (int l = order-1; l >= 0; --l) {
            r1 = rho_coeff[l][k] + r1*dx;
            r2 = rho_coeff[l][k] + r2*dy;
            r3 = rho_coeff[l][k] + r3*dz;
          }
          rho[0][k-nlower] = r1;
          rho[1][k-nlower] = r2;
          rho[2][k-nlower] = r3;
        }
      }

      FFT_SCALAR z0 = fdelvolinv * q[i];

      #if defined(LMP_SIMD_COMPILER)
      #pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
      #endif
      for (int n = 0; n < order; ++n) {
        int mz = n*nix*niy + nzsum;
        FFT_SCALAR y0 = z0*rho[2][n];
        #if defined(LMP_SIMD_COMPILER)
        #pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
        #endif
        for (int m = 0; m < order; ++m) {
          int mzy = m*nix + mz;
          FFT_SCALAR x0 = y0*rho[1][m];
          #if defined(LMP_SIMD_COMPILER)
          #pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
          #endif
          for (int l = 0; l < INTEL_P3M_ALIGNED_MAXORDER; ++l) {
            int mzyx = l + mzy;
            my_density[mzyx] += x0 * rho[0][l];
          }
        }
      }
    }
  }

  if (nthr > 1) {
    #if defined(_OPENMP)
    #pragma omp parallel LMP_DEFAULT_NONE \
      shared(nthr, global_elyte_density) if (!_use_lrt)
    #endif
    {
      int ifrom, ito, tid;
      IP_PRE_omp_range_id(ifrom, ito, tid, ngrid, nthr);

      #if defined(LMP_SIMD_COMPILER)
      #pragma simd
      #endif
      for (int i = ifrom; i < ito; ++i) {
        for (int j = 1; j < nthr; ++j) {
          global_elyte_density[i] += perthread_density[j-1][i];
        }
      }
    }
  }
  FFT_SCALAR * _noalias global_density = 
    &(density_brick[nzlo_out][nylo_out][nxlo_out]);
  memcpy(global_density,global_elyte_density,ngrid*sizeof(FFT_SCALAR));
}

void PPPMCONPIntel::elyte_poisson()
{
  int i,j,k,n;
  int const nfft_c = nfft;
  
  n = 0;
  for (i = 0; i < nfft_c; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);
  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)
  n = 0;
  for (i = 0; i < nfft_c; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  n = 0;
  for (i = 0; i < nfft_c; ++i) {
    work2[n] = work1[n];
    work2[n+1] = work1[n+1];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        u_brick[k][j][i] = work2[n];
        n += 2;
      }
}

void PPPMCONPIntel::update_charge()
{
  if (fix->precision() == FixIntel::PREC_MODE_MIXED) {
    ele_make_rho<float,double>(fix->get_mixed_buffers());
  } else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE) {
    ele_make_rho<double,double>(fix->get_double_buffers());
  } else if (fix->precision() == FixIntel::PREC_MODE_SINGLE) {
    ele_make_rho<float,float>(fix->get_single_buffers());
  }
}

template<class flt_t, class acc_t>
void PPPMCONPIntel::ele_make_rho(IntelBuffers<flt_t,acc_t> *buffers)
{
  FFT_SCALAR * _noalias global_ele_density = 
    &(ele_density_brick[nzlo_out][nylo_out][nxlo_out]);

  int * ele2tag = fixconp->ele2tag;
  int const elenum_c = fixconp->elenum;
  flt_t * _noalias const q = buffers->get_q(0);
  int nthr;
  if (_use_lrt)
    nthr = 1;
  else
    nthr = comm->nthreads;
  
  int nlocal = atom->nlocal;

  #if defined(_OPENMP)
  #pragma omp parallel LMP_DEFAULT_NONE \
    shared(nthr, nlocal, global_ele_density) if(!_use_lrt)
  #endif
  {
    const int nix = nxhi_out - nxlo_out + 1;
    const int niy = nyhi_out - nylo_out + 1;

    const flt_t lo0 = boxlo[0];
    const flt_t lo1 = boxlo[1];
    const flt_t lo2 = boxlo[2];
    const flt_t xi = delxinv;
    const flt_t yi = delyinv;
    const flt_t zi = delzinv;
    const flt_t fshift = shift;
    const flt_t fshiftone = shiftone;
    const flt_t fdelvolinv = delvolinv;

    int ifrom,ito,tid;
    IP_PRE_omp_range_id(ifrom,ito,tid,elenum_c,nthr);

    FFT_SCALAR * _noalias my_density = tid == 0 ?
      global_ele_density : perthread_density[tid-1];
    memset(my_density, 0, ngrid * sizeof(FFT_SCALAR));

    for (int iele = ifrom; iele < ito; ++iele) {

      int i = atom->map(ele2tag[iele]);

      int nx = part2grid[i][0];
      int ny = part2grid[i][1];
      int nz = part2grid[i][2];

      int nysum = nlower + ny - nylo_out;
      int nxsum = nlower + nx - nxlo_out;
      int nzsum = (nlower + nz - nzlo_out)*nix*niy + nysum*nix + nxsum;

      FFT_SCALAR z0 = fdelvolinv * q[i];

      #if defined(LMP_SIMD_COMPILER)
      #pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
      #endif
      for (int n = 0; n < order; ++n) {
        int mz = n*nix*niy + nzsum;
        FFT_SCALAR y0 = z0*ele2rho[iele][2][n];
        #if defined(LMP_SIMD_COMPILER)
        #pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
        #endif
        for (int m = 0; m < order; ++m) {
          int mzy = m*nix + mz;
          FFT_SCALAR x0 = y0*ele2rho[iele][1][m];
          #if defined(LMP_SIMD_COMPILER)
          #pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
          #endif
          for (int l = 0; l < INTEL_P3M_ALIGNED_MAXORDER; ++l) {
            int mzyx = l + mzy;
            my_density[mzyx] += x0 * ele2rho[iele][0][l];
          }
        }
      }
    }
  }

  if (nthr > 1) {
    #if defined(_OPENMP)
    #pragma omp parallel LMP_DEFAULT_NONE \
      shared(nthr, global_ele_density) if (!_use_lrt)
    #endif
    {
      int ifrom, ito, tid;
      IP_PRE_omp_range_id(ifrom, ito, tid, ngrid, nthr);

      #if defined(LMP_SIMD_COMPILER)
      #pragma simd
      #endif
      for (int i = ifrom; i < ito; ++i) {
        for (int j = 1; j < nthr; ++j) {
          global_ele_density[i] += perthread_density[j-1][i];
        }
      }
    }
  }
}


void PPPMCONPIntel::ele_allocate(int elenum)
{
  memory->grow(ele2rho,elenum,3,INTEL_P3M_ALIGNED_MAXORDER,"fixconp:ele2rho");
  //create3d_offset(ele2rho,0,elenum-1,0,2,0,INTEL_P3M_ALIGNED_MAXORDER-1,"fixconp:ele2rho");
}

void PPPMCONPIntel::elyte_allocate(int elytenum)
{
  memory->grow(j2i,elytenum,"fixconp:j2i");
}

void PPPMCONPIntel::ele_deallocate()
{
  memory->destroy(ele2rho);
  //memory->destroy3d_offset(ele2rho,0,0,0);
}

void PPPMCONPIntel::elyte_deallocate()
{
  memory->destroy(j2i);
}

void PPPMCONPIntel::setup_deallocate()
{
  memory->destroy3d_offset(ele_density_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(elyte_density_brick,nzlo_out,nylo_out,nxlo_out);
  if (differentiation_flag == 0) memory->destroy3d_offset(u_brick,nzlo_out,nylo_out,nxlo_out);
}


void PPPMCONPIntel::setup_allocate()
{
  create3d_offset(elyte_density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                  nxlo_out,nxhi_out,"fixconp:elyte_density_brick");
  create3d_offset(ele_density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                  nxlo_out,nxhi_out,"fixconp:ele_density_brick");
  if (differentiation_flag == 0) { // to-do: handle peratom allocate interactions
    create3d_offset(u_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                    nxlo_out,nxhi_out,"fixconp:u_brick");
  }
}

void PPPMCONPIntel::conp_make_rho()
{
  if (first_bcal || fixconp == nullptr) {
    if (fix->precision() == FixIntel::PREC_MODE_MIXED) {
      make_rho<float,double>(fix->get_mixed_buffers());
    } else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE) {
      make_rho<double,double>(fix->get_double_buffers());
    } else if (fix->precision() == FixIntel::PREC_MODE_SINGLE) {
      make_rho<float,float>(fix->get_single_buffers());
    }
  } else {
    if (!elyte_mapped) {
      if (fix->precision() == FixIntel::PREC_MODE_MIXED) {
        elyte_make_rho<float,double>(fix->get_mixed_buffers());
      } else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE) {
        elyte_make_rho<double,double>(fix->get_double_buffers());
      } else if (fix->precision() == FixIntel::PREC_MODE_SINGLE) {
        elyte_make_rho<float,float>(fix->get_single_buffers());
      }
      elyte_mapped = true;
    } memcpy(&(density_brick[nzlo_out][nylo_out][nxlo_out]),
         &(elyte_density_brick[nzlo_out][nylo_out][nxlo_out]),
         ngrid*sizeof(FFT_SCALAR));
    for (int nz = nzlo_out; nz <= nzhi_out; ++nz)
      for (int ny = nylo_out; ny <= nyhi_out; ++ny)
        for (int nx = nxlo_out; nx <= nxhi_out; ++nx) {
          // density_brick[nz][ny][nx] = elyte_density_brick[nz][ny][nx];
          density_brick[nz][ny][nx] += ele_density_brick[nz][ny][nx];
    }
  }
}

void PPPMCONPIntel::compute(int eflag, int vflag)
{
  #ifdef _LMP_INTEL_OFFLOAD
  if (_use_base) {
    PPPM::compute(eflag, vflag);
    return;
  }
  #endif
  conp_compute_first(eflag,vflag);
  compute_second(eflag,vflag);
}

void PPPMCONPIntel::conp_compute_first(int eflag, int vflag)
{
  int i,j;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  ev_init(eflag,vflag);

  if (evflag_atom && !peratom_allocate_flag) allocate_peratom();

  // if atom count has changed, update qsum and qsqsum

  if (atom->natoms != natoms_original) {
    qsum_qsq();
    natoms_original = atom->natoms;
  }

  // return if there are no charges

  if (qsqsum == 0.0) return;

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }

  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy(part2grid);
    if (differentiation_flag == 1) {
      memory->destroy(particle_ekx);
      memory->destroy(particle_eky);
      memory->destroy(particle_ekz);
    }
    nmax = atom->nmax;
    memory->create(part2grid,nmax,3,"pppm:part2grid");
    if (differentiation_flag == 1) {
      memory->create(particle_ekx, nmax, "pppmintel:pekx");
      memory->create(particle_eky, nmax, "pppmintel:peky");
      memory->create(particle_ekz, nmax, "pppmintel:pekz");
    }
  }

  // find grid points for all my particles
  // map my particle charge onto my local 3d density grid
  // optimized versions can only be used for orthogonal boxes

  if (triclinic) {
    PPPM::particle_map();
    PPPM::make_rho();
  } else if (!particles_mapped) {
    if (fix->precision() == FixIntel::PREC_MODE_MIXED) {
      particle_map<float,double>(fix->get_mixed_buffers());
    } else if (fix->precision() == FixIntel::PREC_MODE_DOUBLE) {
      particle_map<double,double>(fix->get_double_buffers());
    } else {
      particle_map<float,float>(fix->get_single_buffers());
    }
    particles_mapped = true;
  }
  conp_make_rho();
  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  gc->reverse_comm_kspace(this,1,sizeof(FFT_SCALAR),REVERSE_RHO,
			  gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  brick2fft();

  // compute potential gradient on my FFT grid and
  //   portion of e_long on this proc's FFT grid
  // return gradients (electric fields) in 3d brick decomposition
  // also performs per-atom calculations via poisson_peratom()

  if (differentiation_flag == 1) poisson_ad();
  else poisson_ik();

  // all procs communicate E-field values
  // to fill ghost cells surrounding their 3d bricks

  if (differentiation_flag == 1)
    gc->forward_comm_kspace(this,1,sizeof(FFT_SCALAR),FORWARD_AD,
			    gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  else
    gc->forward_comm_kspace(this,3,sizeof(FFT_SCALAR),FORWARD_IK,
			    gc_buf1,gc_buf2,MPI_FFT_SCALAR);

  // extra per-atom energy/virial communication

  if (evflag_atom) {
    if (differentiation_flag == 1 && vflag_atom)
      gc->forward_comm_kspace(this,6,sizeof(FFT_SCALAR),FORWARD_AD_PERATOM,
			      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
    else if (differentiation_flag == 0)
      gc->forward_comm_kspace(this,7,sizeof(FFT_SCALAR),FORWARD_IK_PERATOM,
			      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  }
  particles_mapped = false;
}

template <int use_table>
double PPPMCONPIntel::compute_particle_potential(int i)
{
  int mx,my,mz,l,m,n;
  FFT_SCALAR x0,y0,z0;
  double **x = atom->x;
  double *q = atom->q;
  FFT_SCALAR u = 0.;
  const FFT_SCALAR lo0 = boxlo[0];
  const FFT_SCALAR lo1 = boxlo[1];
  const FFT_SCALAR lo2 = boxlo[2];
  const FFT_SCALAR xi = delxinv;
  const FFT_SCALAR yi = delyinv;
  const FFT_SCALAR zi = delzinv;
  const FFT_SCALAR fshift = shift;
  const FFT_SCALAR fshiftone = shiftone;
  const FFT_SCALAR fdelvolinv = delvolinv;

  const int nix = nxhi_out - nxlo_out + 1;
  const int niy = nyhi_out - nylo_out + 1;
  
  int nx = part2grid[i][0];
  int ny = part2grid[i][1];
  int nz = part2grid[i][2];

  int nysum = nlower + ny - nylo_out;
  int nxsum = nlower + nx - nxlo_out;
  int nzsum = (nlower + nz - nzlo_out) * nix * niy + nysum * nix + nxsum;

  FFT_SCALAR dx = nx + fshiftone - (x[i][0] - lo0) * xi;
  FFT_SCALAR dy = ny + fshiftone - (x[i][1] - lo1) * yi;
  FFT_SCALAR dz = nz + fshiftone - (x[i][2] - lo2) * zi;

  _alignvar(FFT_SCALAR rho[3][INTEL_P3M_ALIGNED_MAXORDER], 64) = {0};

  if (use_table)
  {
    dx = dx * half_rho_scale + half_rho_scale_plus;
    int idx = dx;
    dy = dy * half_rho_scale + half_rho_scale_plus;
    int idy = dy;
    dz = dz * half_rho_scale + half_rho_scale_plus;
    int idz = dz;
#if defined(LMP_SIMD_COMPILER)
#pragma simd
#endif
    for (int k = 0; k < INTEL_P3M_ALIGNED_MAXORDER; ++k)
    {
      rho[0][k] = rho_lookup[idx][k];
      rho[1][k] = rho_lookup[idy][k];
      rho[2][k] = rho_lookup[idz][k];
    }
  }
  else
  {
#if defined(LMP_SIMD_COMPILER)
#pragma simd
#endif
    for (int k = nlower; k <= nupper; ++k)
    {
      FFT_SCALAR r1, r2, r3;
      r1 = r2 = r3 = ZEROF;
      for (int l = order - 1; l >= 0; --l)
      {
        r1 = rho_coeff[l][k] + r1 * dx;
        r2 = rho_coeff[l][k] + r2 * dy;
        r3 = rho_coeff[l][k] + r3 * dz;
      }
      rho[0][k - nlower] = r1;
      rho[1][k - nlower] = r2;
      rho[2][k - nlower] = r3;
    }
  }
#if defined(LMP_SIMD_COMPILER)
#pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
#endif
  for (n = 0; n < order; ++n)
  {
    mz = n + nlower + nz;
    z0 = rho[2][n];
#if defined(LMP_SIMD_COMPILER)
#pragma loop_count min(2), max(INTEL_P3M_ALIGNED_MAXORDER), avg(7)
#endif
    for (m = 0; m < order; ++m)
    {
      my = m + nlower + ny;
      y0 = z0 * rho[1][m];
#if defined(LMP_SIMD_COMPILER)
#pragma simd
#endif
      for (l = 0; l < order; ++l)
      {
        mx = l + nlower + nx;
        x0 = y0 * rho[0][l];
        u -= x0 * u_brick[mz][my][mx];
      }
    }
  }
  
  u += 2*g_ewald*q[i]/MY_PIS;
  return static_cast<double>(u);
}