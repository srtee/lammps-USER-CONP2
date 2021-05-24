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

#include "pppm_conp.h"
#include "km_ewald.h"

#include "gridcomm.h"
#include "fft3d_wrap.h"
#include "remap_wrap.h"
#include "atom.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"
#include "force.h"

#define OFFSET 16384
#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

using namespace LAMMPS_NS;
using namespace MathConst;

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

PPPMCONP::PPPMCONP(LAMMPS *lmp) :
  PPPM(lmp),KSpaceModule(),
  j2i(nullptr),ele2rho(nullptr)
{
  first_postneighbor = true;
  first_bcal = true;
  u_brick = nullptr;
}

PPPMCONP::~PPPMCONP()
{
  setup_deallocate();
  elyte_deallocate();
  ele_deallocate();
}

void PPPMCONP::conp_post_neighbor(
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
    int elenum = fixconp->elenum;
    ele_allocate(elenum);
    aaa_map_rho();
  }
}

void PPPMCONP::a_cal(double * aaa)
{
  my_ewald = new KSpaceModuleEwald(lmp);
  my_ewald->register_fix(fixconp);
  my_ewald->conp_setup();
  my_ewald->conp_post_neighbor(false,true); // ele_allocate
  my_ewald->a_cal(aaa);
  delete my_ewald;
  a_read();
}

void PPPMCONP::a_read()
{
  setup_allocate();
  conp_post_neighbor(true,true);
}

void PPPMCONP::elyte_map_rho_pois()
{
  elyte_particle_map();
  elyte_make_rho();
  // communicate my ghosts to others' grid
  gc->reverse_comm_kspace(dynamic_cast<KSpace*>(this),1,sizeof(FFT_SCALAR),REVERSE_RHO,
      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  elyte_mapped = true;
  //printf("comm:  %e\n",density_brick[9][10][10]);

  
  brick2fft();
  elyte_poisson();
  gc->forward_comm_kspace(dynamic_cast<KSpace*>(this),1,sizeof(FFT_SCALAR),FORWARD_AD,
      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
}

void PPPMCONP::elyte_particle_map()
{
  int nx,ny,nz,j;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  int flag = 0;

  if (!std::isfinite(boxlo[0]) || !std::isfinite(boxlo[1]) || !std::isfinite(boxlo[2]))
    error->one(FLERR,"Non-numeric box dimensions - simulation unstable");
  j = 0;

  for (int i = 0; i < nlocal; i++) {
    if (fixconp->electrode_check(i) == 0) {
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

      nx = static_cast<int> ((x[i][0]-boxlo[0])*delxinv+shift) - OFFSET;
      ny = static_cast<int> ((x[i][1]-boxlo[1])*delyinv+shift) - OFFSET;
      nz = static_cast<int> ((x[i][2]-boxlo[2])*delzinv+shift) - OFFSET;

      part2grid[i][0] = nx;
      part2grid[i][1] = ny;
      part2grid[i][2] = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick

      if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
          ny+nlower < nylo_out || ny+nupper > nyhi_out ||
          nz+nlower < nzlo_out || nz+nupper > nzhi_out)
        flag = 1;
      
      if (q[i] != 0) {
        j2i[j] = i;
        ++j;
      }
    }

    if (flag) error->one(FLERR,"Out of range atoms - cannot compute PPPM");
  }
  jmax = j;
}

void PPPMCONP::elyte_make_rho()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density array

  memset(&(elyte_density_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));
  memset(&(density_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt


  double *q = atom->q;
  double **x = atom->x;

  for (int j = 0; j < jmax; ++j) {
    i = j2i[j];

    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0]-boxlo[0])*delxinv;
    dy = ny+shiftone - (x[i][1]-boxlo[1])*delyinv;
    dz = nz+shiftone - (x[i][2]-boxlo[2])*delzinv;

    compute_rho1d(dx,dy,dz);

    z0 = delvolinv * q[i];
    for (n = 0; n < order; n++) {
      mz = n + nlower + nz;
      y0 = z0*rho1d[2][n+nlower];
      for (m = 0; m < order; m++) {
        my = m + nlower + ny;
        x0 = y0*rho1d[1][m+nlower];
        for (l = 0; l < order; l++) {
          mx = l + nlower + nx;
          density_brick[mz][my][mx] += x0*rho1d[0][l+nlower];
        }
      }
    }
  }
  memcpy(&(elyte_density_brick[nzlo_out][nylo_out][nxlo_out]),
         &(density_brick[nzlo_out][nylo_out][nxlo_out]),
         ngrid*sizeof(FFT_SCALAR));
  //for (int nz = nzlo_out; nz <= nzhi_out; ++nz)
  //  for (int ny = nylo_out; ny <= nyhi_out; ++ny)
  //    for (int nx = nxlo_out; nx <= nxhi_out; ++nx) {
  //      density_brick[nz][ny][nx] = elyte_density_brick[nz][ny][nx];
  //}
  //printf("elyte: %e\n",elyte_density_brick[9][10][10]);
}

void PPPMCONP::elyte_poisson()
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

void PPPMCONP::b_cal(double * bbb)
{
  elyte_map_rho_pois(); // FFT rho to kspace
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

    for (n = 0; n < order; ++n) {
      mz = n + nlower + nz;
      z0 = ele2rho[iele][2][n];
      for (m = 0; m < order; ++m) {
        my = m + nlower + ny;
        y0 = z0*ele2rho[iele][1][m];
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

void PPPMCONP::aaa_map_rho()
{
  // fixconp picks up ele_allocate call
  int i,iele,ic,n,l;
  double **x = atom->x;
  int* ele2tag = fixconp->ele2tag;
  int const elenum_c = fixconp->elenum;
  int const elenum_all_c = fixconp->elenum_all;
  double const delinv[3] = {delxinv,delyinv,delzinv};

  double dxyz[3];
  for (iele = 0; iele < elenum_c; ++iele) {
    for (ic = 0; ic < 3; ++ic) {
      i = atom->map(ele2tag[iele]);
      double xlo = x[i][ic] - boxlo[ic];
      n = static_cast<int> (xlo*delinv[ic]+shift)-OFFSET;
      part2grid[i][ic] = n;
      dxyz[ic] = n+shiftone-xlo*delinv[ic];
    }
    compute_rho1d(dxyz[0],dxyz[1],dxyz[2]);
    for (ic = 0; ic < 3; ++ic) {
      for (l = 0; l < order; ++l) {
        ele2rho[iele][ic][l] = rho1d[ic][l+nlower];
      }
    }
  }
}

void PPPMCONP::setup_allocate()
{
  memory->create3d_offset(ele_density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"fixconp:ele_density_brick");
  memory->create3d_offset(elyte_density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"fixconp:elyte_density_brick");
  if (differentiation_flag == 0) { // to-do: handle peratom allocate interactions
  memory->create3d_offset(u_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"fixconp:u_brick");
  }
}

void PPPMCONP::ele_allocate(int elenum)
{
  memory->grow(ele2rho,elenum,3,order,"fixconp:eleall_rho");
}

void PPPMCONP::elyte_allocate(int elytenum)
{
  memory->grow(j2i,elytenum,"fixconp:j2i");
}

void PPPMCONP::setup_deallocate()
{
  memory->destroy3d_offset(ele_density_brick,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(elyte_density_brick,nzlo_out,nylo_out,nxlo_out);
  if (differentiation_flag == 0) memory->destroy3d_offset(u_brick,nzlo_out,nylo_out,nxlo_out);
}

void PPPMCONP::ele_deallocate()
{
  memory->destroy(ele2rho);
}

void PPPMCONP::elyte_deallocate()
{
  memory->destroy(j2i);
}

void PPPMCONP::ele_make_rho()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density array

  memset(&(ele_density_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  const int elenum_c = fixconp->elenum;
  int* ele2tag = fixconp->ele2tag;
  double *q = atom->q;
  ////printf("%d\n",part2grid[atom->map(ele2tag[0])][2]);
  for (int iele = 0; iele < elenum_c; ++iele) {
    i = atom->map(ele2tag[iele]);

    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];

    z0 = delvolinv * q[i];
    for (n = 0; n < order; n++) {
      mz = n + nlower + nz;
      y0 = z0*ele2rho[iele][2][n];
      for (m = 0; m < order; m++) {
        my = m + nlower + ny;
        x0 = y0*ele2rho[iele][1][m];
        for (l = 0; l < order; l++) {
          mx = l + nlower + nx;
          ele_density_brick[mz][my][mx] += x0*ele2rho[iele][0][l];
        }
      }
    }
  }
  //printf("ele:   %e\n",ele_density_brick[9][10][10]);
}

void PPPMCONP::particle_map(){
  if (first_bcal || fixconp == nullptr) PPPM::particle_map();
  else if (!elyte_mapped) elyte_particle_map();
  else return;
}

void PPPMCONP::make_rho(){
  if (first_bcal || fixconp == nullptr) PPPM::make_rho();
  else {
    if (!elyte_mapped) {
      elyte_make_rho();
    }
    memcpy(&(density_brick[nzlo_out][nylo_out][nxlo_out]),
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

double PPPMCONP::compute_particle_potential(int i)
{
  double **x = atom->x;
  double *q = atom->q;
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR u = 0;
  nx = part2grid[i][0];
  ny = part2grid[i][1];
  nz = part2grid[i][2];

  dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
  dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
  dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

  compute_rho1d(dx, dy, dz);

  for (n = 0; n < order; n++)
  {
    mz = n + nlower + nz;
    z0 = rho1d[2][n + nlower];
    for (m = 0; m < order; m++)
    {
      my = m + nlower + ny;
      y0 = z0 * rho1d[1][m + nlower];
      for (l = 0; l < order; l++)
      {
        mx = l + nlower + nx;
        x0 = y0 * rho1d[0][l + nlower];
        u += x0 * u_brick[mz][my][mx];
      }
    }
  }
  
  u += 2*g_ewald*q[i]/MY_PIS;
  return static_cast<double>(u);
}