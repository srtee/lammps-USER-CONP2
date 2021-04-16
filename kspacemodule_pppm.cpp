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

#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "remap_wrap.h"
#include "fft3d_wrap.h"
#include "gridcomm.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neighbor.h"
#include "force.h"
#include "kspace.h"
#include "domain.h"
#include "error.h"
#include "kspacemodule_pppm.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

#define OFFSET 16384
#define EPS_HOC 1.0e-7

enum{REVERSE_RHO_ELYTE};

KSpaceModulePPPM::KSpaceModulePPPM(class LAMMPS * lmp) :
Pointers(lmp)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  collective_flag = 0;
  stagger_flag = 0;
}

void KSpaceModulePPPM::aaa_setup()
{
  // let PPPM decide the global grid
  g_ewald = force->kspace->g_ewald;
  nx_pppm = force->kspace->nx_pppm;
  ny_pppm = force->kspace->ny_pppm;
  nz_pppm = force->kspace->nz_pppm;
  slabflag = force->kspace->slabflag;
  slab_volfactor = force->kspace->slab_volfactor;
  triclinic = domain->triclinic;
  order = force->kspace->order;
  qdist = 0.0; // placeholder for TIP4P code
  boxlo = domain->boxlo;
  double *prd;
  prd = domain->prd;
  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab; 

  delxinv = nx_pppm/xprd;
  delyinv = ny_pppm/yprd;
  delzinv = nz_pppm/zprd_slab;
  delvolinv = delxinv*delyinv*delzinv;

  set_grid_local();
  setup_allocate();
  compute_gf_ik();
}

void KSpaceModulePPPM::pppm_b()
{
  elyte_particle_map();
  elyte_make_rho();

  // communicate my ghosts to others' grid
  gc->reverse_comm_kspace(reinterpret_cast<KSpace*>(this),1,sizeof(FFT_SCALAR),REVERSE_RHO_ELYTE,
      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  
  elyte_brick2fft();
  elyte_poisson_u();
}

void KSpaceModulePPPM::elyte_particle_map()
{
  int nx,ny,nz,j;

  double **x = atom->x;
  int nlocal = atom->nlocal;

  int flag = 0;

  if (!std::isfinite(boxlo[0]) || !std::isfinite(boxlo[1]) || !std::isfinite(boxlo[2]))
    error->one(FLERR,"Non-numeric box dimensions - simulation unstable");
  j = 0;

  for (int i = 0; i < nlocal; i++) {
    if (fixconp->electrode_check(i) == 0) {
      j2i[j] = i;
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

      nx = static_cast<int> ((x[i][0]-boxlo[0])*delxinv+shift) - OFFSET;
      ny = static_cast<int> ((x[i][1]-boxlo[1])*delyinv+shift) - OFFSET;
      nz = static_cast<int> ((x[i][2]-boxlo[2])*delzinv+shift) - OFFSET;

      elyte_grid[i][0] = nx;
      elyte_grid[i][1] = ny;
      elyte_grid[i][2] = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick

      if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
          ny+nlower < nylo_out || ny+nupper > nyhi_out ||
          nz+nlower < nzlo_out || nz+nupper > nzhi_out)
        flag = 1;
    }

    if (flag) error->one(FLERR,"Out of range atoms - cannot compute PPPM");
    ++j;
  }
  jmax = j;
}

void KSpaceModulePPPM::elyte_make_rho()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density array

  memset(&(elyte_density_brick[nzlo_out][nylo_out][nxlo_out]),0,
         ngrid*sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt


  double *q = atom->q;
  double **x = atom->x;

  for (int j = 0; j < jmax; ++j) {
    i = j2i[j];

    nx = elyte_grid[j][0];
    ny = elyte_grid[j][1];
    nz = elyte_grid[j][2];
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
          elyte_density_brick[mz][my][mx] += x0*rho1d[0][l];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   remap density from 3d brick decomposition to FFT decomposition
------------------------------------------------------------------------- */

void KSpaceModulePPPM::elyte_brick2fft()
{
  int n,ix,iy,iz;

  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  for (iz = nzlo_in; iz <= nzhi_in; iz++)
    for (iy = nylo_in; iy <= nyhi_in; iy++)
      for (ix = nxlo_in; ix <= nxhi_in; ix++)
        elyte_density_fft[n++] = elyte_density_brick[iz][iy][ix];

  remap->perform(elyte_density_fft,elyte_density_fft,work1);
}

void KSpaceModulePPPM::elyte_poisson_u()
{
  int i,j,k,n;
  int const nfft_c = nfft;
  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  
  n = 0;
  for (i = 0; i < nfft_c; i++) {
    work1[n++] = elyte_density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);

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

void KSpaceModulePPPM::bbb_from_pppm_b(double * bbb)
{
  int i,iele,iall;
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR u;
  int const elenum_c = fixconp->elenum;
  int* ele2tag = fixconp->ele2tag;
  int* tag2eleall = fixconp->tag2eleall;
  double **x = atom->x;
  for (iele = 0; iele < elenum_c; ++iele) {
    double bbbtmp = 0;
    iall = tag2eleall[ele2tag[iele]];
    nx = eleall_grid[iall][0];
    ny = eleall_grid[iall][1];
    nz = eleall_grid[iall][2];

    for (n = 0; n < order; ++n) {
      mz = n + nlower + nz;
      z0 = eleall_rho[iall][2][n];
      for (m = 0; m < order; ++m) {
        my = m + nlower + ny;
        y0 = z0*eleall_rho[iall][1][m];
        for (l = 0; l < order; ++l) {
          mx = l + nlower + nx;
          x0 = y0*eleall_rho[iall][0][l];
          bbbtmp += x0*u_brick[mz][my][mx];
        }
      }
    }
    bbb[iele] = bbbtmp;
  }
}

void KSpaceModulePPPM::aaa_make_grid_rho()
{
  // fixconp picks up ele_allocate call
  int i,iele,ic,n,l;
  double **x = atom->x;
  int* ele2tag = fixconp->ele2tag;
  int const elenum_c = fixconp->elenum;
  int const elenum_all_c = fixconp->elenum_all;
  double const delinv[3] = {delxinv,delyinv,delzinv};

  int** ele2grid = new int*[3];
  double*** ele2rho = new double**[3];
  double dxyz[3];
  for (ic = 0; ic < 3; ++ic) {
    ele2grid[ic] = new int[elenum_c];
    ele2rho[ic] = new double*[order];
    for (l = 0; l < order; ++l) {
      ele2rho[ic][l] = new double[elenum_c];
    }
  }
  for (iele = 0; iele < elenum_c; ++iele) {
    for (ic = 0; ic < 3; ++ic) {
      double xlo = x[i][ic] - boxlo[ic];
      i = atom->map(ele2tag[iele]);
      n = static_cast<int> (xlo*delinv[ic]+shift)-OFFSET;
      ele2grid[ic][iele] = n;
      dxyz[ic] = n+shiftone-xlo*delinv[ic];
    }
    compute_rho1d(dxyz[0],dxyz[1],dxyz[2]);
    for (ic = 0; ic < 3; ++ic) {
      for (l = 0; l < order; ++l) {
        ele2rho[ic][l][iele] = rho1d[ic][l+nlower];
      }
    }
  }
  
  int* eleall2grid_ic = new int[elenum_all_c];
  double* eleall2rho_icl = new double[elenum_all_c];
  for (ic = 0; ic < 3; ++ic) {
    fixconp->b_comm_int(ele2grid[ic],eleall2grid_ic);
    for (iele = 0; iele < elenum_all_c; ++iele)
      eleall_grid[iele][ic] = eleall2grid_ic[iele];
    for (l = 0; l < order; ++l) {
      fixconp->b_comm(ele2rho[ic][l],eleall2rho_icl);
      for (iele = 0; iele < elenum_all_c; ++iele)
        eleall_rho[iele][ic][l] = eleall2rho_icl[iele];
    }
  }
  delete [] eleall2grid_ic;
  delete [] ele2grid;
}

void KSpaceModulePPPM::ele_allocate(int elenum_all)
{
  memory->grow(eleall_grid,elenum_all,3,"fixconp:eleall_grid");
  memory->grow(eleall_rho,elenum_all,3,order,"fixconp:eleall_rho");
}

void KSpaceModulePPPM::compute_rho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy,
                                     const FFT_SCALAR &dz)
{
  int k,l;
  FFT_SCALAR r1,r2,r3;

  for (k = (1-order)/2; k <= order/2; ++k) {
    r1 = r2 = r3 = ZEROF;

    for (l = order-1; l >= 0; --l) {
      r1 = rho_coeff[l][k] + r1*dx;
      r2 = rho_coeff[l][k] + r2*dy;
      r3 = rho_coeff[l][k] + r3*dz;
    }
    rho1d[0][k] = r1;
    rho1d[1][k] = r2;
    rho1d[2][k] = r3;
  }
}

void KSpaceModulePPPM::setup_allocate()
{
  if (!stagger_flag) memory->create(gf_b,order,"fixconp:gf_b");
  memory->create2d_offset(rho1d,3,-order/2,order/2,"fixconp:rho1d");
  memory->create2d_offset(rho_coeff,order,(1-order)/2,order/2,
                          "fixconp:rho_coeff");

  memory->create3d_offset(elyte_density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"fixconp:elyte_density_brick");
  
  memory->create(elyte_density_fft,nfft_both,"fixconp:elyte_density_fft");
  memory->create(work1,2*nfft_both,"fixconp:work1");
  memory->create(work2,2*nfft_both,"fixconp:work2");
  
  gc = new GridComm(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                    nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                    nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out);
  
  gc->setup(ngc_buf1,ngc_buf2);
  memory->create(gc_buf1,ngc_buf1,"fixconp:gc_buf1");
  memory->create(gc_buf2,ngc_buf2,"fixconp:gc_buf2");

  int tmp;

  fft1 = new FFT3d(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   0,0,&tmp,collective_flag);

  fft2 = new FFT3d(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                   0,0,&tmp,collective_flag);
  
  remap = new Remap(lmp,world,
                    nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                    nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                    1,0,0,FFT_PRECISION,collective_flag);

  memory->create3d_offset(u_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"fixconp:u_brick");
  memory->create(greensfn,nfft_both,"fixconp:greensfn");
}

void KSpaceModulePPPM::set_grid_local()
{
  // global indices of PPPM grid range from 0 to N-1
  // nlo_in,nhi_in = lower/upper limits of the 3d sub-brick of
  //   global PPPM grid that I own without ghost cells
  // for slab PPPM, assign z grid as if it were not extended
  // both non-tiled and tiled proc layouts use 0-1 fractional sumdomain info

  if (comm->layout != Comm::LAYOUT_TILED) {
    nxlo_in = static_cast<int> (comm->xsplit[comm->myloc[0]] * nx_pppm);
    nxhi_in = static_cast<int> (comm->xsplit[comm->myloc[0]+1] * nx_pppm) - 1;

    nylo_in = static_cast<int> (comm->ysplit[comm->myloc[1]] * ny_pppm);
    nyhi_in = static_cast<int> (comm->ysplit[comm->myloc[1]+1] * ny_pppm) - 1;

    nzlo_in = static_cast<int>
      (comm->zsplit[comm->myloc[2]] * nz_pppm/slab_volfactor);
    nzhi_in = static_cast<int>
      (comm->zsplit[comm->myloc[2]+1] * nz_pppm/slab_volfactor) - 1;

  } else {
    nxlo_in = static_cast<int> (comm->mysplit[0][0] * nx_pppm);
    nxhi_in = static_cast<int> (comm->mysplit[0][1] * nx_pppm) - 1;

    nylo_in = static_cast<int> (comm->mysplit[1][0] * ny_pppm);
    nyhi_in = static_cast<int> (comm->mysplit[1][1] * ny_pppm) - 1;

    nzlo_in = static_cast<int> (comm->mysplit[2][0] * nz_pppm/slab_volfactor);
    nzhi_in = static_cast<int> (comm->mysplit[2][1] * nz_pppm/slab_volfactor) - 1;
  }

  // nlower,nupper = stencil size for mapping particles to PPPM grid

  nlower = -(order-1)/2;
  nupper = order/2;

  // shift values for particle <-> grid mapping
  // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

  if (order % 2) shift = OFFSET + 0.5;
  else shift = OFFSET;
  if (order % 2) shiftone = 0.0;
  else shiftone = 0.5;

  // nlo_out,nhi_out = lower/upper limits of the 3d sub-brick of
  //   global PPPM grid that my particles can contribute charge to
  // effectively nlo_in,nhi_in + ghost cells
  // nlo,nhi = global coords of grid pt to "lower left" of smallest/largest
  //           position a particle in my box can be at
  // dist[3] = particle position bound = subbox + skin/2.0 + qdist
  //   qdist = offset due to TIP4P fictitious charge
  //   convert to triclinic if necessary
  // nlo_out,nhi_out = nlo,nhi + stencil size for particle mapping
  // for slab PPPM, assign z grid as if it were not extended

  double *prd,*sublo,*subhi;

  if (triclinic == 0) {
    prd = domain->prd;
    boxlo = domain->boxlo;
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    prd = domain->prd_lamda;
    boxlo = domain->boxlo_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;

  double dist[3] = {0.0,0.0,0.0};
  double cuthalf = 0.5*neighbor->skin + qdist;
  if (triclinic == 0) dist[0] = dist[1] = dist[2] = cuthalf;
  // else kspacebbox(cuthalf,&dist[0]); NOT DOING TRICLINIC STUFF

  int nlo,nhi;
  nlo = nhi = 0;

  nlo = static_cast<int> ((sublo[0]-dist[0]-boxlo[0]) *
                            nx_pppm/xprd + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[0]+dist[0]-boxlo[0]) *
                            nx_pppm/xprd + shift) - OFFSET;
  nxlo_out = nlo + nlower;
  nxhi_out = nhi + nupper;

  nlo = static_cast<int> ((sublo[1]-dist[1]-boxlo[1]) *
                            ny_pppm/yprd + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[1]+dist[1]-boxlo[1]) *
                            ny_pppm/yprd + shift) - OFFSET;
  nylo_out = nlo + nlower;
  nyhi_out = nhi + nupper;

  nlo = static_cast<int> ((sublo[2]-dist[2]-boxlo[2]) *
                            nz_pppm/zprd_slab + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[2]+dist[2]-boxlo[2]) *
                            nz_pppm/zprd_slab + shift) - OFFSET;
  nzlo_out = nlo + nlower;
  nzhi_out = nhi + nupper;

  if (stagger_flag) {
    nxhi_out++;
    nyhi_out++;
    nzhi_out++;
  }

  // for slab PPPM, change the grid boundary for processors at +z end
  //   to include the empty volume between periodically repeating slabs
  // for slab PPPM, want charge data communicated from -z proc to +z proc,
  //   but not vice versa, also want field data communicated from +z proc to
  //   -z proc, but not vice versa
  // this is accomplished by nzhi_in = nzhi_out on +z end (no ghost cells)
  // also insure no other procs use ghost cells beyond +z limit
  // differnet logic for non-tiled vs tiled decomposition

  if (slabflag == 1) {
    if (comm->layout != Comm::LAYOUT_TILED) {
      if (comm->myloc[2] == comm->procgrid[2]-1) nzhi_in = nzhi_out = nz_pppm - 1;
    } else {
      if (comm->mysplit[2][1] == 1.0) nzhi_in = nzhi_out = nz_pppm - 1;
    }
    nzhi_out = MIN(nzhi_out,nz_pppm-1);
  }

  // x-pencil decomposition of FFT mesh
  // global indices range from 0 to N-1
  // each proc owns entire x-dimension, clumps of columns in y,z dimensions
  // npey_fft,npez_fft = # of procs in y,z dims
  // if nprocs is small enough, proc can own 1 or more entire xy planes,
  //   else proc owns 2d sub-blocks of yz plane
  // me_y,me_z = which proc (0-npe_fft-1) I am in y,z dimensions
  // nlo_fft,nhi_fft = lower/upper limit of the section
  //   of the global FFT mesh that I own in x-pencil decomposition

  int npey_fft,npez_fft;
  if (nz_pppm >= nprocs) {
    npey_fft = 1;
    npez_fft = nprocs;
  } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

  int me_y = me % npey_fft;
  int me_z = me / npey_fft;

  nxlo_fft = 0;
  nxhi_fft = nx_pppm - 1;
  nylo_fft = me_y*ny_pppm/npey_fft;
  nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
  nzlo_fft = me_z*nz_pppm/npez_fft;
  nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;

  // ngrid = count of PPPM grid pts owned by this proc, including ghosts

  ngrid = (nxhi_out-nxlo_out+1) * (nyhi_out-nylo_out+1) *
    (nzhi_out-nzlo_out+1);

  // count of FFT grids pts owned by this proc, without ghosts
  // nfft = FFT points in x-pencil FFT decomposition on this proc
  // nfft_brick = FFT points in 3d brick-decomposition on this proc
  // nfft_both = greater of 2 values

  nfft = (nxhi_fft-nxlo_fft+1) * (nyhi_fft-nylo_fft+1) *
    (nzhi_fft-nzlo_fft+1);
  int nfft_brick = (nxhi_in-nxlo_in+1) * (nyhi_in-nylo_in+1) *
    (nzhi_in-nzlo_in+1);
  nfft_both = MAX(nfft,nfft_brick);
}


void KSpaceModulePPPM::elyte_allocate(int elytenum)
{
  memory->grow(j2i,elytenum,"fixconp:j2i");
  memory->grow(elyte_grid,elytenum,3,"fixconp:elyte_grid");
}

/* ----------------------------------------------------------------------
   pack ghost values into buf to send to another proc
------------------------------------------------------------------------- */

void KSpaceModulePPPM::pack_reverse_grid(int flag, void *vbuf, int nlist, int *list)
{
  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;

  if (flag == REVERSE_RHO_ELYTE) {
    FFT_SCALAR *src = &elyte_density_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's ghost values from buf and add to own values
------------------------------------------------------------------------- */

void KSpaceModulePPPM::unpack_reverse_grid(int flag, void *vbuf, int nlist, int *list)
{
  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;

  if (flag == REVERSE_RHO_ELYTE) {
    FFT_SCALAR *dest = &elyte_density_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] += buf[i];
  }
}

void KSpaceModulePPPM::compute_gf_ik()
{
  const double * const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double sum1,dot1,dot2;
  double numerator,denominator;
  double sqk;

  int k,l,m,n,nx,ny,nz,kper,lper,mper;

  const int nbx = static_cast<int> ((g_ewald*xprd/(MY_PI*nx_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nby = static_cast<int> ((g_ewald*yprd/(MY_PI*ny_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nbz = static_cast<int> ((g_ewald*zprd_slab/(MY_PI*nz_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int twoorder = 2*order;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    snz = square(sin(0.5*unitkz*mper*zprd_slab/nz_pppm));

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      sny = square(sin(0.5*unitky*lper*yprd/ny_pppm));

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        snx = square(sin(0.5*unitkx*kper*xprd/nx_pppm));

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

        if (sqk != 0.0) {
          numerator = 12.5663706/sqk;
          denominator = gf_denom(snx,sny,snz);
          sum1 = 0.0;

          for (nx = -nbx; nx <= nbx; nx++) {
            qx = unitkx*(kper+nx_pppm*nx);
            sx = exp(-0.25*square(qx/g_ewald));
            argx = 0.5*qx*xprd/nx_pppm;
            wx = powsinxx(argx,twoorder);

            for (ny = -nby; ny <= nby; ny++) {
              qy = unitky*(lper+ny_pppm*ny);
              sy = exp(-0.25*square(qy/g_ewald));
              argy = 0.5*qy*yprd/ny_pppm;
              wy = powsinxx(argy,twoorder);

              for (nz = -nbz; nz <= nbz; nz++) {
                qz = unitkz*(mper+nz_pppm*nz);
                sz = exp(-0.25*square(qz/g_ewald));
                argz = 0.5*qz*zprd_slab/nz_pppm;
                wz = powsinxx(argz,twoorder);

                dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                dot2 = qx*qx+qy*qy+qz*qz;
                sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
              }
            }
          }
          greensfn[n++] = numerator*sum1/denominator;
        } else greensfn[n++] = 0.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   map nprocs to NX by NY grid as PX by PY procs - return optimal px,py
------------------------------------------------------------------------- */

void KSpaceModulePPPM::procs2grid2d(int nprocs, int nx, int ny, int *px, int *py)
{
  // loop thru all possible factorizations of nprocs
  // surf = surface area of largest proc sub-domain
  // innermost if test minimizes surface area and surface/volume ratio

  int bestsurf = 2 * (nx + ny);
  int bestboxx = 0;
  int bestboxy = 0;

  int boxx,boxy,surf,ipx,ipy;

  ipx = 1;
  while (ipx <= nprocs) {
    if (nprocs % ipx == 0) {
      ipy = nprocs/ipx;
      boxx = nx/ipx;
      if (nx % ipx) boxx++;
      boxy = ny/ipy;
      if (ny % ipy) boxy++;
      surf = boxx + boxy;
      if (surf < bestsurf ||
          (surf == bestsurf && boxx*boxy > bestboxx*bestboxy)) {
        bestsurf = surf;
        bestboxx = boxx;
        bestboxy = boxy;
        *px = ipx;
        *py = ipy;
      }
    }
    ipx++;
  }
}

/* ----------------------------------------------------------------------
   pre-compute Green's function denominator expansion coeffs, Gamma(2n)
------------------------------------------------------------------------- */

void KSpaceModulePPPM::compute_gf_denom()
{
  int k,l,m;

  for (l = 1; l < order; l++) gf_b[l] = 0.0;
  gf_b[0] = 1.0;

  for (m = 1; m < order; m++) {
    for (l = m; l > 0; l--)
      gf_b[l] = 4.0 * (gf_b[l]*(l-m)*(l-m-0.5)-gf_b[l-1]*(l-m-1)*(l-m-1));
    gf_b[0] = 4.0 * (gf_b[0]*(l-m)*(l-m-0.5));
  }

  bigint ifact = 1;
  for (k = 1; k < 2*order; k++) ifact *= k;
  double gaminv = 1.0/ifact;
  for (l = 0; l < order; l++) gf_b[l] *= gaminv;
}

