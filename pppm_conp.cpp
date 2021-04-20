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

enum{REVERSE_RHO,REVERSE_RHO_ELYTE};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM,FORWARD_ELYTE};

PPPMCONP::PPPMCONP(LAMMPS *lmp) :
  PPPM(lmp),KSpaceModule(),
  elyte_density_brick(nullptr),elyte_density_fft(nullptr),
  elyte_u_brick(nullptr),j2i(nullptr),
  elyte_grid(nullptr),eleall_grid(nullptr),eleall_rho(nullptr)
{
  first_postneighbor = true;
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
    if (do_elyte_alloc) {
      int elytenum = fixconp->elytenum;
      elyte_allocate(elytenum);
    }
    if (do_ele_alloc) {
      int elenum_all = fixconp->elenum_all;
      ele_allocate(elenum_all);
    }
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
  aaa_make_grid_rho();
}


void PPPMCONP::elyte_map_fft1()
{
  elyte_particle_map();
  elyte_make_rho();

  // communicate my ghosts to others' grid
  gc->reverse_comm_kspace(dynamic_cast<KSpace*>(this),1,sizeof(FFT_SCALAR),REVERSE_RHO,
      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  
  brick2fft();
  elyte_poisson1();
  elyte_fft1_done = true;
}

void PPPMCONP::elyte_fft2_u()
{
  elyte_poisson2();
  gc->forward_comm_kspace(dynamic_cast<KSpace*>(this),1,sizeof(FFT_SCALAR),FORWARD_AD,
      gc_buf1,gc_buf2,MPI_FFT_SCALAR);
  elyte_fft2_done = true;
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
    if (fixconp->electrode_check(i) == 0 && q[i] != 0) {
      j2i[j] = i;
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

      nx = static_cast<int> ((x[i][0]-boxlo[0])*delxinv+shift) - OFFSET;
      ny = static_cast<int> ((x[i][1]-boxlo[1])*delyinv+shift) - OFFSET;
      nz = static_cast<int> ((x[i][2]-boxlo[2])*delzinv+shift) - OFFSET;

      elyte_grid[j][0] = nx;
      elyte_grid[j][1] = ny;
      elyte_grid[j][2] = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick

      if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
          ny+nlower < nylo_out || ny+nupper > nyhi_out ||
          nz+nlower < nzlo_out || nz+nupper > nzhi_out)
        flag = 1;
      ++j;
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
          density_brick[mz][my][mx] += x0*rho1d[0][l+nlower];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   remap density from 3d brick decomposition to FFT decomposition
------------------------------------------------------------------------- */

void PPPMCONP::elyte_brick2fft()
{
  int n,ix,iy,iz;

  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  for (iz = nzlo_in; iz <= nzhi_in; iz++)
    for (iy = nylo_in; iy <= nyhi_in; iy++)
      for (ix = nxlo_in; ix <= nxhi_in; ix++)
        elyte_density_fft[n++] = density_brick[iz][iy][ix];

  remap->perform(elyte_density_fft,elyte_density_fft,work1);
}

void PPPMCONP::elyte_poisson1()
{
  int i,j,k,n;
  int const nfft_c = nfft;
  
  n = 0;
  for (i = 0; i < nfft_c; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);
}

void PPPMCONP::elyte_poisson2()
{
  int i,j,k,n;
  int const nfft_c = nfft;
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
  elyte_map_fft1(); // FFT rho to kspace
  elyte_fft2_u();   // FFT kspace to u
  int i,iele,iall;
  int l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR u;
  int const elenum_c = fixconp->elenum;
  int* ele2tag = fixconp->ele2tag;
  int* tag2eleall = fixconp->tag2eleall;
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

  //double const qscale = force->qqrd2e*scale;
  //for (iele = 0; iele < elenum_c; ++iele) bbb[iele] *= qscale;
  //int* eleall2ele = fixconp->eleall2ele;
  //printf("%g\t%d\n",bbb[eleall2ele[0]],ele2tag[eleall2ele[0]]);
}

void PPPMCONP::aaa_make_grid_rho()
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
      i = atom->map(ele2tag[iele]);
      double xlo = x[i][ic] - boxlo[ic];
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

void PPPMCONP::setup_allocate()
{
  //memory->create3d_offset(elyte_density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
  //                        nxlo_out,nxhi_out,"fixconp:elyte_density_brick");
  if (differentiation_flag == 0) {
  memory->create3d_offset(u_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                            nxlo_out,nxhi_out,"fixconp:u_brick");
  }
  //memory->create(elyte_density_fft,nfft_both,"fixconp:elyte_density_fft");
}

void PPPMCONP::ele_allocate(int elenum_all)
{
  memory->grow(eleall_grid,elenum_all,3,"fixconp:eleall_grid");
  memory->grow(eleall_rho,elenum_all,3,order,"fixconp:eleall_rho");
}

void PPPMCONP::elyte_allocate(int elytenum)
{
  memory->grow(j2i,elytenum,"fixconp:j2i");
  memory->grow(elyte_grid,elytenum,3,"fixconp:elyte_grid");
}

void PPPMCONP::setup_deallocate()
{
  if (differentiation_flag == 0) memory->destroy3d_offset(u_brick,nzlo_out,nylo_out,nxlo_out);
  //memory->destroy3d_offset(elyte_density_brick,nzlo_out,nylo_out,nxlo_out);
  //memory->destroy(elyte_density_fft);
}

void PPPMCONP::ele_deallocate()
{
  memory->destroy(eleall_grid);
  memory->destroy(eleall_rho);
}

void PPPMCONP::elyte_deallocate()
{
  memory->destroy(j2i);
  memory->destroy(elyte_grid);
}

/* ----------------------------------------------------------------------
   pack ghost values into buf to send to another proc
------------------------------------------------------------------------- */

void PPPMCONP::pack_reverse_grid(int flag, void *vbuf, int nlist, int *list)
{

  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;

  // if (flag == REVERSE_RHO_ELYTE) {
  //   FFT_SCALAR *src = &elyte_density_brick[nzlo_out][nylo_out][nxlo_out];
  //   for (int i = 0; i < nlist; i++) {
  //     buf[i] = src[list[i]];
  //   }
  // }
  /* else */ if (flag == REVERSE_RHO) {
    FFT_SCALAR *src = &density_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[i] = src[list[i]];
    }
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's ghost values from buf and add to own values
------------------------------------------------------------------------- */

void PPPMCONP::unpack_reverse_grid(int flag, void *vbuf, int nlist, int *list)
{

  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;

  // if (flag == REVERSE_RHO_ELYTE) {
  //  FFT_SCALAR *dest = &elyte_density_brick[nzlo_out][nylo_out][nxlo_out];
  //  for (int i = 0; i < nlist; i++) {
  //    dest[list[i]] += buf[i];
  //  }
  //}
  /* else */ if (flag == REVERSE_RHO) {
    FFT_SCALAR *dest = &density_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      dest[list[i]] += buf[i];
    }
  }
}

/* ----------------------------------------------------------------------
   pack own values to buf to send to another proc
------------------------------------------------------------------------- */

void PPPMCONP::pack_forward_grid(int flag, void *vbuf, int nlist, int *list)
{
  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;

  int n = 0;

  if (flag == FORWARD_IK) {
    FFT_SCALAR *xsrc = &vdx_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *ysrc = &vdy_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *zsrc = &vdz_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = xsrc[list[i]];
      buf[n++] = ysrc[list[i]];
      buf[n++] = zsrc[list[i]];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *src = &u_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  } else if (flag == FORWARD_IK_PERATOM) {
    FFT_SCALAR *esrc = &u_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      if (eflag_atom) buf[n++] = esrc[list[i]];
      if (vflag_atom) {
        buf[n++] = v0src[list[i]];
        buf[n++] = v1src[list[i]];
        buf[n++] = v2src[list[i]];
        buf[n++] = v3src[list[i]];
        buf[n++] = v4src[list[i]];
        buf[n++] = v5src[list[i]];
      }
    }
  } else if (flag == FORWARD_AD_PERATOM) {
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = v0src[list[i]];
      buf[n++] = v1src[list[i]];
      buf[n++] = v2src[list[i]];
      buf[n++] = v3src[list[i]];
      buf[n++] = v4src[list[i]];
      buf[n++] = v5src[list[i]];
    }
  } // else if (flag == FORWARD_ELYTE) {
    // FFT_SCALAR *esrc = &elyte_u_brick[nzlo_out][nylo_out][nxlo_out];
    // for (int i = 0; i < nlist; i++) {
    //  buf[n++] = esrc[list[i]];
    // }
  // }
}

/* ----------------------------------------------------------------------
   unpack another proc's own values from buf and set own ghost values
------------------------------------------------------------------------- */

void PPPMCONP::unpack_forward_grid(int flag, void *vbuf, int nlist, int *list)
{
  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;

  int n = 0;

  if (flag == FORWARD_IK) {
    FFT_SCALAR *xdest = &vdx_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *ydest = &vdy_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *zdest = &vdz_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      xdest[list[i]] = buf[n++];
      ydest[list[i]] = buf[n++];
      zdest[list[i]] = buf[n++];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *dest = &u_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] = buf[i];
  } else if (flag == FORWARD_IK_PERATOM) {
    FFT_SCALAR *esrc = &u_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      if (eflag_atom) esrc[list[i]] = buf[n++];
      if (vflag_atom) {
        v0src[list[i]] = buf[n++];
        v1src[list[i]] = buf[n++];
        v2src[list[i]] = buf[n++];
        v3src[list[i]] = buf[n++];
        v4src[list[i]] = buf[n++];
        v5src[list[i]] = buf[n++];
      }
    }
  } else if (flag == FORWARD_AD_PERATOM) {
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      v0src[list[i]] = buf[n++];
      v1src[list[i]] = buf[n++];
      v2src[list[i]] = buf[n++];
      v3src[list[i]] = buf[n++];
      v4src[list[i]] = buf[n++];
      v5src[list[i]] = buf[n++];
    }
  } // else if (flag == FORWARD_ELYTE) {
    // FFT_SCALAR *esrc = &elyte_u_brick[nzlo_out][nylo_out][nxlo_out];
    // for (int i = 0; i < nlist; i++) {
    //  esrc[list[i]] = buf[n++];
    // }
  //}
}

/* ----------------------------------------------------------------------
   pack ghost values into buf to send to another proc
------------------------------------------------------------------------- */

