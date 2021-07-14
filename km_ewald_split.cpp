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

#include "km_ewald_split.h"
#include "force.h"
#include "atom.h"
#include "memory.h"
#include "mpi.h"
#include "kspace.h"
#include "domain.h"
#include "math_const.h"

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429
#define ERFC_MAX  5.8        // erfc(ERFC_MAX) ~ double machine epsilon (2^-52)

using namespace LAMMPS_NS;
using namespace MathConst;

KSpaceModuleEwaldSplit::KSpaceModuleEwaldSplit(LAMMPS *lmp) : 
  KSpaceModule(),Pointers(lmp),ug(nullptr),
  kxvecs(nullptr),kyvecs(nullptr),kzvecs(nullptr),
  cs(nullptr),sn(nullptr),csk(nullptr),snk(nullptr),
  qj_global(nullptr),kcount_dims(nullptr),
  kxy_list(nullptr),kz_list(nullptr),
  sfacrl(nullptr),sfacrl_all(nullptr),
  sfacim(nullptr),sfacim_all(nullptr)
{
  slabflag = 0;
  aread_sincos_a = false;
}

KSpaceModuleEwaldSplit::~KSpaceModuleEwaldSplit()
{
  setup_deallocate();
  elyte_deallocate();
  ele_deallocate();
}

void KSpaceModuleEwaldSplit::conp_setup(bool /* lowmemflag */)
{
  g_ewald = force->kspace->g_ewald;
  slab_volfactor = force->kspace->slab_volfactor;
  slabflag = force->kspace->slabflag;
  double accuracy = force->kspace->accuracy;

  int i;
  double qsqsum = 0.0;
  for (i = 0; i < atom->nlocal; i++) {
    qsqsum += atom->q[i]*atom->q[i];
  }
  double tmp,q2;
  MPI_Allreduce(&qsqsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsqsum = tmp;
  q2 = qsqsum * force->qqrd2e / force->dielectric;

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd_slab;

  bigint natoms = atom->natoms;
  double err;
  kxmax = 1;
  kymax = 1;
  kzmax = 1;

  err = rms(kxmax,xprd,natoms,q2);
  while (err > accuracy) {
    kxmax++;
    err = rms(kxmax,xprd,natoms,q2);
  }

  err = rms(kymax,yprd,natoms,q2);
  while (err > accuracy) {
    kymax++;
    err = rms(kymax,yprd,natoms,q2);
  }

  err = rms(kzmax,zprd_slab,natoms,q2);
  while (err > accuracy) {
    kzmax++;
    err = rms(kzmax,zprd_slab,natoms,q2);
  }

  kmax = MAX(kxmax,kymax);
  kmax = MAX(kmax,kzmax);
  kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;


  double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
  double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
  double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
  gsqmx = MAX(gsqxmx,gsqymx);
  gsqmx = MAX(gsqmx,gsqzmx);

  gsqmx *= 1.00001;
  setup_allocate();
  make_kvecs_ewald();
  make_ug_from_kvecs();
  make_kxy_list_from_kvecs();
  kmax_created = kmax;
}

void KSpaceModuleEwaldSplit::a_read()
{
  int const elenum_c = fixconp->elenum;
  double **csk_one,**snk_one;
  memory->create(csk_one,kcount_flat,elenum_c,"fixconp:csk_one");
  memory->create(snk_one,kcount_flat,elenum_c,"fixconp:snk_one");
  sincos_a_ele(csk_one,snk_one);
  sincos_a_transpose(csk_one,csk);
  sincos_a_transpose(snk_one,snk);
  memory->destroy(csk_one);
  memory->destroy(snk_one);
  aread_sincos_a = true;
}

void KSpaceModuleEwaldSplit::sincos_a_transpose(double** k_one, double** k_dest) {
  int const elenum_c = fixconp->elenum;
  int i,k;
  for (i = 0; i < elenum_c; ++i) {
    for (k = 0; k < kcount_flat; ++k) {
      k_dest[i][k] = k_one[k][i];
    }
  }
}

void KSpaceModuleEwaldSplit::a_cal(double* aaa)
{
  a_read();
  aaa_from_sincos_a(aaa);
}

void KSpaceModuleEwaldSplit::b_cal(double* bbb)
{
  sincos_b();
  bbb_from_sincos_b(bbb);
}

void KSpaceModuleEwaldSplit::setup_allocate()
{
  if (kcount_dims == nullptr) kcount_dims = new int[7];
  if (kxvecs != nullptr) delete [] kxvecs;
  kxvecs = new int[kmax3d];
  memset(kxvecs,0,kmax3d*sizeof(int));
  if (kyvecs != nullptr) delete [] kyvecs;
  kyvecs = new int[kmax3d];
  memset(kyvecs,0,kmax3d*sizeof(int));
  if (kzvecs != nullptr) delete [] kzvecs;
  kzvecs = new int[kmax3d];
  memset(kzvecs,0,kmax3d*sizeof(int));
  if (ug != nullptr) delete [] ug;
  ug = new double[kmax3d];
  if (sfacrl != nullptr) delete [] sfacrl;
  sfacrl = new double[kmax3d];
  if (sfacim != nullptr) delete [] sfacim;
  sfacim = new double[kmax3d];
  if (sfacrl_all != nullptr) delete [] sfacrl_all;
  sfacrl_all = new double[kmax3d];
  if (sfacim_all != nullptr) delete [] sfacim_all;
  sfacim_all = new double[kmax3d];
}

void KSpaceModuleEwaldSplit::setup_deallocate()
{
  delete [] kcount_dims;
  delete [] kxy_list;
  delete [] kz_list;
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;
  delete [] ug;
  delete [] sfacrl;
  delete [] sfacim;
  delete [] sfacrl_all;
  delete [] sfacim_all;
}

void KSpaceModuleEwaldSplit::conp_post_neighbor(
  bool do_elyte_alloc, bool do_ele_alloc)
{
  if (do_elyte_alloc) {
    int elytenum = fixconp->elytenum;
    elyte_allocate(elytenum);
  }
  //if (do_ele_alloc) {
  //  int elenum_all = fixconp->elenum_all;
  //  ele_allocate(elenum_all);
  //}
  int elenum = fixconp->elenum;
  ele_allocate(elenum);
  if (aread_sincos_a) sincos_a_ele(csk,snk);
}

void KSpaceModuleEwaldSplit::elyte_allocate(int elytenum)
{
  if (cs != nullptr) memory->destroy(cs);
  memory->create(cs,kcount_flat,elytenum,"fixconp:cs");
  if (sn != nullptr) memory->destroy(sn);
  memory->create(sn,kcount_flat,elytenum,"fixconp:sn");
  memory->grow(qj_global,elytenum,"fixconp:qj_global");
}

void KSpaceModuleEwaldSplit::elyte_deallocate()
{
  memory->destroy(cs);
  memory->destroy(sn);
  memory->destroy(qj_global);
}

void KSpaceModuleEwaldSplit::ele_allocate(int elenum)
{
  if (csk != nullptr) memory->destroy(csk);
  memory->create(csk,elenum,kcount_flat,"fixconp:csk");
  if (snk != nullptr) memory->destroy(snk);
  memory->create(snk,elenum,kcount_flat,"fixconp:snk");
}

void KSpaceModuleEwaldSplit::ele_deallocate()
{
  memory->destroy(csk);
  memory->destroy(snk);
}

double KSpaceModuleEwaldSplit::rms(int km, double prd, bigint natoms, double q2)
{
  double value = 2.0*q2*g_ewald/prd *
    sqrt(1.0/(MY_PI*km*natoms)) *
    exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));
  return value;
}

void KSpaceModuleEwaldSplit::make_kvecs_ewald()
{
  int k,l,m,ic;
  double sqk;
  kcount = 0;
  for (int i = 0; i < 7; ++i) kcount_dims[i] = 0;

  int const kmax_c = kmax;
  int const kmaxes[3] = {kxmax,kymax,kzmax};
  double unitksq[3];

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ++ic) {
    unitksq[ic] = unitk[ic]*unitk[ic];
    for (m = 1; m <= kmaxes[ic]; ++m) {
      sqk = m*m*unitksq[ic];
      if (sqk <= gsqmx) {
        if (ic == 0) kxvecs[kcount] = m;
        else if (ic == 1) kyvecs[kcount] = m;
        else if (ic == 2) kzvecs[kcount] = m;
        ++kcount;
        ++kcount_dims[ic];
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)
  // 1 = (0,k,l), 2 = (0,k,-l)
  // 1 = (k,0,l), 2 = (k,0,-l)
  
  int icA, icB;

  for (ic = 3; ic < 6; ++ic) {
    if (ic == 3) {icA = 0; icB = 1;}
    else if (ic == 4) {icA = 1; icB = 2;}
    else if (ic == 5) {icA = 0; icB = 2;}
    for (k = 1; k <= kmaxes[icA]; ++k) {
      for (l = 1; l <= kmaxes[icB]; ++l) {
        sqk = k*k*unitksq[icA] + l*l*unitksq[icB];
        if (sqk <= gsqmx) {
          if (ic == 3) {
            kxvecs[kcount] = k; kyvecs[kcount] =  l; ++kcount;
            kxvecs[kcount] = k; kyvecs[kcount] = -l; ++kcount;
          }
          else if (ic == 4) {
            kyvecs[kcount] = k; kzvecs[kcount] =  l; ++kcount;
            kyvecs[kcount] = k; kzvecs[kcount] = -l; ++kcount;
          }
          else if (ic == 5) {
            kxvecs[kcount] = k; kzvecs[kcount] =  l; ++kcount;
            kxvecs[kcount] = k; kzvecs[kcount] = -l; ++kcount;
          }
          ++kcount_dims[ic];
        }
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kmaxes[0]; ++k) {
    for (l = 1; l <= kmaxes[1]; ++l) {
      for (m = 1; m <= kmaxes[2]; ++m) {
        sqk = k*k*unitksq[0] + l*l*unitksq[1] + m*m*unitksq[2];
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k; kyvecs[kcount] =  l; kzvecs[kcount] =  m; ++kcount;
          kxvecs[kcount] = k; kyvecs[kcount] =  l; kzvecs[kcount] = -m; ++kcount;
          kxvecs[kcount] = k; kyvecs[kcount] = -l; kzvecs[kcount] =  m; ++kcount;
          kxvecs[kcount] = k; kyvecs[kcount] = -l; kzvecs[kcount] = -m; ++kcount;
          ++kcount_dims[6];
        }
      }
    }
  }
  kcount_flat = kcount_dims[0]+kcount_dims[1]+kcount_dims[2]+2*kcount_dims[3];
  kcount_expand = kcount_dims[4]+kcount_dims[5]+2*kcount_dims[6];
}

void KSpaceModuleEwaldSplit::make_ug_from_kvecs()
{
  int const kcount_c = kcount;
  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;
  double sqk;
  double* __restrict__ ugr = ug;

  for (int k = 0; k < kcount_c; ++k) {
    sqk  = kxvecs[k]*kxvecs[k]*unitk[0]*unitk[0];
    sqk += kyvecs[k]*kyvecs[k]*unitk[1]*unitk[1];
    sqk += kzvecs[k]*kzvecs[k]*unitk[2]*unitk[2];
    ugr[k] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
  }
}

void KSpaceModuleEwaldSplit::make_kxy_list_from_kvecs()
{
  int k, kx, ky, kf;
  if (kxy_list != nullptr) delete [] kxy_list;
  kxy_list = new int[kcount_expand];
  if (kz_list != nullptr) delete [] kz_list;
  kz_list = new int[kcount_expand];

  kf = kcount_flat;

  int const kcount_dims4_c = kcount_dims[4];
  for (k = 0; k < kcount_dims4_c; ++k) {
    kxy_list[k] = kyvecs[kf]+kcount_dims[0]-1;
    kz_list[k]  = kzvecs[kf]+kcount_dims[0]+kcount_dims[1]-1;
    kf += 2;
  }

  int const kcount_dims5_c = kcount_dims[5];
  for (k = kcount_dims4_c;
       k < kcount_dims4_c+kcount_dims5_c; ++k) {
    kxy_list[k] = kxvecs[kf]-1;
    kz_list[k]  = kzvecs[kf]+kcount_dims[0]+kcount_dims[1]-1;
    kf += 2;
  }

  int kxy = kcount_dims[0] + kcount_dims[1] + kcount_dims[2];
  int const kcount_dims6_c = kcount_dims[6];
  
  int kloc = kcount_dims4_c+kcount_dims5_c;
  
  for (k = 0; k < kcount_dims6_c; ++k) {
    kx = kxvecs[kf];
    ky = kyvecs[kf];
    while (kxvecs[kxy] != kx || kyvecs[kxy] != ky) kxy += 2;
    kxy_list[kloc] = kxy;
    kxy_list[kloc+1] = kxy+1;
    kz_list[kloc] = kzvecs[kf]+kcount_dims[0]+kcount_dims[1]-1;
    kz_list[kloc+1] = kzvecs[kf]+kcount_dims[0]+kcount_dims[1]-1;
    kf += 4;
    kloc += 2;
  }
}

void KSpaceModuleEwaldSplit::sincos_a_ele(double ** csk_one, double ** snk_one)
{
  int* ele2tag = fixconp->ele2tag;
  int i,j,m,k,ic,iele;
  int kx,ky,kz,kxy,kf;
  double **x = atom->x;
  const int elenum_c = fixconp->elenum;
  double *elex = new double[elenum_c];

  for (i = 0; i < elenum_c; ++i) {
    kf = 0;
    iele = atom->map(ele2tag[i]);
    for (ic = 0; ic < 3; ++ic) {
      double xdotk = unitk[ic]*x[iele][ic];
      csk_one[kf][i] = cos(xdotk);
      snk_one[kf][i] = sin(xdotk);
      kf += kcount_dims[ic];
    }
  }

  kf = 0;
  for (ic = 0; ic < 3; ++ic) {
    for (m = 1; m < kcount_dims[ic]; ++m) {
      double* __restrict__ cskf1=csk_one[kf+m];
      double* __restrict__ snkf1=snk_one[kf+m];
      for (i = 0; i < elenum_c; ++i) {
        cskf1[i] = csk_one[kf+m-1][i]*csk_one[kf][i] - snk_one[kf+m-1][i]*snk_one[kf][i];
        snkf1[i] = snk_one[kf+m-1][i]*csk_one[kf][i] + csk_one[kf+m-1][i]*snk_one[kf][i];
      }
    }
    kf += kcount_dims[ic];
  }

    // (k, l, 0); (k, -l, 0)
  
  for (m = 0; m < kcount_dims[3]; ++m) {
    kx = kxvecs[kf]-1;
    ky = kyvecs[kf]+kcount_dims[0]-1;
    double* __restrict__ cskf0 = csk_one[kf];
    double* __restrict__ snkf0 = snk_one[kf];
    double* __restrict__ cskf1 = csk_one[kf+1];
    double* __restrict__ snkf1 = snk_one[kf+1];
    for (i = 0; i < elenum_c; ++i) {
      // todo: tell compiler that kf, kx and ky do not alias
      cskf0[i] = csk_one[kx][i]*csk_one[ky][i] - snk_one[kx][i]*snk_one[ky][i];
      snkf0[i] = csk_one[kx][i]*snk_one[ky][i] + snk_one[kx][i]*csk_one[ky][i];
      cskf1[i] = csk_one[kx][i]*csk_one[ky][i] + snk_one[kx][i]*snk_one[ky][i];
      snkf1[i] = -csk_one[kx][i]*snk_one[ky][i] + snk_one[kx][i]*csk_one[ky][i];
    }
    kf += 2;
  }
  delete [] elex;
}

void KSpaceModuleEwaldSplit::sincos_a_comm_eleall(double ** k_one, double ** k_all)
{
  int i,k;
  const int kcount_c = kcount;
  const int elenum_all_c = fixconp->elenum_all;
  bool transposeflag = true; // will have to transpose to fit csk_p and snk_p
  if (transposeflag) {
    double *trigbuf = new double[elenum_all_c];
    for (k = 0; k < kcount_c; ++k) {
      fixconp->b_comm(k_one[k],trigbuf);
      for (i = 0; i < elenum_all_c; ++i) {
        k_all[i][k] = trigbuf[i];
      }
    }
    delete [] trigbuf;
  }
  else {
    for (k = 0; k < kcount_c; ++k) {
      fixconp->b_comm(k_one[k],k_all[k]);
    }
  }
}

void KSpaceModuleEwaldSplit::kz_expand(
  double* __restrict__ cski,
  double* __restrict__ snki,
  double* __restrict__ cski_e, //  2*kcount_expand_c
  double* __restrict__ snki_e, //  2*kcount_expand_c
  double* __restrict__ buf4 // len 4*kcount_expand_c
)
{
  int const kcount_expand_c = kcount_expand;
  
  double* __restrict__ csxyi = buf4;
  double* __restrict__ snxyi = buf4+kcount_expand_c;
  double* __restrict__ cszi  = buf4+2*kcount_expand_c;
  double* __restrict__ snzi  = buf4+3*kcount_expand_c;
  
  int ki;
  for (ki = 0; ki < kcount_expand_c; ++ki) {
    csxyi[ki] = cski[kxy_list[ki]];
    snxyi[ki] = snki[kxy_list[ki]];
    cszi[ki] = cski[kz_list[ki]];
    snzi[ki] = snki[kz_list[ki]];
    // if (ki == 2000) printf("%d\n",ki);	
  }
  for (ki = 0; ki < kcount_expand_c; ++ki) {
    cski_e[2*ki] = csxyi[ki]*cszi[ki] - snxyi[ki]*snzi[ki];
    snki_e[2*ki] = snxyi[ki]*cszi[ki] + csxyi[ki]*snzi[ki];
    cski_e[2*ki+1] = csxyi[ki]*cszi[ki] + snxyi[ki]*snzi[ki];
    snki_e[2*ki+1] = snxyi[ki]*cszi[ki] - csxyi[ki]*snzi[ki];
  }
}

double KSpaceModuleEwaldSplit::ewald_dot_ij(
  double* __restrict__ cski,
  double* __restrict__ snki,
  double* __restrict__ cski_e, //  2*kcount_expand_c
  double* __restrict__ snki_e, //  2*kcount_expand_c
  double* __restrict__ cskj,
  double* __restrict__ snkj,
  double* __restrict__ cskj_e, //  2*kcount_expand_c
  double* __restrict__ snkj_e  //  2*kcount_expand_c
)
{
  int k;
  int const kcount_flat_c = kcount_flat;
  int const kcount_expand_c = kcount_expand;
  double aaatmp = 0;
  for (k = 0; k < kcount_flat_c; ++k) {
    aaatmp += 2*ug[k]*(cski[k]*cskj[k]+snki[k]*snkj[k]);
  }
  for (k = 0; k < 2*kcount_expand_c; ++k) {
    aaatmp += 2*ug[kcount_flat_c+k]*(cski_e[k]*cskj_e[k]
           + snki_e[k]*snkj_e[k]);
  }
  return aaatmp;
}

double KSpaceModuleEwaldSplit::ewald_dot_ii(
  double* __restrict__ cski,
  double* __restrict__ snki,
  double* __restrict__ cski_e, //  2*kcount_expand_c
  double* __restrict__ snki_e  //  2*kcount_expand_c
)
{
  int k;
  int const kcount_flat_c = kcount_flat;
  int const kcount_expand_c = kcount_expand;
  double aaatmp = 0;
  for (k = 0; k < kcount_flat_c; ++k) {
    aaatmp += 2*ug[k]*(cski[k]*cski[k]+snki[k]*snki[k]);
  }
  for (k = 0; k < 2*kcount_expand_c; ++k) {
    aaatmp += 2*ug[kcount_flat_c+k]*(cski_e[k]*cski_e[k]
           + snki_e[k]*snki_e[k]);
  }
  return aaatmp;
}

double KSpaceModuleEwaldSplit::ewald_dot_ib(
  int elei,
  double* __restrict__ cski_e, //  2*kcount_expand_c
  double* __restrict__ snki_e, //  2*kcount_expand_c
  double* __restrict__ buf4
)
{
  double* __restrict__ cski = csk[elei];
  double* __restrict__ snki = snk[elei];
  kz_expand(cski,snki,cski_e,snki_e,buf4);
  int k;
  int const kcount_flat_c = kcount_flat;
  int const kcount_expand_c = kcount_expand;
  double aaatmp = 0;
  for (k = 0; k < kcount_flat_c; ++k) {
    aaatmp += 2*ug[k]*(cski[k]*sfacrl_all[k]+snki[k]*sfacim_all[k]);
  }
  for (k = 0; k < 2*kcount_expand_c; ++k) {
    aaatmp += 2*ug[kcount_flat_c+k]*(cski_e[k]*sfacrl_all[kcount_flat_c+k]
           + snki_e[k]*sfacim_all[kcount_flat_c+k]);
  }
  return aaatmp;
}

void KSpaceModuleEwaldSplit::aaa_from_sincos_a(double* aaa)
{
  int* ele2eleall = fixconp->ele2eleall;
  int* ele2tag = fixconp->ele2tag;
  int const elenum_c = fixconp->elenum;
  int const elenum_all_c = fixconp->elenum_all;
  double CON_s2overPIS = sqrt(2.0)/MY_PIS;
  double CON_2overPIS = 2.0/MY_PIS;
  
  int i,j,k,idx1d,n;
  int const kcount_c = kcount;
  double aaatmp;

  // build buffers
  double* cski_e = new double[2*kcount_expand];
  double* snki_e = new double[2*kcount_expand];
  double* cskj_e = new double[2*kcount_expand];
  double* snkj_e = new double[2*kcount_expand];
  double* buf4 = new double[4*kcount_expand];

  // do self entries -- self-other entries done with b_bcast
  for (i = 0; i < elenum_c; ++i) {
    int const elealli = ele2eleall[i];
    kz_expand(csk[i],snk[i],cski_e,snki_e,buf4);
    for (j = 0; j < elenum_c; ++j) {
      int eleallj = ele2eleall[j];
      if ((eleallj < elealli) ^ ((elealli%2) ^ (eleallj%2))) {
        kz_expand(csk[j],snk[j],cskj_e,snkj_e,buf4);
        idx1d = i*elenum_all_c + eleallj;
        aaa[idx1d] = ewald_dot_ij(csk[i],snk[i],cski_e,
          snki_e,csk[j],snk[j],cskj_e,snkj_e);
      }
    }
    idx1d = i*elenum_all_c+elealli;
    aaa[idx1d] = ewald_dot_ii(csk[i],snk[i],cski_e,snki_e);
    aaa[idx1d] += CON_s2overPIS*fixconp->eta-CON_2overPIS*g_ewald;
  }
  // comm stuff

  int me,nprocs;
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  if (nprocs > 1) {
    int ki,kj,jkf;
    for (n = 0; n < nprocs; ++n) {
      int const elenum_n_c = fixconp->elenum_list[n];
      if (elenum_n_c > 0) {
        int* eleall_n_list;
        double *cskbuf,*snkbuf;
        if (me == n) {
          eleall_n_list = fixconp->ele2eleall;
          cskbuf = &(csk[0][0]);
          snkbuf = &(snk[0][0]);
        } else {
          eleall_n_list = new int[elenum_n_c];
          cskbuf = new double[elenum_n_c*kcount_flat];
          snkbuf = new double[elenum_n_c*kcount_flat];
        }
        MPI_Barrier(world);
        fixconp->b_bcast(n,kcount_flat,eleall_n_list,cskbuf);
        fixconp->b_bcast(n,kcount_flat,eleall_n_list,snkbuf);
        MPI_Barrier(world);
        //if (me == n) {
          //delete eleall_n_list;
          //cskbuf = nullptr;
          //snkbuf = nullptr;
          //delete eleall_n_list;
          //delete cskbuf;
          //delete snkbuf;
        //}
        if (me != n) {
          for (i = 0; i < elenum_c; ++i) {
            int const elealli = ele2eleall[i];
            kz_expand(csk[i],snk[i],cski_e,snki_e,buf4);
            for (j = 0; j < elenum_n_c; ++j) {
              int const eleallj = eleall_n_list[j];
              if ((eleallj < elealli) ^ 
                  ((elealli%2) ^ (eleallj%2))) {
                jkf = j*kcount_flat;
                kz_expand(&cskbuf[jkf],&snkbuf[jkf],cskj_e,snkj_e,buf4);
                idx1d = i*elenum_all_c+eleallj;
                aaa[idx1d] = ewald_dot_ij(csk[i],snk[i],cski_e,
                  snki_e,&cskbuf[jkf],&snkbuf[jkf],cskj_e,snkj_e);
              }
            }
          }
          delete [] eleall_n_list;
          delete [] cskbuf;
          delete [] snkbuf;
        }
        MPI_Barrier(world);
      }
    }
  }

  delete [] cski_e;
  delete [] snki_e;
  delete [] cskj_e;
  delete [] snkj_e;
  delete [] buf4;

  // implement slab corrections
  if (slabflag == 1) {
    double CON_4PIoverV = MY_4PI/volume;
    double *eleallz = new double[elenum_all_c];
    double *elez = new double[elenum_c];
    double **x = atom->x;
    for (i = 0; i < elenum_c; ++i) {
      elez[i] = x[atom->map(ele2tag[i])][2];
    }
    fixconp->b_comm(elez,eleallz);
    delete [] elez;
    for (i = 0; i < elenum_c; ++i) {
      int const elealli = ele2eleall[i];
      idx1d = i*elenum_all_c;
      for (j = 0; j <= elealli; ++j) {
        aaa[idx1d] += CON_4PIoverV*eleallz[elealli]*eleallz[j];
        idx1d++;
      }
    } 
  }
}

void KSpaceModuleEwaldSplit::sincos_b()
{
  int i,j,k,l,m,n,ic,kf;
  int kx,ky,kz,kxy;

  double temprl0,temprl1,temprl2,temprl3;
  double tempim0,tempim1,tempim2,tempim3;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  double* __restrict__ qj = qj_global;

  j = 0;
  kf = 0;

  for (i = 0; i < nlocal; ++i) {
    if (fixconp->electrode_check(i) == 0 && q[i] != 0){
      qj[j] = q[i];
      kf = 0;
      for (ic = 0; ic < 3; ++ic) {
        double xdotk = unitk[ic]*x[i][ic];
        cs[kf][j] = cos(xdotk);
        sn[kf][j] = sin(xdotk);
        kf += kcount_dims[ic];
      }
      ++j;
    }
  }
  const int jmax = j;
  kf = 0;
  for (ic = 0; ic < 3; ++ic) {
    temprl0 = 0;
    tempim0 = 0;
    for (j = 0; j < jmax; ++j) {
      temprl0 += qj[j]*cs[kf][j];
      tempim0 += qj[j]*sn[kf][j];
    }
    sfacrl[kf] = temprl0;
    sfacim[kf] = tempim0;
    for (m = 1; m < kcount_dims[ic]; ++m) {
      double* __restrict__ cskf1=cs[kf+m];
      double* __restrict__ snkf1=sn[kf+m];
      temprl0 = 0;
      tempim0 = 0;
      for (j = 0; j < jmax; ++j) {
        cskf1[j] = cs[kf+m-1][j]*cs[kf][j] - sn[kf+m-1][j]*sn[kf][j];
        snkf1[j] = sn[kf+m-1][j]*cs[kf][j] + cs[kf+m-1][j]*sn[kf][j];
        temprl0 += qj[j]*cskf1[j];
        tempim0 += qj[j]*snkf1[j];
      }
      sfacrl[kf+m] = temprl0;
      sfacim[kf+m] = tempim0;
    }
    kf += kcount_dims[ic];
  }
  
  // (k, l, 0); (k, -l, 0)

  for (m = 0; m < kcount_dims[3]; ++m) {
    kx = kxvecs[kf]-1;
    ky = kyvecs[kf]+kcount_dims[0]-1;
    temprl0 = 0;
    tempim0 = 0;
    temprl1 = 0;
    tempim1 = 0;
    double* __restrict__ cskf0 = cs[kf];
    double* __restrict__ snkf0 = sn[kf];
    double* __restrict__ cskf1 = cs[kf+1];
    double* __restrict__ snkf1 = sn[kf+1];
    for (j = 0; j < jmax; ++j) {
      cskf0[j] = cs[kx][j]*cs[ky][j] - sn[kx][j]*sn[ky][j];
      snkf0[j] = cs[kx][j]*sn[ky][j] + sn[kx][j]*cs[ky][j];
      temprl0 += qj[j]*cskf0[j];
      tempim0 += qj[j]*snkf0[j];
      cskf1[j] = cs[kx][j]*cs[ky][j] + sn[kx][j]*sn[ky][j];
      snkf1[j] = -cs[kx][j]*sn[ky][j] + sn[kx][j]*cs[ky][j];
      temprl1 += qj[j]*cskf1[j];
      tempim1 += qj[j]*snkf1[j];
    }
    sfacrl[kf] = temprl0;
    sfacim[kf] = tempim0;
    sfacrl[kf+1] = temprl1;
    sfacim[kf+1] = tempim1;
    kf += 2;
  }

  // (0, l, m); (0, l, -m)
  // (k, 0, m); (k, 0, -m)
  // (k, l, m); (k, l, -m)
  // (k, -l, m); (k, -l, -m)

  for (m = 0; m < kcount_expand; ++m) {
    kxy = kxy_list[m];
    kz = kz_list[m];
    temprl0 = 0;
    tempim0 = 0;
    temprl1 = 0;
    tempim1 = 0;
    for (j = 0; j < jmax; ++j) {
      temprl0 += qj[j]*(cs[kxy][j]*cs[kz][j]-sn[kxy][j]*sn[kz][j]);
      tempim0 += qj[j]*(cs[kxy][j]*sn[kz][j]+sn[kxy][j]*cs[kz][j]);
      temprl1 += qj[j]*(cs[kxy][j]*cs[kz][j]+sn[kxy][j]*sn[kz][j]);
      tempim1 += qj[j]*(-cs[kxy][j]*sn[kz][j]+sn[kxy][j]*cs[kz][j]);
    }
    sfacrl[kf] = temprl0;
    sfacim[kf] = tempim0;
    sfacrl[kf+1] = temprl1;
    sfacim[kf+1] = tempim1;
    kf += 2;
  }

  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- */
void KSpaceModuleEwaldSplit::bbb_from_sincos_b(double* bbb)
{
  int* ele2tag = fixconp->ele2tag;
  int* ele2eleall = fixconp->ele2eleall;
  int k,elei,elealli;
  int const elenum_c = fixconp->elenum;
  int const kcount_c = kcount;
  double bbbtmp;
  // int one = 1;
  
  double* cski_e = new double[2*kcount_expand];
  double* snki_e = new double[2*kcount_expand];
  double* buf4   = new double[4*kcount_expand];

  for (elei = 0; elei < elenum_c; ++elei) {
    bbb[elei] = ewald_dot_ib(elei,cski_e,
      snki_e,buf4);
  }

  delete [] cski_e;
  delete [] snki_e;
  delete [] buf4;

  if (slabflag == 1) {
    int i;
    int nlocal = atom->nlocal;
    double **x = atom->x;
    double *q = atom->q;
    double slabcorr = 0.0;
    #pragma ivdep
    for (i = 0; i < nlocal; i++) {
      if (fixconp->electrode_check(i) == 0) {
        slabcorr += 4*q[i]*MY_PI*x[i][2]/volume;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE,&slabcorr,1,MPI_DOUBLE,MPI_SUM,world);
    for (elei = 0; elei < elenum_c; ++elei) {
      i = atom->map(ele2tag[elei]);
      bbb[elei] -= x[i][2]*slabcorr;
    }
  }
}

void KSpaceModuleEwaldSplit::make_kvecs_brick()
{
  int k,l,m,ic;
  kcount = 0;
  for (int i = 0; i < 7; ++i) kcount_dims[i] = 0;

  int const kmax_c = kmax;
  int const kmaxes[3] = {kxmax,kymax,kzmax};
  double unitksq[3];

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ++ic) {
    unitksq[ic] = unitk[ic]*unitk[ic];
    for (m = 1; m <= kmaxes[ic]; ++m) {
      if (ic == 0) kxvecs[kcount] = m;
      else if (ic == 1) kyvecs[kcount] = m;
      else if (ic == 2) kzvecs[kcount] = m;
      ++kcount;
      ++kcount_dims[ic];
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)
  // 1 = (0,k,l), 2 = (0,k,-l)
  // 1 = (k,0,l), 2 = (k,0,-l)
  
  int icA, icB;

  for (ic = 3; ic < 6; ++ic) {
    if (ic == 3) {icA = 0; icB = 1;}
    else if (ic == 4) {icA = 1; icB = 2;}
    else if (ic == 5) {icA = 0; icB = 2;}
    for (k = 0; k <= kmaxes[icA]; ++k) {
      for (l = 0; l <= kmaxes[icB]; ++l) {
        if (ic == 3) {
          kxvecs[kcount] = k; kyvecs[kcount] =  l; ++kcount;
          kxvecs[kcount] = k; kyvecs[kcount] = -l; ++kcount;
        }
        else if (ic == 4) {
          kyvecs[kcount] = k; kzvecs[kcount] =  l; ++kcount;
          kyvecs[kcount] = k; kzvecs[kcount] = -l; ++kcount;
        }
        else if (ic == 5) {
          kxvecs[kcount] = k; kzvecs[kcount] =  l; ++kcount;
          kxvecs[kcount] = k; kzvecs[kcount] = -l; ++kcount;
        }
        ++kcount_dims[ic];
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,l,-m), 3 = (k,-l,m), 4 = (k,-l,-m)

  for (k = 0; k <= kmaxes[0]; ++k) {
    for (l = 0; l <= kmaxes[1]; ++l) {
      for (m = 0; m <= kmaxes[2]; ++m) {
        kxvecs[kcount] = k; kyvecs[kcount] =  l; kzvecs[kcount] =  m; ++kcount;
        kxvecs[kcount] = k; kyvecs[kcount] =  l; kzvecs[kcount] = -m; ++kcount;
        kxvecs[kcount] = k; kyvecs[kcount] = -l; kzvecs[kcount] =  m; ++kcount;
        kxvecs[kcount] = k; kyvecs[kcount] = -l; kzvecs[kcount] = -m; ++kcount;
        ++kcount_dims[6];
      }
    }
  }
  kcount_flat = kcount_dims[0]+kcount_dims[1]+kcount_dims[2]+2*kcount_dims[3];
  kcount_expand = kcount_dims[4]+kcount_dims[5]+2*kcount_dims[6];
}
