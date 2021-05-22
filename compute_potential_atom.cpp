// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_potential_atom.h"
#include <cstring>
#include "atom.h"
#include "update.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "neigh_list.h"

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429
#define ERFC_MAX  5.8        // erfc(ERFC_MAX) ~ double machine epsilon (2^-52)

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePotentialAtom::ComputePotentialAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  potential(nullptr),kspmod(nullptr)
{
  if (narg < 3) error->all(FLERR,"Illegal compute pe/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;
  peatomflag = 1;
  timeflag = 1;
  comm_reverse = 1;

  if (narg == 3) {
    pairflag = true;
    kspaceflag = true;
  } else {
    pairflag = false;
    kspaceflag = false;
    int iarg = 3;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"pair") == 0) pairflag = true;
      else if (strcmp(arg[iarg],"kspace") == 0) kspaceflag = true;
      else error->all(FLERR,"Illegal compute pe/atom command");
      iarg++;
    }
  }

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputePotentialAtom::~ComputePotentialAtom()
{
  memory->destroy(potential);
}

/* ---------------------------------------------------------------------- */

void ComputePotentialAtom::setup()
{
  evscale = force->qe2f/force->qqr2e;
  if (kspaceflag) {
  kspmod = dynamic_cast<KSpaceModule *>(force->kspace);
  if (kspmod == nullptr)
    error->all(FLERR,"Compute requires a compatible KSpace provider like pppm/conp");
  }
}

/* ---------------------------------------------------------------------- */

void ComputePotentialAtom::compute_peratom()
{
  int i;

  invoked_peratom = update->ntimestep;
  if (update->eflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom energy was not tallied on needed timestep");

  // grow local energy array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(potential);
    nmax = atom->nmax;
    memory->create(potential,nmax,"pe/atom:energy");
    vector_atom = potential;
  }

  // npair includes ghosts if either newton flag is set
  //   b/c some bonds/dihedrals call pair::ev_tally with pairwise info
  // nbond includes ghosts if newton_bond is set
  // ntotal includes ghosts if either newton flag is set
  // KSpace includes ghosts if tip4pflag is set

  int nlocal = atom->nlocal;
  int npair = nlocal;
  int ntotal = nlocal;
  int nkspace = nlocal;
  if (force->newton) npair += atom->nghost;
  if (force->newton) ntotal += atom->nghost;
  if (force->kspace && force->kspace->tip4pflag) nkspace += atom->nghost;

  // clear local energy array

  for (i = 0; i < ntotal; i++) potential[i] = 0.0;

  // add in per-atom contributions from each force

  if (pairflag) compute_pair_potential();

  if (kspaceflag && force->kspace && force->kspace->compute_flag) {
    for (i = 0; i < nkspace; i++) potential[i] += kspmod->compute_particle_potential(i);
    // add slabcorr!!!!
  }

  if (force->newton || (force->kspace && force->kspace->tip4pflag))
    comm->reverse_comm_compute(this);

  for (i = 0; i < ntotal; i++) potential[i] *= evscale;
}

/* ---------------------------------------------------------------------- */

int ComputePotentialAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) buf[m++] = potential[i];
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputePotentialAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    potential[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputePotentialAtom::memory_usage()
{
  double bytes = (double)nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
    compute pairwise Coulomb potential
------------------------------------------------------------------------- */

void ComputePotentialAtom::compute_pair_potential()
{
  int i,j,k,ii,jj,jnum,itype,jtype,idx1d;
  int checksum,elei,elej,elealli,eleallj,tagi;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz;
  double r,r2inv,rsq,grij,etarij,expm2,t,erfc,dudq;
  double forcecoul,ecoul,prefactor,fpair;

  class NeighList* list = force->pair->list; // to-do: this doesn't work for hybrid
  int inum = list->inum;
  int nlocal = atom->nlocal;
  int newton = force->newton_pair;
  int *atomtype = atom->type;
  int *tag = atom->tag;
  int *mask = atom->mask;
  int *ilist = list->ilist;
  int *jlist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  double qqrd2e = force->qqrd2e;
  double **cutsq = force->pair->cutsq;
  int itmp;

  double g_ewald=force->kspace->g_ewald;
  double *p_cut_coul = (double *) force->pair->extract("cut_coul",itmp);
  double cut_coulsq = (*p_cut_coul)*(*p_cut_coul);
  double cut_erfc = ERFC_MAX*ERFC_MAX/(g_ewald*g_ewald);
  if (cut_coulsq > cut_erfc) cut_coulsq = cut_erfc;
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  bool gcib,gcjb; // group checks

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    gcib = (mask[i] & groupbit);
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = atomtype[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      gcjb = (mask[i] & groupbit);
      if ((gcib ^ gcjb) &&
          (newton || gcib || j < nlocal)) {
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = atomtype[j];
        if (rsq < cutsq[itype][jtype]) {
          if (rsq < cut_coulsq) {
            r2inv = 1.0/rsq;
            dudq = 0.0;
            r = sqrt(rsq);
            grij = g_ewald * r;
            expm2 = exp(-grij*grij);
            t = 1.0 / (1.0 + EWALD_P*grij);
            erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            dudq = erfc/r;
	      }
          //etarij = eta*r;
	      //if (etarij < ERFC_MAX) {
          //  expm2 = exp(-etarij*etarij);
          //  t = 1.0 / (1.0+EWALD_P*etarij);
          //  erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
          //  dudq -= erfc/r;
	      //}
          if (gcib) {
            potential[i] -= q[j]*dudq;
          }
	      else if (j < nlocal || newton) {
            potential[j] -= q[i]*dudq;
          }
        }
      }
    }
  }
}