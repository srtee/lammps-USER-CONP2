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

#include "compute_nghosts.h"

#include "atom.h"
#include "memory.h"

using namespace LAMMPS_NS;

ComputeNghosts::ComputeNghosts(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  nghosts(nullptr)
{
  peratom_flag = 1;
  size_peratom_cols = 0;
  nmax = 0;
}

ComputeNghosts::~ComputeNghosts()
{
  memory->destroy(nghosts);
}

void ComputeNghosts::compute_peratom()
{
  if (atom->nmax > nmax) {
    memory->destroy(nghosts);
    nmax = atom->nmax;
    memory->create(nghosts,nmax,"compute_nghosts:nghosts");
    vector_atom = nghosts;
  }

  int const nlocal = atom->nlocal;
  int const nall = nlocal + atom->nghost;

  for (int i = 0; i < nlocal; i++) nghosts[i] = 0.0;

  int* mask = atom->mask;
  tagint* tag = atom->tag;
  for (int i = nlocal; i < nall; i++) {
    if (mask[i] & groupbit) {
      int ni = atom->map(tag[i]);
      if (ni < nlocal) nghosts[ni] += 1.0;
    }
  }
}

double ComputeNghosts::memory_usage()
{
  double bytes = (double)nmax * sizeof(double);
  return bytes;
}
