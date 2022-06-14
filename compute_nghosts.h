/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Version: May/2021
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(nghosts,ComputeNghosts);
// clang-format on
#else

#ifndef LMP_COMPUTE_NGHOSTS_H
#define LMP_COMPUTE_NGHOSTS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeNghosts : public Compute {
 public:
  ComputeNghosts(class LAMMPS *, int, char **);
  ~ComputeNghosts();
  void init() override {}
  void compute_peratom() override;
  double memory_usage() override;

 private:
  int nmax;
  double *nghosts;
};

} // namespace LAMMPS_NS

#endif
#endif
