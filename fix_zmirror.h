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

#ifdef FIX_CLASS

FixStyle(zmirror,FixZmirror)

#else

#ifndef LMP_FIX_ZMIRROR_H
#define LMP_FIX_ZMIRROR_H

#include "fix.h"

namespace LAMMPS_NS {

class FixZmirror : public Fix {
 public:
  FixZmirror(class LAMMPS *, int, char **);
  ~FixZmirror();
  int setmask();
  void setup(int);
  void allocate();
  void post_integrate();
  void end_of_step();
  double memory_usage();
 protected:
  tagint *tag_send_buf;
  tagint *tag_recv_buf;
  double *coord_send_buf;
  double *coord_recv_buf;
  tagint send_mintag, send_maxtag;
  tagint recv_mintag, recv_maxtag;
  int me,nprocs;
  int ngroup,everynum;
  char *group2;
  int jgroup,jgroupbit;
  bool will_recv, ran_postint, allocated;
  int *nsend_all, *coord_nsend_all, *tag_displs, *coord_displs;
};
}
#endif
#endif

