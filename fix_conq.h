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

FixStyle(conq,FixConq)

#else

#ifndef LMP_FIX_CONQ_H
#define LMP_FIX_CONQ_H

#include "fix_conp.h"
#include "pair.h"
#include "kspacemodule.h"

namespace LAMMPS_NS {

class FixConq : public FixConp {
 public:
  FixConq(class LAMMPS *, int, char **);
  ~FixConq() {}
  void update_charge();

 protected:
  double leftcharge;
  int leftchargestyle;
};

}

#endif
#endif

// crosslist naming conventions:
// the array 'A2B' holds values such that A2B[A] == B
// for example, 'ele2eleall[elei]' returns the eleall index
// of atom with ele index i
//
// Important lists:
// eleall: global permanent numbering of electrode atoms
// from 0 to elenum_all-1
// ele: local volatile numbering of electrode atoms
// from 0 to elenum-1
// i: local volatile numbering of all atoms
// from 0 to nlocal-1 for locals 
// and nlocal to nlocal+nghost-1 for ghosts
// tag: global permanent numbering of all atoms
// from *1* to natoms
//
// Important cross-lists:
// ele2eleall: length elenum     list holding eleall idx
// eleall2ele: length elenum_all list holding ele    idx
// ele2tag:    length elenum     list holding tag    idx
// eleall2tag: length elenum_all list holding tag    idx
// tag2eleall: length natoms+1   list holding eleall idx
// for conversions involving i, always use tag!
// i2ele[i] = eleall2ele[tag2eleall[tag[i]]]
// ele2i[ele] = atom->map(eleall2tag[ele2eleall[ele]])
