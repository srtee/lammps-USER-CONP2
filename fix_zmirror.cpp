/* ---------------------------------------------------------------------
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

#include "lmptype.h"
#include "fix_zmirror.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "group.h"
#include "error.h"
#include <cstddef>

using namespace LAMMPS_NS;
using namespace FixConst;

FixZmirror::FixZmirror(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),tag_send_buf(nullptr),
  tag_recv_buf(nullptr),coord_send_buf(nullptr),
  coord_recv_buf(nullptr),ngroup(0),will_recv(false),
  send_mintag(0),send_maxtag(0),recv_mintag(0),recv_maxtag(0),
  ran_postint(false), allocated(false),coord_nsend_all(nullptr),
  nsend_all(nullptr),tag_displs(nullptr),coord_displs(nullptr)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  if (narg != 5) error->all(FLERR,"Illegal fix zmirror command"
         " (incorrect no. of parameters)");

  everynum = utils::inumeric(FLERR,arg[3],false,lmp);
  
  group2 = utils::strdup(arg[4]);
  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Fix zmirror group ID does not exist");
  jgroupbit = group->bitmask[jgroup];
  delete [] group2;
}

int FixZmirror::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= END_OF_STEP;
  return mask;
}

void FixZmirror::setup(int /* ev_flag */)
{
  int const nlocal = atom->nlocal;
  int *mask = atom->mask;
  int *tag  = atom->tag;
  tagint maxtag = 0;
  for (int i = 0; i < nlocal; ++i) maxtag = MAX(tag[i],maxtag);
  MPI_Allreduce(&maxtag,MPI_IN_PLACE,1,MPI_LMP_TAGINT,MPI_MAX,world);
  send_mintag = maxtag;
  recv_mintag = maxtag;
  for (int i = 0; i < nlocal; ++i) {
    if (mask[i] & groupbit) {
      send_mintag = MIN(tag[i],send_mintag);
      send_maxtag = MAX(tag[i],send_maxtag);
    }
    if (mask[i] & jgroupbit) {
      recv_mintag = MIN(tag[i],recv_mintag);
      recv_maxtag = MAX(tag[i],recv_maxtag);
    }
  }
  MPI_Allreduce(&send_mintag,MPI_IN_PLACE,1,MPI_LMP_TAGINT,MPI_MIN,world);
  MPI_Allreduce(&send_maxtag,MPI_IN_PLACE,1,MPI_LMP_TAGINT,MPI_MAX,world);
  MPI_Allreduce(&recv_mintag,MPI_IN_PLACE,1,MPI_LMP_TAGINT,MPI_MIN,world);
  MPI_Allreduce(&recv_maxtag,MPI_IN_PLACE,1,MPI_LMP_TAGINT,MPI_MAX,world);
  ngroup = send_maxtag - send_mintag + 1;
  int ncheck = recv_maxtag - recv_mintag + 1;
  if (ncheck != ngroup) error->all(FLERR,"Groups do not have same number of tags");
  allocate();
}

void FixZmirror::allocate()
{
  nsend_all = new int[nprocs];
  coord_nsend_all = new int[nprocs];
  tag_displs = new int[nprocs];
  coord_displs = new int[nprocs];
  tag_send_buf = new tagint[ngroup];
  tag_recv_buf = new tagint[ngroup];
  coord_send_buf = new double[3*ngroup];
  coord_recv_buf = new double[3*ngroup];
  allocated = true;
}

FixZmirror::~FixZmirror()
{
  if (allocated) {
    delete [] nsend_all;
    delete [] coord_nsend_all;
    delete [] tag_displs;
    delete [] coord_displs;
    delete [] tag_send_buf;
    delete [] tag_recv_buf;
    delete [] coord_send_buf;
    delete [] coord_recv_buf;
  }
}

void FixZmirror::post_integrate()
{
  if(update->ntimestep % everynum == 0) {
    int const nlocal = atom->nlocal;
    int *mask = atom->mask;
    int *tag  = atom->tag;
    double **x = atom->x;
    memset(tag_displs,0,nprocs*sizeof(int));
    memset(coord_displs,0,nprocs*sizeof(int));
    memset(tag_send_buf,0,ngroup*sizeof(tagint));
    memset(coord_send_buf,0,3*ngroup*sizeof(double));
    int nsend = 0;
    will_recv = false;
    for (int i = 0; i < nlocal; ++i) {
      if (mask[i] & groupbit) {
        tag_send_buf[nsend] = tag[i];
        coord_send_buf[3*nsend] = x[i][0];
        coord_send_buf[3*nsend+1] = x[i][1];
        coord_send_buf[3*nsend+2] = x[i][2];
        ++nsend;
      }
      if (mask[i] & jgroupbit) will_recv = true;
    }
    MPI_Allgather(&nsend,1,MPI_INT,nsend_all,nprocs,MPI_INT,world);
    coord_nsend_all[0] = 3*nsend_all[0];
    for (int i = 1; i < nprocs; ++i) {
      coord_nsend_all[i] = 3*nsend_all[i];
      tag_displs[i] = tag_displs[i-1] + nsend_all[i-1];
      coord_displs[i] = coord_displs[i-1] + 3*nsend_all[i-1];
    }
    int ncheck = tag_displs[nprocs-1]+nsend_all[nprocs-1];
    if (ncheck != ngroup) error->all(FLERR,"Insufficient atoms communicated");
    MPI_Allgatherv(tag_send_buf,nsend,MPI_LMP_TAGINT,
                   tag_recv_buf,nsend_all,tag_displs,
                   MPI_LMP_TAGINT,world);
    MPI_Allgatherv(coord_send_buf,3*nsend,MPI_DOUBLE,
                   coord_recv_buf,coord_nsend_all,coord_displs,
                   MPI_DOUBLE,world);
    double const zoffset = 2*domain->boxlo[2]+domain->zprd;
    if (will_recv) {
      memset(coord_send_buf,0,3*ngroup*sizeof(double)); // reuse for coord_sorted
      tagint tag_recv;
      int loc;
      for (int i = 0; i < ngroup; ++i) {
        tag_recv = tag_recv_buf[i];
        // if (tag_recv < send_mintag || tag_recv > send_maxtag) {
        //  error->all(FLERR,"Invalid tag sent!");
        // }
        loc = tag_recv_buf[i] - send_mintag;
        coord_send_buf[3*loc] = coord_recv_buf[3*i];
        coord_send_buf[3*loc+1] = coord_recv_buf[3*i+1];
        coord_send_buf[3*loc+2] = coord_recv_buf[3*i+2];
      }
      for (int i = 0; i < nlocal; ++i) {
        if (mask[i] & jgroupbit) {
          // if (tag[i] < recv_mintag || tag[i] > recv_maxtag) {
          //  error->all(FLERR,"Invalid tag being set!");
          // }
          loc = tag[i] - recv_mintag;
          x[i][0] = coord_send_buf[3*loc];
          x[i][1] = coord_send_buf[3*loc+1];
          x[i][2] = zoffset - coord_send_buf[3*loc+2];
        }
      }
    }
    ran_postint = true;
  }
}

void FixZmirror::end_of_step()
{
  if (!ran_postint) post_integrate();
  else ran_postint = false;
}

double FixZmirror::memory_usage()
{
  double nbytes = 2*sizeof(tagint)+6*sizeof(double);
  nbytes *= ngroup;
  nbytes += 3*nprocs*sizeof(int);
  return nbytes;
}