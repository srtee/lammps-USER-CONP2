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
#include "math.h"
#include "stdlib.h"
#include "stddef.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "force.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "compute.h"
#include "group.h"
#include "kspacemodule.h"
#include "km_ewald.h"
#include "km_ewald_split.h"
#include "fix_conp.h"
#include "pair_hybrid.h"

#include "pair.h"
#include "kspace.h"
#include "comm.h"
#include "mpi.h"
#include "math_const.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "domain.h"
#include "utils.h"
#include <iostream>
#include <string>

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429
#define ERFC_MAX  5.8        // erfc(ERFC_MAX) ~ double machine epsilon (2^-52)

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{CONSTANT,EQUAL,ATOM};
enum{CG,INV};
enum{NORMAL,FFIELD,NOSLAB};
enum{ETA,EHGO};
extern "C" {
  double ddot_(const int *N, const double *SX, const int *INCX, const double *SY, const int *INCY);
  void daxpy_(const int *N, const double *alpha, const double *X, const int *incX, double *Y, const int *incY);
  void dgetrf_(const int *M,const int *N,double *A,const int *lda,int *ipiv,int *info);
  void dgetri_(const int *N,double *A,const int *lda,const int *ipiv,double *work,const int *lwork,int *info);
}

/* ---------------------------------------------------------------------- */

FixConp::FixConp(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),coulpair(nullptr),potdiffstr(nullptr),group2(nullptr),
  pair_potential(nullptr),pair_force(nullptr),ehgo_allocated(false),
  kappa(1.), eta_i(nullptr),eta_ij(nullptr),u0_i(nullptr),fo_ij(nullptr)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  if (narg < 8) error->all(FLERR,"Illegal fix conp command (too few input parameters)");
  ilevel_respa = 0;
  maxiter = 100;
  tolerance = 0.000001;
  minimizer = 1; // default is inverse matrix solver

  zneutrflag = false;
  pppmflag = false;
  splitflag = false;
  qinitflag = false;
  lowmemflag = true;
  nullneutralflag = true;
  one_electrode_flag = false;
  potdiffstyle = CONSTANT;
  pairmode = ETA;
  
  everynum = utils::inumeric(FLERR,arg[3],false,lmp);
  
  group2 = utils::strdup(arg[4]);
  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Fix conp group ID does not exist");
  jgroupbit = group->bitmask[jgroup];
  
  eta = utils::numeric(FLERR,arg[5],false,lmp);
  
  if (strstr(arg[6],"v_") == arg[6]) {
    int n = strlen(&arg[6][2]) + 1;
    potdiffstr = new char[n];
    strcpy(potdiffstr,&arg[7][2]);
    potdiffstyle = EQUAL;
  } else {
    potdiff = utils::numeric(FLERR,arg[6],false,lmp);
  }
  
  outf = fopen(arg[7],"w");
  ff_flag = NORMAL; // turn on ff / noslab options if needed.
  a_matrix_f = 0;
  smartlist = false; // get regular neighbor lists
  matoutflag = false; // turn off matrix output unless explicitly requested
  int iarg;
  for (iarg = 8; iarg < narg; ++iarg){
    if (strcmp(arg[iarg],"ffield") == 0) {
      if (ff_flag == NOSLAB) error->all(FLERR,"Invalid fix conp command (ffield and noslab cannot both be chosen)");
      ff_flag = FFIELD;
    }
    else if (strcmp(arg[iarg],"noslab") == 0) {
      if (ff_flag == FFIELD) error->all(FLERR,"Invalid fix conp command (ffield and noslab cannot both be chosen)");
      ff_flag = NOSLAB;     
    }
    else if (strcmp(arg[iarg],"org") == 0 || strcmp(arg[iarg],"inv") == 0) {
      if (a_matrix_f != 0) error->all(FLERR,"Invalid fix conp command (A matrix file specified more than once)");
      if (strcmp(arg[iarg],"org") == 0) a_matrix_f = 1;
      else if (strcmp(arg[iarg],"inv") == 0) a_matrix_f = 2;
      ++iarg;
      if (iarg >= narg) error->all(FLERR,"Invalid fix conp command (No A matrix filename given)");
      if (me == 0) {
        a_matrix_fp = fopen(arg[iarg],"r");
        // printf("Opened file %s for reading A matrix\n",arg[iarg]);
        if (a_matrix_fp == nullptr) error->all(FLERR,"Invalid fix conp command (Cannot open A matrix file)");
      }
    }
    else if (strcmp(arg[iarg],"etypes") == 0) {
      ++iarg;
      if (iarg >= narg-1) error->all(FLERR,"Invalid fix conp command (Insufficient input entries for etypes)");
      eletypenum = utils::inumeric(FLERR,arg[iarg],false,lmp);
      eletypes = new int[eletypenum+1];
      for (int i = 0; i < eletypenum; ++i) {
        ++iarg;
        eletypes[i] = utils::inumeric(FLERR,arg[iarg],false,lmp);
      }
      eletypes[eletypenum] = -1;
      int ntypes = atom->ntypes;
      for (int i = 0; i < eletypenum; ++i) {
        if (eletypes[i] > ntypes) error->all(FLERR,"Invalid fix conp command (Invalid atom type in etypes)");
      }
      smartlist = true;
    }
    else if (strcmp(arg[iarg],"zneutr") == 0) zneutrflag = true;
    else if (strcmp(arg[iarg],"matout") == 0) matoutflag = true;
    else if (strcmp(arg[iarg],"pppm") == 0) pppmflag = true;
    else if (strcmp(arg[iarg],"split") == 0) splitflag = true;
    else if (strcmp(arg[iarg],"qinit") == 0) qinitflag = true;
    else if (strcmp(arg[iarg],"himem") == 0) lowmemflag = false;
    else if (strcmp(arg[iarg],"nonneutral") == 0) nullneutralflag = false;
    else if (strcmp(arg[iarg],"ehgo") == 0) pairmode = EHGO;
    else {
      std::string errmsg = "Invalid fix conp commmand (unknown option: ";
      errmsg += arg[iarg];
      errmsg += ")";
      error->all(FLERR,errmsg);
    }
  }
  scalar_flag = 1;
  extscalar = 0;
  global_freq = 1;
  initflag = false;
  runstage = 0; //after operation
                //0:init; 1: a_cal; 2: first sin/cos cal; 3: inv only, aaa inverse
  elenum = elenum_old = 0;
  aaa_all = nullptr;
  bbb_all = nullptr;
  eleallq = elesetq = eleinitq = nullptr;
  tag2eleall = eleall2tag = ele2tag = nullptr;
  elecheck_eleall = nullptr;
  eleall2ele = ele2eleall = elebuf2eleall = nullptr;
  bbb = bbuf = nullptr;
  newtonbuf = nullptr;
  Btime = cgtime = Ctime = Ktime = 0;
  alist = blist = list = nullptr;
  elenum_list = displs = nullptr;
  totsetq = 0;
  gotsetq = 0;  //=1 after getting setq vector
  newton = !!(force->newton_pair);
  preforceflag = postforceflag = false;

  //kspmod_constructor();
}

/* ---------------------------------------------------------------------- */

FixConp::~FixConp()
{
  if (!pppmflag && kspmod!=nullptr) delete kspmod;
  fclose(outf);
  memory->destroy(bbb);
  memory->destroy(ele2tag);
  memory->destroy(ele2eleall);
  memory->destroy(eleall2tag);
  memory->destroy(elecheck_eleall);
  memory->destroy(eleall2ele);
  memory->destroy(aaa_all);
  memory->destroy(bbb_all);
  memory->destroy(eleallq);
  memory->destroy(elebuf2eleall);
  memory->destroy(bbuf);
  memory->destroy(elesetq);
  if (qinitflag) memory->destroy(eleinitq);
  if (newton) memory->destroy(newtonbuf);
  if (ehgo_allocated) ehgo_deallocate();
  delete [] tag2eleall;
  delete [] potdiffstr;
  delete [] displs;
  delete [] elenum_list;
  delete [] group2;
}

/* ---------------------------------------------------------------------- */

int FixConp::setmask()
{
  int mask = 0;
  mask |= POST_NEIGHBOR;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixConp::init()
{
  MPI_Comm_rank(world,&me);

  // assign coulpair to either the existing pair style if it matches 'coul'
  // or, if hybrid, the pair style matching 'coul'
  // and if neither are true then something has gone horribly wrong!
  coulpair = nullptr;
  coulpair = (Pair *) force->pair_match("coul",0);
  if (coulpair == nullptr) {
    // return 1st hybrid substyle matching coul (inexactly)
    coulpair = (Pair *) force->pair_match("coul",0,1);
    }
  if (coulpair == nullptr) error->all(FLERR,"Fix conp couldn't detect a Coulombic pair style");
  
  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
  
  // check variables
  if (potdiffstr) {
    potdiffvar = input->variable->find(potdiffstr);
    if (potdiffvar < 0)
      error->all(FLERR,"Fix conp potential difference variable does not exist");
    if (!input->variable->equalstyle(potdiffvar))
      error->all(FLERR,"Fix conp potential difference variable is invalid style");
  }
 
  intelflag = false;
  int ifix = modify->find_fix("package_intel");
  if (ifix >= 0) intelflag = true;

  if (alist == nullptr || blist == nullptr) {
    // request neighbor list 
    // if not smart list half, newton off
    // else do request_smartlist()
    if (smartlist) request_smartlist();
    // TO-DO: check failure conditions in request_smartlist
    // and flip smartlist bool to trigger this loop as backup
    if (!smartlist) {
      int irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->pair = 0;
      neighbor->requests[irequest]->fix  = 1;
      // neighbor->requests[irequest]->newton = 2;
      if (intelflag) neighbor->requests[irequest]->intel = 1;
    }
  }
  // initflag = true;


  if (groupbit == jgroupbit) one_electrode_flag = true;
}

/* ---------------------------------------------------------------------- */

void FixConp::request_smartlist() {
  int itype,jtype,ieletype;
  int ntypes = atom->ntypes;
  int *iskip_a = new int[ntypes+1];
  int *iskip_b = new int[ntypes+1];
  int **ijskip_a;
  memory->create(ijskip_a,ntypes+1,ntypes+1,"fixconp:ijskip_a");
  int **ijskip_b;
  memory->create(ijskip_b,ntypes+1,ntypes+1,"fixconp:ijskip_b");
  // at this point eletypes has eletypenum types
  for (itype = 0; itype <= ntypes; ++itype) { // yes, itype is 1-based numbering
  // starting from 0 because being a bit anal
    iskip_a[itype] = 1; // alist skips all except eletype by default
    iskip_b[itype] = 0;
    for (jtype = 0; jtype <= ntypes; ++jtype) {
      ijskip_a[itype][jtype] = 1;
    }
  }
  for (ieletype = 0; ieletype < eletypenum; ++ieletype) {
    iskip_a[eletypes[ieletype]] = 0;
    ijskip_a[eletypes[ieletype]][eletypes[ieletype]] = 0;
  } // now, iskip_a[itype] == 0 (1) if eletype (soltype)
  // set ijskip_b[itype][jtype] == 0 if (i is eletype XOR j is eletype)
  for (itype = 0; itype <= ntypes; ++itype) {
    for (jtype = 0; jtype <= ntypes; ++jtype) {
      bool ele_and_sol = (!!(iskip_a[itype]) ^ !!(iskip_a[jtype]));
      ijskip_b[itype][jtype] = (ele_and_sol) ? 0 : 1;
    }
  }
  if (a_matrix_f == 0) {
    arequest = neighbor->request(this,instance_me);
    NeighRequest *aRq = neighbor->requests[arequest];
    aRq->pair = 0;
    aRq->fix  = 1;
    aRq->half = 1;
    aRq->full = 0;
    // aRq->newton = 2;
    aRq->occasional = 1;
    aRq->skip = 1;
    aRq->iskip = iskip_a;
    aRq->ijskip = ijskip_a;
    if (intelflag) aRq->intel = 1;
  }

  brequest = neighbor->request(this,instance_me);
  NeighRequest *bRq = neighbor->requests[brequest];
  bRq->pair = 0;
  bRq->fix  = 1;
  bRq->half = 1;
  bRq->full = 0;
  //bRq->newton = 2;
  bRq->skip = 1;
  bRq->iskip = iskip_b;
  bRq->ijskip = ijskip_b;
  if (intelflag) bRq->intel = 1;

  delete [] eletypes;
}

/* ---------------------------------------------------------------------- */

void FixConp::init_list(int /* id */, NeighList *ptr) {
  if (smartlist) {
    if (ptr->index == arequest) {
      alist = ptr;
    }
    else if (ptr->index == brequest) {
      blist = ptr;
    }
  }
  else {
    alist = ptr;
    blist = ptr;
  }
}

/* ---------------------------------------------------------------------- */

void FixConp::setup_post_neighbor(){
  linalg_init();
  post_neighbor();
}

void FixConp::setup_pre_force(int vflag){
  force->kspace->setup();
  linalg_setup();
  pre_force(vflag);
}

void FixConp::linalg_init()
{
  // To-do: encapsulate runstage == 0 into a discrete member function?
  // Especially because we should check that electrode atoms obey the
  // smartlist listings, and if not, get the list pointer from coulpair,
  // and if _that_ fails, or if coulpair has newton on, we should bail
  // not too late to process that here because we haven't done a_cal yet
  if (runstage == 0) {
    if (pppmflag) {
      kspmod = dynamic_cast<KSpaceModule *>(force->kspace);
      if (kspmod == nullptr)
        error->all(FLERR,"Fix conp couldn't detect a pppm/conp kspace style (which is required with the pppm flag)");
    }
    else
      if (splitflag) kspmod = new KSpaceModuleEwaldSplit(lmp);
      else kspmod = new KSpaceModuleEwald(lmp);
    kspmod->register_fix(this);
    kspmod->conp_setup(lowmemflag);
    g_ewald = force->kspace->g_ewald;
    evscale = force->qe2f/force->qqr2e;
    tagint maxtag = 0;
    for (int i = 0; i < atom->nlocal; ++i) maxtag = MAX(atom->tag[i],maxtag);
    MPI_Allreduce(&maxtag,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
    tag2eleall = new int[maxtag_all+1];
    int nprocs = comm->nprocs;
    elenum_list = new int[nprocs];
    displs = new int[nprocs];
    elenum = 0;
    elenum_all = 0;
    elytenum = 0;
  }
}

void FixConp::linalg_setup()
{
  if (runstage == 0) {
    if (pairmode == EHGO) {
      ehgo_setup_tables();
      pair_potential = &FixConp::ehgo_potential;
      pair_force = &FixConp::ehgo_force;
    }
    // if, not elseif, so ehgo_setup_tables can set pairmode to ETA if no parameters were given
    if (pairmode == ETA) pair_potential = &FixConp::eta_potential_A;
    if (a_matrix_f == 0) {
      if (me == 0) printf("Fix conp is now calculating A matrix ... ");
      a_cal();
      if (me == 0) printf(" ... done!\n");
    } else {
      if (me == 0) printf("Fix conp is now reading in A matrix type %d ... \n", a_matrix_f);
      a_read();
    }
    runstage = 1;

    if (pairmode == ETA) {
      pair_potential = &FixConp::eta_potential;
      pair_force = &FixConp::eta_force;
    }

    gotsetq = 0;
    b_setq_cal();
    equation_solve();
    get_setq();
    gotsetq = 1;
    dyn_setup(); // additional setup for dynamic versions
  }
}

/* ----------------------------------------------------------------------*/

void FixConp::post_neighbor()
{
  bigint natoms = atom->natoms;
  int elytenum_old = elytenum;
  int elenum_old = elenum;
  int elenum_all_old = elenum_all;
  int* tag = atom->tag;
  int const nlocal = atom->nlocal;
  int const nprocs = comm->nprocs;
  int i,elealli,tagi;
  elenum = 0;
  elenum_all = 0;
  for (i = 0; i < nlocal; ++i) {
    if (electrode_check(i)) {
      ++elenum;
    }
  }
  elytenum = nlocal - elenum;
  if (elenum > elenum_old) {
    memory->grow(ele2tag,elenum,"fixconp:ele2tag");
    memory->grow(ele2eleall,elenum,"fixconp:ele2eleall");
    memory->grow(bbb,elenum,"fixconp:bbb");
  }
  int j = 0;
  MPI_Allgather(&elenum,1,MPI_INT,elenum_list,1,MPI_INT,world);
  for (i = 0; i < nlocal; ++i) {
    if (electrode_check(i)) {
      ele2tag[j] = tag[i];
      j++;
    }
  }
  int displssum = 0;
  for (i = 0; i < nprocs; ++i) {
    displs[i] = displssum;
    displssum += elenum_list[i];
  }
  elenum_all = displssum;
  int const elenum_all_c = elenum_all;

  if (elenum_all > elenum_all_old) {
    memory->grow(eleall2tag,elenum_all,"fixconp:eleall2tag");
    memory->grow(elecheck_eleall,elenum_all,"fixconp:elecheck_eleall");
    memory->grow(eleall2ele,elenum_all+1,"fixconp:eleall2ele");
    memory->grow(aaa_all,elenum_all*elenum_all,"fixconp:aaa_all");
    memory->grow(bbb_all,elenum_all,"fixconp:bbb_all");
    memory->grow(eleallq,elenum_all,"fixconp:eleallq");
    memory->grow(elebuf2eleall,elenum_all,"fixconp:elebuf2eleall");
    memory->grow(bbuf,elenum_all,"fixconp:bbuf");
    memory->grow(elesetq,elenum_all,"fixconp:elesetq");
    if (newton) memory->grow(newtonbuf,elenum_all,"fixconp:newtonbuf");
    if (qinitflag) memory->grow(eleinitq,elenum_all,"fixconp:eleinitq");
    MPI_Barrier(world); // otherwise next MPI_Allgatherv can race??
    for (i = 0; i < elenum_all; i++) elecheck_eleall[i] = 0;
    for (i = 0; i < maxtag_all+1; i++) tag2eleall[i] = elenum_all;
    eleall2ele[elenum_all] = -1; // not a typo
    MPI_Allgatherv(ele2tag,elenum,MPI_INT,eleall2tag,elenum_list,displs,MPI_INT,world);
    for (i = 0; i < elenum_all_c; ++i) tag2eleall[eleall2tag[i]] = i;
  }
  j = 0;
  for (i = 0; i < elenum_all_c; ++i) eleall2ele[i] = -1;
  for (i = 0; i < nlocal; ++i) {
    if (electrode_check(i)) {
      ele2eleall[j] = tag2eleall[tag[i]];
      eleall2ele[ele2eleall[j]] = j;
      ++j;
    }
  }
  MPI_Allgatherv(ele2eleall,elenum,MPI_INT,elebuf2eleall,elenum_list,displs,MPI_INT,world);
  bool do_elyte_alloc = (elytenum > elytenum_old);
  bool do_ele_alloc = (elenum_all > elenum_all_old);
  kspmod->conp_post_neighbor(do_elyte_alloc,do_ele_alloc);
}

/* ---------------------------------------------------------------------- */

void FixConp::pre_force(int /* vflag */)
{
  kspmod->conp_pre_force();
  if(update->ntimestep % everynum == 0) {
    preforceflag = true;
    if (strstr(update->integrate_style,"verlet")) { //not respa
      Btime1 = MPI_Wtime();
      b_cal();
      Btime2 = MPI_Wtime();
      Btime += Btime2-Btime1;
      if (update->laststep == update->ntimestep) {
        double Btime_all;
        MPI_Reduce(&Btime,&Btime_all,1,MPI_DOUBLE,MPI_SUM,0,world);
        double Ctime_all;
        MPI_Reduce(&Ctime,&Ctime_all,1,MPI_DOUBLE,MPI_SUM,0,world);
        double Ktime_all;
        MPI_Reduce(&Ktime,&Ktime_all,1,MPI_DOUBLE,MPI_SUM,0,world);
        if (me == 0) {
          Btime = Btime_all/comm->nprocs;
          Ctime = Ctime_all/comm->nprocs;
          Ktime = Ktime_all/comm->nprocs;
          fprintf(outf,"B vector calculation time = %g\n",Btime);
          fprintf(outf,"Coulomb calculation time = %g\n",Ctime);
          fprintf(outf,"Kspace calculation time = %g\n",Ktime);
        }
      }
    }
    equation_solve();
    update_charge();
  }
}

/* ---------------------------------------------------------------------- */

void FixConp::post_force(int vflag) {
  postforceflag = true;
  force_cal(vflag);
}

/* ---------------------------------------------------------------------- */

void FixConp::end_of_step()
{
  if ( !postforceflag ) post_force(0);
  postforceflag = false;
}

/* ---------------------------------------------------------------------- */

double FixConp::compute_scalar()
{
  return scalar_output;
}

/* ---------------------------------------------------------------------- */

int FixConp::electrode_check(int atomid)
{
  int *mask = atom->mask;
  if (mask[atomid] & groupbit) return 1;
  else if (mask[atomid] & jgroupbit) return -1;
  else return 0;
}

/* ----------------------------------------------------------------------*/

void FixConp::b_setq_cal()
{
  int i,iall,iloc,eci;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double zprd = domain->zprd;
  double zprd_half = domain->zprd_half;
  double zhalf = zprd_half + domain->boxlo[2];
  int const elenum_c = elenum;
  for (iloc = 0; iloc < elenum_c; ++iloc) {
    iall = ele2eleall[iloc];
    i = atom->map(ele2tag[iloc]);
    eci = electrode_check(i);
    if (ff_flag == FFIELD) {
      if (eci == 1 && x[i][2] < zhalf) {
        bbb[iloc] = -(x[i][2]/zprd + 1)*evscale;
      }
      else bbb[iloc] = -x[i][2]*evscale/zprd;
    }
    else bbb[iloc] = -0.5*evscale*eci;
    elecheck_eleall[iall] = eci;
  }
  MPI_Allreduce(MPI_IN_PLACE,elecheck_eleall,elenum_all,MPI_INT,MPI_SUM,world);
  // this really could be a b_comm but I haven't written the method for an int array
  b_comm(bbb,bbb_all);
  if (runstage == 1) runstage = 2;
}

/* ----------------------------------------------------------------------*/

void FixConp::b_comm(double* bsend, double* brecv)
{
  MPI_Allgatherv(bsend,elenum,MPI_DOUBLE,bbuf,elenum_list,displs,MPI_DOUBLE,world);
  int iall;
  for (iall = 0; iall < elenum_all; ++iall) {
    brecv[elebuf2eleall[iall]] = bbuf[iall];
  }
}

/* ----------------------------------------------------------------------*/

void FixConp::b_bcast(int nproc, int ncount, int* eleall_list, double* bcastbuf)
{
  int const elenum_n=elenum_list[nproc];
  int* eleall_list_buf;
  if (me == nproc) eleall_list_buf = ele2eleall;
  else eleall_list_buf = eleall_list;
  MPI_Bcast(eleall_list_buf,elenum_n,MPI_INT,nproc,world);
  MPI_Bcast(bcastbuf,elenum_n*ncount,MPI_DOUBLE,nproc,world);
}

/* ----------------------------------------------------------------------*/

void FixConp::b_comm_int(int* bsend, int* brecv)
{
  int* bbuf_int = new int[elenum_all];
  MPI_Allgatherv(bsend,elenum,MPI_INT,bbuf_int,elenum_list,displs,MPI_INT,world);
  int iall;
  for (iall = 0; iall < elenum_all; ++iall) {
    brecv[elebuf2eleall[iall]] = bbuf_int[iall];
  }
  delete [] bbuf_int;
}

/* ----------------------------------------------------------------------*/

void FixConp::b_cal() {
  bool do_bp=true;
  update_bk(do_bp,bbb_all);
}

/* ----------------------------------------------------------------------*/

void FixConp::update_bk(bool coulyes, double* bbb_all)
{
  Ktime1 = MPI_Wtime();
  kspmod->b_cal(bbb);
  Ktime2 = MPI_Wtime();
  Ktime += Ktime2-Ktime1;

  if (coulyes) {
    blist_coul_cal(bbb);
  }
  b_comm(bbb,bbb_all);
}

/*----------------------------------------------------------------------- */
void FixConp::equation_solve()
{
//solve equations
  if (minimizer == 0) {
    cgtime1 = MPI_Wtime();
    cg();
    cgtime2 = MPI_Wtime();
    cgtime += cgtime2-cgtime1;
    if (update->laststep == update->ntimestep) {
      double cgtime_all;
      MPI_Reduce(&cgtime,&cgtime_all,1,MPI_DOUBLE,MPI_SUM,0,world);
      if (me == 0) {
        cgtime = cgtime_all/comm->nprocs;
        if (screen) fprintf(screen,"conjugate gradient solver time = %g\n",cgtime);
        if (logfile) fprintf(logfile,"conjugate gradient solver time = %g\n",cgtime);
      }
    }
  } else if (minimizer == 1) {
    inv();
  }
}

/*----------------------------------------------------------------------- */
void FixConp::a_read()
{
  int i = 0;
  int idx1d;
  if (me == 0) {
    int maxchar = 21*elenum_all+1;
    char line[maxchar];
    char *word;
    while(fgets(line,maxchar,a_matrix_fp) != NULL) {
      word = strtok(line," \t");
      while(word != NULL) {
        if (i < elenum_all) {
          eleall2tag[i] = atoi(word);
        } else {
          idx1d = i-elenum_all;
          if (idx1d >= elenum_all*elenum_all) {
            error->all(FLERR,"Too many entries in A matrix file");
          }
          aaa_all[idx1d] = atof(word);
        }
        word = strtok(NULL," ");
        i++;
      }
    }
    fclose(a_matrix_fp);
    if (idx1d != elenum_all*elenum_all-1) {
      error->all(FLERR,"Too few entries in A matrix file");
    }
  }
  MPI_Bcast(eleall2tag,elenum_all,MPI_INT,0,world);
  MPI_Bcast(aaa_all,elenum_all*elenum_all,MPI_DOUBLE,0,world);

  int tagi;
  int const elenum_all_c = elenum_all;
  for (i = 0; i < elenum_all_c; i++) {
    eleall2ele[i] = -1;
    tagi = eleall2tag[i];
    tag2eleall[tagi] = i;
  }
  int const nlocal = atom->nlocal;
  int *tag = atom->tag;
  int j = 0;
  for (i = 0; i < nlocal; ++i) {
    if (electrode_check(i)) {
      ele2tag[j] = tag[i];
      ele2eleall[j] = tag2eleall[tag[i]];
      eleall2ele[ele2eleall[j]] = j;
      ++j;
    }
  }
  MPI_Allgatherv(ele2eleall,elenum,MPI_INT,elebuf2eleall,elenum_list,displs,MPI_INT,world);
  kspmod->a_read();
}

/*----------------------------------------------------------------------- */

void FixConp::a_cal()
{
  double t1,t2;
  int i,j,k,iele;
  int const elenum_all_c = elenum_all;
  int nprocs = comm->nprocs;
  double CON_s2overPIS = sqrt(2.0)/MY_PIS;
  t1 = MPI_Wtime();
  Ktime1 = MPI_Wtime();
  if (me == 0) {
    fprintf(outf,"A matrix calculating ...\n");
  }

  // gather tag,x and q
  int const elenum_c = elenum;
  double *aaa = new double[elenum*elenum_all];
  memset(aaa,0,elenum*elenum_all*sizeof(double));
  kspmod->a_cal(aaa);
  
  if (pairmode == ETA) {
    for (i = 0; i < elenum; ++i) {
      int idx1d = i*elenum_all_c + ele2eleall[i];
      aaa[idx1d] += CON_s2overPIS*eta;
    }
  }

  else if (pairmode == EHGO) {
    int* atomtype = atom->type;
    for (i = 0; i < elenum; ++i) {
      int idx1d = i*elenum_all_c + ele2eleall[i];
      int itype = atom->map(ele2tag[i]);
      aaa[idx1d] += u0_i[itype];
    }
  }

  //if (smartlist) alist_coul_cal(aaa);
  //else coul_cal(2,aaa);
  alist_coul_cal(aaa);

  int elenum_list3[nprocs];
  int displs3[nprocs];
  for (i = 0; i < nprocs; i++) {
    elenum_list3[i] = elenum_list[i]*elenum_all;
    displs3[i] = displs[i]*elenum_all;
  }
  MPI_Allgatherv(aaa,elenum*elenum_all,MPI_DOUBLE,aaa_all,elenum_list3,displs3,MPI_DOUBLE,world);
  delete [] aaa;

  // #pragma ivdep
  for (i = 1; i < elenum_all_c; ++i) {
    for (j = 0; j < i; ++j) {
      aaa_all[i*elenum_all_c+j] += aaa_all[j*elenum_all_c+i];
      aaa_all[j*elenum_all_c+i] =  aaa_all[i*elenum_all_c+j];
    }
  }

  if (matoutflag && me == 0) {
    int idx1d;
    FILE *outa = fopen("amatrix","w");
    fprintf(outa," ");
    for (i = 0; i < elenum_all_c; ++i) fprintf(outa,"%20d",eleall2tag[i]);
    fprintf(outa,"\n");
    idx1d = 0;
    for (i = 0; i < elenum_all_c; ++i) {
      fprintf(outa," ");
      for (j = 0; j < elenum_all_c; ++j) {
        fprintf (outa,"%20.12f",aaa_all[idx1d]);
        idx1d++;
      }
      fprintf(outa,"\n");
    }
    fclose(outa);
  }

  t2 = MPI_Wtime();
  double tsum = t2 - t1;
  double tsum_all;
  MPI_Allreduce(&tsum,&tsum_all,1,MPI_DOUBLE,MPI_SUM,world);
  if (me == 0) {
    tsum = tsum_all/nprocs;
    fprintf(outf,"A matrix calculation time  = %g\n",tsum);
  }
  Ktime2 = MPI_Wtime();
  Ktime += Ktime2-Ktime1;
}

/* ---------------------------------------------------------------------- */
void FixConp::cg()
{
  int iter,i,j,idx1d;
  double alpha,beta,ptap,lresnorm,netr,tmp;
  double lgamma,gamma,avenetr;  /* lX = new; X = old */
  double res[elenum_all],p[elenum_all],ap[elenum_all];
  for (i = 0; i < elenum_all; i++) eleallq[i] = 0.0;
  lresnorm = 0.0;
  netr = 0.0; /* = -Q_target */
  for (i = 0; i < elenum_all; ++i) {
    res[i] = bbb_all[i];
    for (j = 0; j < elenum_all; ++j) {
      idx1d = i*elenum_all+j;
      tmp = aaa_all[idx1d]*eleallq[j];
      res[i] -= tmp;
    }
    netr += res[i];
    lresnorm += res[i]*res[i];
  }
  avenetr = netr/elenum_all;
  for (i = 0; i < elenum_all; i++) p[i] = res[i]-avenetr;
  lresnorm -= netr*avenetr;
  lgamma = lresnorm;
  for (iter = 1; iter < maxiter; ++iter) {
    for (i = 0; i < elenum_all; ++i) {
      ap[i] = 0.0;
      for (j = 0; j < elenum_all; ++j) {
        idx1d = i*elenum_all+j;
        ap[i] += aaa_all[idx1d]*p[j];
      }
    }
    ptap = 0.0;
    for (i = 0; i < elenum_all; ++i) {
      ptap += p[i]*ap[i];
    }
    alpha = lresnorm/ptap;
    gamma = lgamma;
    lgamma = 0.0;
    netr = 0.0;
    for (i = 0; i <elenum_all; ++i) {
      eleallq[i] = eleallq[i]+alpha*p[i];
      res[i] = res[i]-alpha*ap[i];
      lgamma += res[i]*res[i];
      netr += res[i];
    }
    avenetr = netr/elenum_all;
    lgamma -= netr*avenetr;
    beta = lgamma/gamma;
    lresnorm = 0.0;
    for (i = 0; i < elenum_all; i++) {
      p[i] = beta*p[i]+res[i]-avenetr;
      lresnorm += res[i]*p[i];
    }
    if (lresnorm/elenum_all < tolerance) {
      netr = 0.0;
      for (i = 0; i < elenum_all; ++i) netr += eleallq[i];
      if (me == 0) {
        fprintf(outf,"***** Converged at iteration %d. res = %g netcharge = %g\n",
            iter,lresnorm,netr);
      }
      break;
    }
    if (me == 0) {
      fprintf(outf,"Iteration %d: res = %g\n",iter,lresnorm);
    }
  }
}
/* ---------------------------------------------------------------------- */
void FixConp::inv()
{
  int i,j,k,idx1d;
  if (runstage == 2 && a_matrix_f < 2) {
    int m = elenum_all;
    int n = elenum_all;
    int lda = elenum_all;
    int *ipiv = new int[elenum_all+1];
    int lwork = elenum_all*elenum_all;
    double *work = new double[lwork];
    int info;
    int infosum;
    int const elenum_c = elenum;
    int const elenum_all_c = elenum_all;

    dgetrf_(&m,&n,aaa_all,&lda,ipiv,&info);
    infosum = info;
    dgetri_(&n,aaa_all,&lda,ipiv,work,&lwork,&info);
    infosum += info;
    delete [] ipiv;
    ipiv = NULL;
    delete [] work;
    work = NULL;

    if (infosum != 0) error->all(FLERR,"Inversion failed!");

    if (!one_electrode_flag) inv_project();

    if (matoutflag && me == 0) {
      FILE *outinva = fopen("inv_a_matrix","w");
      for (i = 0; i < elenum_all_c; i++) {
        if(i == 0) fprintf (outinva," ");
        fprintf (outinva,"%20d",eleall2tag[i]);
      }
      fprintf (outinva,"\n");
      for (k = 0; k < elenum_all_c*elenum_all_c; k++) {
        if (k%elenum_all != 0) {
          fprintf (outinva," ");
        }
        fprintf(outinva,"%20.10f",aaa_all[k]);
        if ((k+1)%elenum_all == 0) {
          fprintf(outinva,"\n");
        }
      }
      fclose(outinva);
    }
  }
  if (runstage == 2) runstage = 3;
}

void FixConp::inv_project()
{
  // here we project aaa_all onto
  // the null space of e
  if (nullneutralflag) {
    double *ainve = new double[elenum_all];
    double ainvtmp;
    double totinve = 0;
    int const elenum_c = elenum;
    int const elenum_all_c = elenum_all;
    int i,j;
    int idx1d = 0;

    for (i = 0; i < elenum_all_c; i++) {
      ainvtmp = 0;
      for (j = 0; j < elenum_all_c; j++) {
        ainvtmp += aaa_all[idx1d];
        idx1d++;
      }
      totinve += ainvtmp;
      ainve[i] = ainvtmp;
    }

    if (totinve * totinve > 1e-8) {
      idx1d = 0;
      for (i = 0; i < elenum_all_c; i++) {
        for (j = 0; j < elenum_all_c; j++) {
          aaa_all[idx1d] -= ainve[i] * ainve[j] / totinve;
          idx1d++;
        }
      }
    }

    // here we project aaa_all onto
    // the null space of e_pos
    // if zneutr has been called (i.e. we want each half of the unit cell
    // to be neutral, not just the overall electrodes, in noslab)

    if (zneutrflag) {
      int iele;
      double zprd_half = domain->zprd_half;
      double zhalf = zprd_half + domain->boxlo[2];
      double *elez = new double[elenum];
      double *eleallz = new double[elenum_all];
      int nlocal = atom->nlocal;
      double **x = atom->x;
      for (iele = 0; iele < elenum_c; ++iele) { elez[iele] = x[atom->map(ele2tag[iele])][2]; }
      b_comm(elez, eleallz);

      bool *zele_is_pos = new bool[elenum_all];
      for (iele = 0; iele < elenum_all_c; ++iele) { zele_is_pos[iele] = (eleallz[iele] > zhalf); }
      idx1d = 0;
      totinve = 0;
      for (i = 0; i < elenum_all_c; i++) {
        ainvtmp = 0;
        for (j = 0; j < elenum_all_c; j++) {
          if (zele_is_pos[j]) ainvtmp += aaa_all[idx1d];
          idx1d++;
        }
        ainve[i] = ainvtmp;
        if (zele_is_pos[i]) totinve += ainvtmp;
      }

      if (totinve * totinve > 1e-8) {
        idx1d = 0;
        for (i = 0; i < elenum_all_c; i++) {
          for (j = 0; j < elenum_all_c; j++) {
            aaa_all[idx1d] -= ainve[i] * ainve[j] / totinve;
            idx1d++;
          }
        }
      }
      delete[] zele_is_pos;
      delete[] elez;
      delete[] eleallz;
    }
    delete[] ainve;
  }
}

/* ---------------------------------------------------------------------- */

void FixConp::get_setq()
{
  int iall,jall,iloc,i,j,idx1d;
  int elealli,tagi;
  double eleallq_i,bbbtmp;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nall = nlocal+atom->nghost;
  double netcharge_left_local = 0;
  int const elenum_all_c = elenum_all;
  int const elenum_c = elenum;

  if (minimizer == 0) { // cg solver used
    for (iall = 0; iall < elenum_all_c; ++iall) {
      elesetq[iall] = eleallq[iall];
    }
  } else if (minimizer == 1) { // inv solver used
    int one = 1;
    idx1d = 0;
    for (iloc = 0; iloc < elenum_c; ++iloc) {
      iall = ele2eleall[iloc];
      idx1d = iall*elenum_all;
      bbbtmp = ddot_(&elenum_all,&aaa_all[idx1d],&one,bbb_all,&one);
      bbb[iloc] = bbbtmp;
    }
    b_comm(bbb,elesetq);
  }
  totsetq = 0;
  for (iloc = 0; iloc < elenum_c; ++iloc) {
    iall = ele2eleall[iloc];
    if (elecheck_eleall[iall] == 1) {
      totsetq += elesetq[iall];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&totsetq,1,MPI_DOUBLE,MPI_SUM,world);

  if (qinitflag) {
    double* q = atom->q;
    for (iloc = 0; iloc < elenum_c; ++iloc) {
      i = atom->map(ele2tag[iloc]);
      bbb[iloc] = q[i];
    }
    b_comm(bbb,eleinitq);
  }
  if (one_electrode_flag) inv_project();
}

/* ---------------------------------------------------------------------- */

void FixConp::update_charge()
{
  int i,j,idx1d,iall,jall,iloc;
  int elealli,tagi;
  double eleallq_i;
  int *tag = atom->tag;
  int const nlocal = atom->nlocal;
  int const nall = nlocal+atom->nghost;
  double netcharge_left = 0;
  double *q = atom->q;    
  int const elenum_c = elenum;
  int const elenum_all_c = elenum_all;
  if (minimizer == 1) {
    idx1d = 0;
    int one = 1;
    for (iloc = 0; iloc < elenum_c; ++iloc) {
      iall = ele2eleall[iloc];
      idx1d = iall*elenum_all;
      bbb[iloc] = ddot_(&elenum_all,&aaa_all[idx1d],&one,bbb_all,&one);
    }
    b_comm(bbb,eleallq);
  } // if minimizer == 0 then we already have eleallq ready;

  if (potdiffstyle == EQUAL) potdiff = input->variable->compute_equal(potdiffvar);
  //  now qL and qR are left and right *voltages*
  //  evscale was included in the precalculation of eleallq

  //  update charges including additional charge needed
  //  this fragment is the only difference from fix_conq
  for (iall = 0; iall < elenum_all_c; ++iall) {
    if (elecheck_eleall[iall] == 1) netcharge_left += eleallq[iall];
    i = atom->map(eleall2tag[iall]);
    if (i != -1) {
      q[i] = eleallq[iall] + potdiff*elesetq[iall];
      if (qinitflag) q[i] += eleinitq[iall];
    }
  } // we need to loop like this to correctly charge ghost atoms

  scalar_output = potdiff*totsetq+netcharge_left;
  kspmod->update_charge();
}
/* ---------------------------------------------------------------------- */
void FixConp::force_cal(int vflag)
{
  int i;
  if (force->kspace->energy) {
    if (pairmode == ETA) {
      double eleqsqsum = 0.0;
      int nlocal = atom->nlocal;
      for (i = 0; i < nlocal; i++) {
        if (electrode_check(i)) {
          eleqsqsum += atom->q[i]*atom->q[i];
        }
      }
      double tmp;
      MPI_Allreduce(&eleqsqsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
      eleqsqsum = tmp;
      double scale = 1.0;
      double qscale = force->qqrd2e*scale;
      force->kspace->energy += qscale*eta*eleqsqsum/(sqrt(2)*MY_PIS);
    }
    else if (pairmode == EHGO) {
      double u0qsqsum = 0.0;
      int nlocal = atom->nlocal;
      int* atomtype = atom->type;
      double* q = atom->q;
      for (i = 0; i < nlocal; i++) {
        if (electrode_check(i)) {
          u0qsqsum += u0_i[atomtype[i]]*q[i]*q[i];
        }
      }
      double tmp;
      MPI_Allreduce(&u0qsqsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
      u0qsqsum = tmp;
      double scale = 1.0;
      double qscale = force->qqrd2e*scale;
      force->kspace->energy += qscale*u0qsqsum;
    }
  }
  blist_coul_cal_post_force();
}
/* ---------------------------------------------------------------------- */
/*
Electrode_check(i) is the standard way to check electrode membership
and all other ways must be checked and double checked and triple checked.
Many coder-hours and rubber duckies died to give us this information.
/*
/* ---------------------------------------------------------------------- */
void FixConp::alist_coul_cal(double* m)
{
  Ctime1 = MPI_Wtime();
  if (alist->occasional) {
    neighbor->build(0);
    neighbor->build_one(alist,1);
  }
  //coulcalflag = 2: a_cal; 1: b_cal; 0: force_cal
  int i,j,k,ii,jj,jnum,itype,jtype,idx1d;
  int elei,elej,elealli,eleallj;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double r,r2inv,rsq,grij,etarij2,expm2,t,erfc,dudq;
  double forcecoul,ecoul,prefactor,fpair;
  bool ecib,ecjb;
  int inum = alist->inum;
  int nlocal = atom->nlocal;
  int *atomtype = atom->type;
  int *tag = atom->tag;
  int *ilist = alist->ilist;
  int *jlist;
  int *numneigh = alist->numneigh;
  int **firstneigh = alist->firstneigh;

  double qqrd2e = force->qqrd2e;
  double **cutsq = coulpair->cutsq;
  int itmp;
  double *p_cut_coul = (double *) coulpair->extract("cut_coul",itmp);
  double cut_coulsq = (*p_cut_coul)*(*p_cut_coul);
  double cut_erfc = ERFC_MAX*ERFC_MAX/(g_ewald*g_ewald);
  if (cut_coulsq > cut_erfc) cut_coulsq = cut_erfc;
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = atomtype[i];
    ecib = !!(electrode_check(i));
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      ecjb = !!(electrode_check(j));
      if (ecib && ecjb) {
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = atomtype[j];
        if (rsq < cutsq[itype][jtype]) {
          if (rsq < cut_coulsq) {
            dudq = erfcr_sqrt(g_ewald*g_ewald*rsq)*g_ewald;
            dudq += (this->*pair_potential)(rsq, itype, jtype);
            elealli = tag2eleall[tag[i]];
            eleallj = tag2eleall[tag[j]];
            elei = eleall2ele[elealli];
            if (j < nlocal || !(!newton && eleallj > elealli)) {
              idx1d = elei*elenum_all + eleallj;
              m[idx1d] += dudq;
            }
          }
        }
      }
    }
  }
  Ctime2 = MPI_Wtime();
  Ctime += Ctime2-Ctime1;
}
/* ---------------------------------------------------------------------- */
void FixConp::blist_coul_cal(double* m)
{
  Ctime1 = MPI_Wtime();
  int i,j,k,ii,jj,jnum,itype,jtype,idx1d;
  int checksum,elei,elej,elealli,eleallj,tagi;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz;
  double r,r2inv,rsq,grij,etarij2,expm2,t,erfc,dudq;
  double forcecoul,ecoul,prefactor,fpair;

  int inum = blist->inum;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int *atomtype = atom->type;
  int *tag = atom->tag;
  int *ilist = blist->ilist;
  int *jlist;
  int *numneigh = blist->numneigh;
  int **firstneigh = blist->firstneigh;
  double qqrd2e = force->qqrd2e;
  double **cutsq = coulpair->cutsq;
  int itmp;
  double *p_cut_coul = (double *) coulpair->extract("cut_coul",itmp);
  double cut_coulsq = (*p_cut_coul)*(*p_cut_coul);
  double cut_erfc = ERFC_MAX*ERFC_MAX/(g_ewald*g_ewald);
  if (cut_coulsq > cut_erfc) cut_coulsq = cut_erfc;
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  bool ecib,ecjb;

  if (newton) memset(newtonbuf,0,elenum_all*sizeof(double));

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    ecib = electrode_check(i);
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = atomtype[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      ecjb = electrode_check(j);
      if ((ecib ^ ecjb) &&
          (newton || ecib || j < nlocal)) {
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = atomtype[j];
        if (rsq < cutsq[itype][jtype]) {
          if (rsq < cut_coulsq) {
            dudq = erfcr_sqrt(g_ewald*g_ewald*rsq)*g_ewald;
            dudq += (this->*pair_potential)(rsq, itype, jtype);
            if (ecib) {
              elei = eleall2ele[tag2eleall[tag[i]]];
              m[elei] -= q[j]*dudq;
            }
	          else if (j < nlocal) {
              elej = eleall2ele[tag2eleall[tag[j]]];
              m[elej] -= q[i]*dudq;
            }
	          else if (newton) {
	            eleallj = tag2eleall[tag[j]];
	            newtonbuf[eleallj] -= q[i]*dudq;
            }
	        }
        }
      }
    }
  }

  if (newton) {
    MPI_Allreduce(MPI_IN_PLACE,newtonbuf,elenum_all,MPI_DOUBLE,MPI_SUM,world);
    for (elei = 0; elei < elenum; ++elei) {
      tagi = ele2tag[elei];
      m[elei] += newtonbuf[tag2eleall[tagi]];
    }
  }

  Ctime2 = MPI_Wtime();
  Ctime += Ctime2-Ctime1;
}

/* ---------------------------------------------------------------------- */
void FixConp::blist_coul_cal_post_force()
{
  Ctime1 = MPI_Wtime();
  int i,j,k,ii,jj,jnum,itype,jtype,idx1d;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz;
  double r,r2inv,rsq,grij,etarij2,expm2,t,erfcr,dudq;
  double forcecoul,ecoul,prefactor,fpair;
  int newton_pair = force->newton_pair;
  int inum = blist->inum;
  int nlocal = atom->nlocal;
  int *atomtype = atom->type;
  int *tag = atom->tag;
  int *ilist = blist->ilist;
  int *jlist;
  int *numneigh = blist->numneigh;
  int **firstneigh = blist->firstneigh;
  
  double qqrd2e = force->qqrd2e;
  double **cutsq = coulpair->cutsq;
  int itmp;
  double *p_cut_coul = (double *) coulpair->extract("cut_coul",itmp);
  double cut_coulsq = (*p_cut_coul)*(*p_cut_coul);
  double cut_erfc = ERFC_MAX*ERFC_MAX/(eta*eta); // only eta*r used in erfc
  if (cut_coulsq > cut_erfc) cut_coulsq = cut_erfc;
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  bool eleilocal,elejlocal;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    eleilocal = !!(electrode_check(i));
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = atomtype[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      elejlocal = !!(electrode_check(j));
      if (eleilocal ^ elejlocal) {
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = atomtype[j];
        if (rsq < cutsq[itype][jtype]) {
          etarij2 = eta*eta*rsq;
          if (etarij2 < ERFC_MAX) {
            prefactor = qqrd2e*qtmp*q[j];
            forcecoul = -prefactor*(this->*pair_force)(etarij2, itype, jtype);
            fpair = forcecoul/rsq;
            // following logic is asymmetric
            // because we always know i < nlocal (but j could be ghost)
            if (!eleilocal) {
              f[i][0] += delx*forcecoul;
              f[i][1] += dely*forcecoul;
              f[i][2] += delz*forcecoul;
            }
            else if (newton || j < nlocal) {
              f[j][0] -= delx*forcecoul;
              f[j][1] -= dely*forcecoul;
              f[j][2] -= delz*forcecoul;
            }
            ecoul = -prefactor*(this->*pair_potential)(etarij2, itype, jtype);
            force->pair->ev_tally(i,j,nlocal,newton_pair,0,ecoul,fpair,delx,dely,delz); //evdwl=0
          }
        }
      }
    }
  }
  Ctime2 = MPI_Wtime();
  Ctime += Ctime2-Ctime1;
}

double FixConp::erfcr_sqrt(double a2_r2) {
  double a_r = sqrt(a2_r2);
  double expm2 = exp(-a2_r2);
  double t = 1.0 / (1.0 + EWALD_P*a_r);
  return t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2 / a_r;
}

double FixConp::ferfcr_sqrt(double a2_r2) {
  double a_r = sqrt(a2_r2);
  double expm2 = exp(-a2_r2);
  double t = 1.0 / (1.0 + EWALD_P*a_r);
  double erfcr = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2 / a_r;
  return erfcr + EWALD_F*expm2;
}

double FixConp::eta_potential_A(double rsq, int /*itype*/, int /*jtype*/) {
  double dudq = 0;
  double etarij2 = eta*eta*rsq/2;
  if (etarij2 < ERFC_MAX) dudq -= erfcr_sqrt(etarij2)*eta/sqrt(2);
  return dudq;
}

double FixConp::eta_potential(double rsq, int /*itype*/, int /*jtype*/) {
  double dudq = 0;
  double etarij2 = eta*eta*rsq;
  if (etarij2 < ERFC_MAX) dudq -= erfcr_sqrt(etarij2)*eta;
  return dudq;
}

double FixConp::eta_force(double rsq, int /*itype*/, int /*jtype*/) {
  double etarij2 = eta*eta*rsq;
  return eta*ferfcr_sqrt(etarij2);
}

int FixConp::modify_param(int narg, char ** arg) {
  if (pairmode == ETA) {
    error->all(FLERR,"Can't fix_modify conp parameters in basic pair mode");
  }
  if (strcmp(arg[0], "ehgo") == 0) {
    if (!ehgo_allocated) ehgo_allocate();
    if (strcmp(arg[1], "kappa") == 0) kappa = utils::numeric(FLERR,arg[2],false,lmp);
    else if (strcmp(arg[1], "coeff") == 0) {
      if (narg != 5) error->all(FLERR,"Invalid number of inputs for EHGO coeff setting");
      int ilo, ihi;
      utils::bounds(FLERR,arg[2],1,atom->ntypes,ilo,ihi,error);
      double eta_one = utils::numeric(FLERR,arg[3],false,lmp);
      double u0_one = utils::numeric(FLERR,arg[4],false,lmp);
      int count = 0;
      for (int i = ilo; i <= ihi; ++i) {
        eta_i[i] = eta_one;
        u0_i[i] = u0_one / evscale; // convert from eV/e^2 to A^-1
        ++count;
      }
      if (count == 0) error->all(FLERR,"Couldn't set EHGO coeffs with mintype more than maxtype");
    }
    else error->all(FLERR,"Invalid entry for EHGO coeff setting");
  }
  return 0;
}

void FixConp::ehgo_setup_tables() {
  int ntypes = atom->ntypes;
  double CON_s2overPIS = sqrt(2.0)/MY_PIS;
  double sq8 = sqrt(8.0);
  // check if any coeffs set
  bool setflag = false;
  int i,j;
  for (i = 1; i <= ntypes; ++i) {
    if (eta_i[i] && u0_i[i]) setflag = true;
  }
  if (setflag) {
    double* f_i = new double[ntypes+1];
    memset(f_i,0,(ntypes+1)*sizeof(double));
    for (i = 1; i <= ntypes; ++i) {
      f_i[i] = u0_i[i] - CON_s2overPIS*eta_i[i];
    }
    for (i = 1; i <= ntypes; ++i) {
      for (j = 1; j <= i; ++j) {
        if (eta_i[i] && eta_i[j]) { // both etas non-zero
          double etasq = eta_i[i]*eta_i[i]+eta_i[j]*eta_i[j];
          double etaprod = eta_i[i]*eta_i[j];
          eta_ij[i][j] = etaprod/sqrt(etasq);
          double o_ij = sq8*pow(eta_ij[i][j],3.0)/(etaprod*sqrt(etaprod));
          double f_ij = 0.5*kappa*(f_i[i]+f_i[j]);
          fo_ij[i][j] = f_ij*o_ij;
        }
        else { // one eta is zero
          eta_ij[i][j] = eta_i[i] + eta_i[j]; // set eta_ij to the nonzero eta_i
          // leave fo_ij[i][j] = 0 from earlier memset
        }
      }
      eta_ij[j][i] = eta_ij[i][j]; // not sure if need to exclude i == j
      fo_ij[j][i] = fo_ij[i][j];
    }
  } else {
    ehgo_deallocate();
    pairmode = ETA;
    error->warning(FLERR,"Switching back to eta pairmode because no EHGO settings were detected");
  }
}

double FixConp::ehgo_potential(double rsq, int itype, int jtype) {
  double dudq = 0;
  double etaij = eta_ij[itype][jtype];
  double foij = fo_ij[itype][jtype];
  double etarij2 = etaij*etaij*rsq;
  if (etarij2 < 17) dudq += foij*exp(-2*etarij2); 
  if (etarij2 < ERFC_MAX) dudq -= erfcr_sqrt(etarij2)*etaij;
  return dudq;
}

double FixConp::ehgo_force(double rsq, int itype, int jtype) {
  double etaij = eta_ij[itype][jtype];
  double foij = fo_ij[itype][jtype];
  double etarij2 = etaij*etaij*rsq;
  return -4*etarij2*foij*exp(-2*etarij2)+eta*ferfcr_sqrt(etarij2);
}

void FixConp::ehgo_allocate() {
  int ntypes = atom->ntypes;
  eta_i = new double[ntypes+1];
  u0_i  = new double[ntypes+1];
  memory->create(eta_ij,ntypes+1,ntypes+1,"fixconp:eta_ij");
  memory->create(fo_ij,ntypes+1,ntypes+1,"fixconp:fo_ij");
  memset(eta_ij,0,(ntypes+1)*(ntypes+1)*sizeof(double));
  memset(fo_ij,0,(ntypes+1)*(ntypes+1)*sizeof(double));
  ehgo_allocated = true;
}

void FixConp::ehgo_deallocate() {
  delete [] eta_i;
  delete [] u0_i;
  memory->destroy(eta_ij);
  memory->destroy(fo_ij);
  ehgo_allocated = false;
}
