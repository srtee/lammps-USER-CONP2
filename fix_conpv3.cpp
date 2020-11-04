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
   Version Nov/2020
   Shern Ren Tee (UQ AIBN), s.tee@uq.edu.au
------------------------------------------------------------------------- */

#include "math.h"
#include "string.h"
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
#include "fix_conpv3.h"
#include "pair_hybrid.h"

#include "pair.h"
#include "kspace.h"
#include "comm.h"
#include "mpi.h"
#include "math_const.h"
#include "neigh_list.h"
#include "domain.h"
#include "iostream"

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{CONSTANT,EQUAL,ATOM};

extern "C" {
  void dgetrf_(const int *M,const int *N,double *A,const int *lda,int *ipiv,int *info);
  void dgetri_(const int *N,double *A,const int *lda,const int *ipiv,double *work,const int *lwork,int *info);
}
/* ---------------------------------------------------------------------- */

FixConpV3::FixConpV3(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),coulpair(NULL),qlstr(NULL),qrstr(NULL),
  i2eleall(NULL),arrelesetq(NULL)
{
  if (narg < 11) error->all(FLERR,"Illegal fix conp command");
  qlstyle = qrstyle = CONSTANT;
  ilevel_respa = 0;
  maxiter = 100;
  tolerance = 0.000001;
  everynum = force->numeric(FLERR,arg[3]);
  eta = force->numeric(FLERR,arg[4]);
  molidL = force->inumeric(FLERR,arg[5]);
  molidR = force->inumeric(FLERR,arg[6]);
  if (strstr(arg[7],"v_") == arg[7]) {
    int n = strlen(&arg[7][2]) + 1;
    qlstr = new char[n];
    strcpy(qlstr,&arg[7][2]);
    qlstyle = EQUAL;
  } else {
    qL = force->numeric(FLERR,arg[7]);
  }
  if (strstr(arg[8],"v_") == arg[8]) {
    int n = strlen(&arg[8][2]) + 1;
    qrstr = new char[n];
    strcpy(qrstr,&arg[8][2]);
    qrstyle = EQUAL;
  } else {
    qR = force->numeric(FLERR,arg[8]);
  }
  if (strcmp(arg[9],"cg") == 0) {
    minimizer = 0;
  } else if (strcmp(arg[9],"inv") == 0) {
    minimizer = 1;
  } else error->all(FLERR,"Unknown minimization method");
  
  outf = fopen(arg[10],"w");
  if (narg == 12) {
    outa = NULL;
    a_matrix_fp = fopen(arg[11],"r");
    if (a_matrix_fp == NULL) error->all(FLERR,"Cannot open A matrix file");
    if (strcmp(arg[11],"org") == 0) {
      a_matrix_f = 1;
    } else if (strcmp(arg[11],"inv") == 0) {
      a_matrix_f = 2;
    } else {
      error->all(FLERR,"Unknown A matrix type");
    }
  } else {
    a_matrix_f = 0;
  }
  elenum = elenum_old = 0;
  csk = snk = NULL;
  aaa_all = NULL;
  bbb_all = NULL;
  tag2eleall = eleall2tag = curr_tag2eleall = ele2tag = NULL;
  Btime = cgtime = Ctime = Ktime = 0;
  runstage = 0; //after operation
                //0:init; 1: a_cal; 2: first sin/cos cal; 3: inv only, aaa inverse
  totsetq = 0;
  gotsetq = 0;  //=1 after getting setq vector

  scalar_flag = 1;
  extscalar = 0;
  global_freq = 1;

  grow_arrays(atom->nmax);
  comm_forward = 2;
}

/* ---------------------------------------------------------------------- */

FixConpV3::~FixConpV3()
{
  fclose(outf);
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);
  memory->destroy(i2eleall);
  memory->destroy(arrelesetq);
  delete [] aaa_all;
  delete [] bbb_all;
  delete [] curr_tag2eleall;
  delete [] tag2eleall;
  delete [] eleall2tag;
  delete [] ele2tag;
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;
  delete [] ug;
  delete [] sfacrl;
  delete [] sfacim;
  delete [] sfacrl_all;
  delete [] sfacim_all;
  delete [] qlstr;
  delete [] qrstr;
  delete [] elesetq;
}

/* ---------------------------------------------------------------------- */

int FixConpV3::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixConpV3::init()
{
  MPI_Comm_rank(world,&me);
  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
  
  // check variables
  if (qlstr) {
    qlvar = input->variable->find(qlstr);
    if (qlvar < 0)
      error->all(FLERR,"Variable name 1 for fix conp does not exist");
    if (!input->variable->equalstyle(qlvar))
      error->all(FLERR,"Variable 1 for fix conp is invalid style");
  }
 
  if (qrstr) {
    qrvar = input->variable->find(qrstr);
    if (qrvar < 0)
      error->all(FLERR,"Variable name 2 for fix conp does not exist");
    if (!input->variable->equalstyle(qrvar))
      error->all(FLERR,"Variable 2 for fix conp is invalid style");
  }
}

/* ---------------------------------------------------------------------- */

void FixConpV3::setup(int vflag)
{
  //Pair *coulpair;

  // assign coulpair to either the existing pair style if it matches 'coul'
  // or, if hybrid, the pair style matching 'coul'
  // and if neither are true then something has gone horribly wrong!
  coulpair = NULL;
  coulpair = (Pair *) force->pair_match("coul",0);
  if (coulpair == NULL) {
    // return 1st hybrid substyle matching coul (inexactly)
    coulpair = (Pair *) force->pair_match("coul",0,1);
    }
  if (coulpair == NULL) error->all(FLERR,"Must use conp with coul pair style");
  //PairHybrid *pairhybrid = dynamic_cast<PairHybrid*>(force->pair);
  //Pair *coulpair = pairhybrid->styles[0];



  g_ewald = force->kspace->g_ewald;
  slab_volfactor = force->kspace->slab_volfactor;
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

// Copied from ewald.cpp
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

  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];
  ug = new double[kmax3d];

  double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
  double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
  double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
  gsqmx = MAX(gsqxmx,gsqymx);
  gsqmx = MAX(gsqmx,gsqzmx);

  gsqmx *= 1.00001;

  coeffs();
  kmax_created = kmax;

//copied from ewald.cpp end

  int nmax = atom->nmax;
  //double evscale = 0.069447;
  //vL *= evscale;
  //vR *= evscale;
  
  memory->create3d_offset(cs,-kmax,kmax,3,nmax,"fixconpv3:cs");
  memory->create3d_offset(sn,-kmax,kmax,3,nmax,"fixconpv3:sn");
  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];
  sfacrl_all = new double[kmax3d];
  sfacim_all = new double[kmax3d];
  tag2eleall = new int[natoms+1];
  curr_tag2eleall = new int[natoms+1];
  if (runstage == 0) {
    int i;
    int nlocal = atom->nlocal;
    for ( i = 0; i < nlocal; i++) {
      if (electrode_check(i)) ++elenum;
      i2eleall[i] = -1;
      arrelesetq[i] = 0.0;
    }
    MPI_Allreduce(&elenum,&elenum_all,1,MPI_INT,MPI_SUM,world);
    
    eleall2tag = new int[elenum_all];
    aaa_all = new double[elenum_all*elenum_all];
    bbb_all = new double[elenum_all];
    ele2tag = new int[elenum];
    for (i = 0; i < natoms+1; i++) tag2eleall[i] = -1;
    for (i = 0; i < natoms+1; i++) curr_tag2eleall[i] = -1;
    if (minimizer == 0) {
      eleallq = new double[elenum_all];
    }
    if (a_matrix_f == 0) {
      if (me == 0) outa = fopen("amatrix","w");
      a_cal();
    } else {
      a_read();
    }
    runstage = 1;
    
    int gotsetq = 0;
    double totsetq = 0;
    b_setq_cal();
    equation_solve();
    double addv = 0;
    elesetq = new double[elenum_all]; 
    get_setq();
    gotsetq = 1;

  }
    
}

/* ---------------------------------------------------------------------- */

void FixConpV3::pre_force(int vflag)
{
  if(update->ntimestep % everynum == 0) {
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
  force_cal(vflag);
}

/* ---------------------------------------------------------------------- */

void FixConpV3::pre_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

int FixConpV3::electrode_check(int atomid)
{
  int *molid = atom->molecule;
  if (molid[atomid] == molidL) return 1;
  else if (molid[atomid] == molidR) return -1;
  else return 0;
}

/* ----------------------------------------------------------------------*/

void FixConpV3::b_setq_cal()
{
  int i,j;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  double evscale = force->qe2f/force->qqr2e;
  fprintf(outf,"%g \n",evscale);
  int elenum = 0;
  for (i = 0; i < nlocal; i++) {
    if(electrode_check(i)) elenum++;
  }
  double bbb[elenum];
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i) == 1) {
      bbb[j] = -0.5*evscale;
      j++;
    }
    if (electrode_check(i) == -1) {
      bbb[j] = 0.5*evscale;
      j++;
    }
  }
  b_comm(elenum,bbb);
  if (runstage == 1) runstage = 2;
  //delete [] bbb;
}

/* ----------------------------------------------------------------------*/

void FixConpV3::b_cal()
{
  //fprintf(outf,"i   id    Bvec\n");
  Ktime1 = MPI_Wtime();
  int i,j,k;
  int nmax = atom->nmax;
  if (atom->nlocal > nmax) {
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    kmax_created = kmax;
  }
  sincos_b();
  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  double **x = atom->x;
  double *q = atom->q;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int kx,ky,kz;
  double cypz,sypz,exprl,expim,kspacetmp;
  int elenum = 0;
  for (i = 0; i < nlocal; i++) {
    if(electrode_check(i)) elenum++;
  }
  double bbb[elenum];
  j = 0;

  // update voltages due to variables
  // double evscale = 0.069447;
  // fprintf(outf,"Voltages: (left) %g   (right) %g\n",vL,vR);
  j=0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      bbb[j] = 0;
  	    //fprintf(outf,"%d      %d      %g\n",j,ele2tag[j],bbb[j]);
      j++;
    }
  }
  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    j = 0;
    for (i = 0; i < nlocal; i++) {
      if (electrode_check(i)) {
        cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
        sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
        exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
        expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;
        bbb[j] -= 2.0*ug[k]*(exprl*sfacrl_all[k]+expim*sfacim_all[k]);
        j++;
      }
    }
  }
  j=0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      // fprintf(outf,"%d      %d      %g\n",j,ele2tag[j],bbb[j]);
      j++;
    }
  }
 
  //slabcorrection and create ele tag list in current timestep
  double slabcorrtmp = 0.0;
  double slabcorrtmp_all = 0.0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i) == 0) {
      slabcorrtmp += 4*q[i]*MY_PI*x[i][2]/volume;
    }
  }
  MPI_Allreduce(&slabcorrtmp,&slabcorrtmp_all,1,MPI_DOUBLE,MPI_SUM,world);
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      bbb[j] -= x[i][2]*slabcorrtmp_all;
      ele2tag[j] = tag[i];
      j++;
    }
  }
  Ktime2 = MPI_Wtime();
  Ktime += Ktime2-Ktime1;
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      //fprintf(outf,"%d      %d      %g\n",j,ele2tag[j],bbb[j]);
      j++;
    }
  }
  coul_cal(1,bbb,ele2tag);
  b_comm(elenum,bbb);
}

/* ----------------------------------------------------------------------*/

void FixConpV3::b_comm(int ielenum, double* bbbptr)
{
  //elenum_list and displs for gathering ele tag list and bbb
  int i;
  int *tag = atom->tag;
  int nprocs = comm->nprocs;
  int elenum_list[nprocs];
  MPI_Allgather(&ielenum,1,MPI_INT,elenum_list,1,MPI_INT,world);
  int displs[nprocs];
  displs[0] = 0;
  int displssum = 0;
  for (i = 1; i < nprocs; ++i) {
    displssum += elenum_list[i-1];
    displs[i] = displssum;
  }

  //gather ele tag list
  int ele_taglist_all[elenum_all];
  int tagi;
  MPI_Allgatherv(ele2tag,ielenum,MPI_INT,&ele_taglist_all,elenum_list,displs,MPI_INT,world);
  for (i = 0; i < elenum_all; i++) {
    tagi = ele_taglist_all[i];
    curr_tag2eleall[tagi] = i;
  }  

  //gather b to bbb_all and sort in the same order as aaa_all
  double bbb_buf[elenum_all];
  MPI_Allgatherv(bbbptr,ielenum,MPI_DOUBLE,&bbb_buf,elenum_list,displs,MPI_DOUBLE,world);
  int elei;
  for (i = 0; i < elenum_all; i++) {
    tagi = eleall2tag[i];
    elei = curr_tag2eleall[tagi];
    bbb_all[i] = bbb_buf[elei];
    //fprintf(outf,"%d      %d       %g\n",i,tagi,bbb_all[i]);
  }
}

/*----------------------------------------------------------------------- */

void FixConpV3::equation_solve()
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
void FixConpV3::a_read()
{
  int nlocal = atom->nlocal;
  int nmax = nlocal + atom->nghost;
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
          aaa_all[idx1d] = atof(word);
        }
        word = strtok(NULL," \t");
        i++;
      }
    }
    fclose(a_matrix_fp);
  }
  MPI_Bcast(eleall2tag,elenum_all,MPI_INT,0,world);
  MPI_Bcast(aaa_all,elenum_all*elenum_all,MPI_DOUBLE,0,world);

  int tagi,imap;
  for (i = 0; i < elenum_all; i++) {
    tagi = eleall2tag[i];
    imap = atom->map(tagi);
    if (imap >= 0 && imap < nmax) i2eleall[imap] = tagi;
    tag2eleall[tagi] = i;
  }
}

/*----------------------------------------------------------------------- */
void FixConpV3::a_cal()
{
  double t1,t2;
  t1 = MPI_Wtime();
  Ktime1 = MPI_Wtime();
  if (me == 0) {
    fprintf(outf,"A matrix calculating ...\n");
  }

  double **eleallx = NULL;
  memory->create(eleallx,elenum_all,3,"fixconpv3:eleallx");

  int nprocs = comm->nprocs;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;
  int i,j,k;
  int elenum_list[nprocs];
  MPI_Allgather(&elenum,1,MPI_INT,elenum_list,1,MPI_INT,world);
  int displs[nprocs];
  displs[0] = 0;
  int displssum = 0;
  for (i = 1; i < nprocs; ++i) {
    displssum += elenum_list[i-1];
    displs[i] = displssum;
  }
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      i2eleall[i] = j+displs[me];
      ele2tag[j] = tag[i];
      j++;
    }
  }
  comm->forward_comm_fix(this);

  //gather tag,x and q
  double **x = atom->x;
  
  double *elexyzlist = new double[3*elenum];
  double *elexyzlist_all = new double[3*elenum_all];
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      elexyzlist[j] = x[i][0];
      j++;
      elexyzlist[j] = x[i][1];
      j++;
      elexyzlist[j] = x[i][2];
      j++;
    }
  }
  MPI_Allgatherv(ele2tag,elenum,MPI_INT,eleall2tag,elenum_list,displs,MPI_INT,world);
  int displs2[nprocs];
  int elenum_list2[nprocs];
  for (i = 0; i < nprocs; i++) {
    elenum_list2[i] = elenum_list[i]*3;
    displs2[i] = displs[i]*3;
  }
  MPI_Allgatherv(elexyzlist,elenum*3,MPI_DOUBLE,elexyzlist_all,elenum_list2,displs2,MPI_DOUBLE,world);

  double *aaa = new double[elenum*elenum_all];
  for (i = 0; i < elenum*elenum_all; i++) {
    aaa[i] = 0.0;
  }
  j = 0;
  for (i = 0; i < elenum_all; i++) {
    if (i == 0 && me == 0) fprintf(outa," ");
    if (me == 0) fprintf (outa,"%20d",eleall2tag[i]);
    tag2eleall[eleall2tag[i]] = i;
    eleallx[i][0] = elexyzlist_all[j];
    j++;
    eleallx[i][1] = elexyzlist_all[j];
    j++;
    eleallx[i][2] = elexyzlist_all[j];
    j++;
  }
  if (me == 0) fprintf (outa,"\n");


  memory->create(csk,kcount,elenum_all,"fixconpv3:csk");
  memory->create(snk,kcount,elenum_all,"fixconpv3:snk");
  sincos_a(eleallx);
  delete [] elexyzlist;
  delete [] elexyzlist_all;

  int elealli,elei,idx1d;
  double zi;
  double CON_4PIoverV = MY_4PI/volume;
  double CON_s2overPIS = sqrt(2.0)/MY_PIS;
  double CON_2overPIS = 2.0/MY_PIS;
  int ele2tag2[elenum];
  for (i = 0; i < nlocal; ++i) {
    zi = x[i][2];
    if (electrode_check(i)) {
      elealli = i2eleall[i];
      printf("%d\n",elealli);
      for (k = 0; k < elenum; ++k) {
        if (ele2tag[k] == tag[i]) {
          elei = k;
          break;
        }
      }
      for (j = 0; j < elenum_all; ++j) {
        idx1d = elei*elenum_all+j;
        for (k = 0; k < kcount; ++k) {
          aaa[idx1d] += 2.0*ug[k]*(csk[k][elealli]*csk[k][j]+snk[k][elealli]*snk[k][j]);
        }
        aaa[idx1d] += CON_4PIoverV*zi*eleallx[j][2];
      }
      idx1d = elei*elenum_all+elealli;
      aaa[idx1d] += CON_s2overPIS*eta-CON_2overPIS*g_ewald; //gaussian self correction
    }
  }
  
  memory->destroy(eleallx);
  memory->destroy(csk);
  memory->destroy(snk);

  coul_cal(2,aaa,ele2tag);
  
  int elenum_list3[nprocs];
  int displs3[nprocs];
  for (i = 0; i < nprocs; i++) {
    elenum_list3[i] = elenum_list[i]*elenum_all;
    displs3[i] = displs[i]*elenum_all;
  }
  MPI_Allgatherv(aaa,elenum*elenum_all,MPI_DOUBLE,aaa_all,elenum_list3,displs3,MPI_DOUBLE,world);
  delete [] aaa;
  aaa = NULL;
  for (i = 0; i < elenum_all; ++i) {
    for (j = 0; j < elenum_all; ++j) {
      idx1d = i*elenum_all+j;
      if (j != 0 && me == 0) fprintf(outa," ");
      if (me == 0) fprintf (outa,"%20.12f",aaa_all[idx1d]);
    }
    if (me == 0) fprintf (outa,"\n");
  }
  if(me == 0) fclose(outa);

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
/*--------------------------------------------------------------*/

void FixConpV3::sincos_a(double **eleallx)
{
  int i,m,k,ic;
  int kx,ky,kz;
  double ***csele,***snele;
  memory->create3d_offset(csele,-kmax,kmax,3,elenum_all,"fixconpv3:csele");
  memory->create3d_offset(snele,-kmax,kmax,3,elenum_all,"fixconpv3:snele");
  double sqk,cypz,sypz;
  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      for (i = 0; i < elenum_all; i++) {
        csele[0][ic][i] = 1.0;
        snele[0][ic][i] = 0.0;
        csele[1][ic][i] = cos(unitk[ic]*eleallx[i][ic]);
        snele[1][ic][i] = sin(unitk[ic]*eleallx[i][ic]);
        csele[-1][ic][i] = csele[1][ic][i];
        snele[-1][ic][i] = -snele[1][ic][i];
      }
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        for (i = 0; i < elenum_all; i++) {
          csele[m][ic][i] = csele[m-1][ic][i]*csele[1][ic][i] -
            snele[m-1][ic][i]*snele[1][ic][i];
          snele[m][ic][i] = snele[m-1][ic][i]*csele[1][ic][i] +
            csele[m-1][ic][i]*snele[1][ic][i];
          csele[-m][ic][i] = csele[m][ic][i];
          snele[-m][ic][i] = -snele[m][ic][i];
        }
      }
    }
  }
  for (k = 0; k < kcount; ++k) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    for (i = 0; i < elenum_all; ++i) {
      cypz = csele[ky][1][i]*csele[kz][2][i] - snele[ky][1][i]*snele[kz][2][i];
      sypz = snele[ky][1][i]*csele[kz][2][i] + csele[ky][1][i]*snele[kz][2][i];
      csk[k][i] = csele[kx][0][i]*cypz - snele[kx][0][i]*sypz;
      snk[k][i] = snele[kx][0][i]*cypz + csele[kx][0][i]*sypz;
    }
  }
  memory->destroy3d_offset(csele,-kmax_created);
  memory->destroy3d_offset(snele,-kmax_created);
} 

/*--------------------------------------------------------------*/
void FixConpV3::sincos_b()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  n = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      for (i = 0; i < nlocal; i++) {
          cs[0][ic][i] = 1.0;
          sn[0][ic][i] = 0.0;
          cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
          sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
          cs[-1][ic][i] = cs[1][ic][i];
          sn[-1][ic][i] = -sn[1][ic][i];
        if (electrode_check(i) == 0) {
          cstr1 += q[i]*cs[1][ic][i];
          sstr1 += q[i]*sn[1][ic][i];
        }
      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }
  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        for (i = 0; i < nlocal; i++) {
            cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
              sn[m-1][ic][i]*sn[1][ic][i];
            sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
              cs[m-1][ic][i]*sn[1][ic][i];
            cs[-m][ic][i] = cs[m][ic][i];
            sn[-m][ic][i] = -sn[m][ic][i];
          if (electrode_check(i) == 0) {
            cstr1 += q[i]*cs[m][ic][i];
            sstr1 += q[i]*sn[m][ic][i];
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)
  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          if (electrode_check(i) == 0) {
            cstr1 += q[i]*(cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
            sstr1 += q[i]*(sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
            cstr2 += q[i]*(cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
            sstr2 += q[i]*(sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          if (electrode_check(i) == 0) {
            cstr1 += q[i]*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
            sstr1 += q[i]*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
            cstr2 += q[i]*(cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
            sstr2 += q[i]*(sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          if (electrode_check(i) == 0) {
            cstr1 += q[i]*(cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
            sstr1 += q[i]*(sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
            cstr2 += q[i]*(cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
            sstr2 += q[i]*(sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          for (i = 0; i < nlocal; i++) {
            if (electrode_check(i) == 0) {
              clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
              slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
              cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
              slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
              cstr2 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr2 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
              slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
              cstr3 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr3 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

              clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
              slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
              cstr4 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr4 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }
          }
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;
        }
      }
    }
  }
  if (runstage == 1) runstage = 2;
}

/* ---------------------------------------------------------------------- */
void FixConpV3::cg()
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
      idx1d= i*elenum_all+j;
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
void FixConpV3::inv()
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

    dgetrf_(&m,&n,aaa_all,&lda,ipiv,&info);
    infosum = info;
    dgetri_(&n,aaa_all,&lda,ipiv,work,&lwork,&info);
    infosum += info;
    delete [] ipiv;
    ipiv = NULL;
    delete [] work;
    work = NULL;

    if (infosum != 0) error->all(FLERR,"Inversion failed!");
    
    // here we project aaa_all onto
    // the null space of e

    double *ainve = new double[elenum_all];
    double totinve = 0;
    idx1d = 0;

    for (i = 0; i < elenum_all; i++) {
      ainve[i] = 0;
      for (j = 0; j < elenum_all; j++) {
        ainve[i] += aaa_all[idx1d];
	idx1d++;
      }
      totinve += ainve[i];
    }

    if (totinve*totinve > 1e-8) {
      idx1d = 0;
      for (i = 0; i < elenum_all; i++) {
        for (j = 0; j < elenum_all; j++) {
          aaa_all[idx1d] -= ainve[i]*ainve[j]/totinve;
	  idx1d++;
	}
      }
    }

    delete [] ainve;
    ainve = NULL;

    if (me == 0) {
      FILE *outinva = fopen("inv_a_matrix","w");
      for (i = 0; i < elenum_all; i++) {
        if(i == 0) fprintf (outinva," ");
        fprintf (outinva,"%12d",eleall2tag[i]);
      }
      fprintf (outinva,"\n");
      for (k = 0; k < elenum_all*elenum_all; k++) {
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

/* ---------------------------------------------------------------------- */

void FixConpV3::get_setq()
{
  int iall,jall,i,j,idx1d;
  int elealli,tagi;
  double eleallq_i;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nall = nlocal+atom->nghost;
  double netcharge_left_local = 0;
  //double netcharge = 0;

  if (minimizer == 0) { // cg solver used
    for (iall = 0; iall < elenum_all; ++iall) {
      elesetq[iall] = eleallq[iall];
      //netcharge += eleallq[iall];
    }
  } else if (minimizer == 1) { // inv solver used
    idx1d = 0;
    for (iall = 0; iall < elenum_all; ++iall) {
      elesetq[iall] = 0;
      for (jall = 0; jall < elenum_all; ++jall) {
        elesetq[iall] += aaa_all[idx1d]*bbb_all[jall];
	idx1d++;
      }
      //fprintf(outf,"%d     %g\n",iall,elesetq[iall]);
      //netcharge += elesetq[iall];
    }
  }
  //for (iall = 0; iall < elenum_all; ++iall) {
  //  elesetq[iall] -= netcharge/elenum_all;
    // if (me == 0) fprintf(outf,"%d      %g\n",iall,elesetq[iall]);
  //}

  //  now we need to get total left charge

  for (i = 0; i < nlocal; ++i) {
    if (electrode_check(i) == 1) {
      tagi = tag[i];
      elealli = tag2eleall[tagi];
      netcharge_left_local += elesetq[elealli];
      //fprintf(outf,"%g\n",netcharge_left_local);
    }
  }
  MPI_Allreduce(&netcharge_left_local,&totsetq,1,MPI_DOUBLE,MPI_SUM,world);
  //if (me == 0) fprintf(outf,"%g\n",totsetq);
}

/* ---------------------------------------------------------------------- */

void FixConpV3::update_charge()
{
  int i,j,idx1d;
  int elealli,tagi;
  double eleallq_i;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nall = nlocal+atom->nghost;
  //double netcharge_local = 0;
  //double netcharge = 0;
  double netcharge_left_local = 0;
  double netcharge_left = 0;
  double *q = atom->q;    
  if (minimizer == 0) {
    for (i = 0; i < nall; ++i) {
      if (electrode_check(i)) {
        tagi = tag[i];
        elealli = tag2eleall[tagi];
        q[i] = eleallq[elealli];
	// if (i < nlocal) netcharge_local += q[i];
      }
    }
  } else if (minimizer == 1) {
    for (i = 0; i < nall; ++i) {
      if (electrode_check(i)) {
        tagi = tag[i];
        elealli = tag2eleall[tagi];
	idx1d = elealli*elenum_all;
        eleallq_i = 0.0;
        for (j = 0; j < elenum_all; j++) {
          eleallq_i += aaa_all[idx1d]*bbb_all[j];
	  idx1d++;
        }
        q[i] = eleallq_i;
        // if (i < nlocal) netcharge_local += eleallq_i;
      } 
    }
  }
  // MPI_Allreduce(&netcharge_local,&netcharge,1,MPI_DOUBLE,MPI_SUM,world);
  // for (i = 0; i < nall; ++i) {
  //   if (electrode_check(i)) {
  //     q[i] -= netcharge/elenum_all;
  //     if (i < nlocal && electrode_check(i) == 1) netcharge_left_local += q[i];
  //   }
  // }

  //  now we need to get total left charge
  for (i = 0; i < nlocal; ++i) {
    if (electrode_check(i) == 1) netcharge_left_local += q[i];
  }
  MPI_Allreduce(&netcharge_left_local,&netcharge_left,1,MPI_DOUBLE,MPI_SUM,world);

  //  calculate additional charge needed
  //  this fragment is the only difference from fix_conq

  //  now qL and qR are left and right *voltages*
  //  evscale was included in the precalculation of elesetq
  if (qlstyle == EQUAL) qL = input->variable->compute_equal(qlvar);
  if (qrstyle == EQUAL) qR = input->variable->compute_equal(qrvar);
  addv = qR - qL;
  for (i = 0; i < nall; ++i) {
    if (electrode_check(i)) {
      tagi = tag[i];
      elealli = tag2eleall[tagi];
      q[i] += addv*elesetq[elealli];
    }
  }
  //  hack: we will use addv to store total electrode charge
  addv *= totsetq;
  addv += netcharge_left;

}
/* ---------------------------------------------------------------------- */
void FixConpV3::force_cal(int vflag)
{
  int i;
  if (force->kspace->energy) {
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
  coul_cal(0,NULL,NULL);

}
/* ---------------------------------------------------------------------- */
void FixConpV3::coul_cal(int coulcalflag,double *m,int *ele2tag)
{
  Ctime1 = MPI_Wtime();
  //coulcalflag = 2: a_cal; 1: b_cal; 0: force_cal
  int i,j,k,ii,jj,jnum,itype,jtype,idx1d;
  int checksum,elei,elej,elealli,eleallj;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz;
  double r,r2inv,rsq,grij,etarij,expm2,t,erfc,dudq;
  double forcecoul,ecoul,prefactor,fpair;

  int inum = coulpair->list->inum;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int *atomtype = atom->type;
  int *tag = atom->tag;
  int *ilist = coulpair->list->ilist;
  int *jlist;
  int *numneigh = coulpair->list->numneigh;
  int **firstneigh = coulpair->list->firstneigh;
  
  double qqrd2e = force->qqrd2e;
  double **cutsq = coulpair->cutsq;
  int itmp;
  double *p_cut_coul = (double *) coulpair->extract("cut_coul",itmp);
  double cut_coulsq = (*p_cut_coul)*(*p_cut_coul);
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
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
      checksum = abs(electrode_check(i))+abs(electrode_check(j));
      if (checksum == 1 || checksum == 2) {
        if (coulcalflag == 0 || checksum == coulcalflag) {
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;
          jtype = atomtype[j];
          if (rsq < cutsq[itype][jtype]) {
            r2inv = 1.0/rsq;
            if (rsq < cut_coulsq) {
              dudq =0.0;
              r = sqrt(rsq);
              if (coulcalflag != 0) {
                grij = g_ewald * r;
                expm2 = exp(-grij*grij);
                t = 1.0 / (1.0 + EWALD_P*grij);
                erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                dudq = erfc/r;
              }
              if (checksum == 1) etarij = eta*r;
              else if (checksum == 2) etarij = eta*r/sqrt(2);
              expm2 = exp(-etarij*etarij);
              t = 1.0 / (1.0+EWALD_P*etarij);
              erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
  
              if (coulcalflag == 0) {
                prefactor = qqrd2e*qtmp*q[j]/r;
                forcecoul = -prefactor*(erfc+EWALD_F*etarij*expm2);
                fpair = forcecoul*r2inv;
                f[i][0] += delx*forcecoul;
                f[i][1] += dely*forcecoul;
                f[i][2] += delz*forcecoul;
                if (newton_pair || j < nlocal) {
                  f[j][0] -= delx*forcecoul;
                  f[j][1] -= dely*forcecoul;
                  f[j][2] -= delz*forcecoul;
                }
                ecoul = -prefactor*erfc;
                force->pair->ev_tally(i,j,nlocal,newton_pair,0,ecoul,fpair,delx,dely,delz); //evdwl=0
              } else {
                dudq -= erfc/r;
                elei = -1;
                elej = -1;
                for (k = 0; k < elenum; ++k) {
                  if (i < nlocal) {
                    if (ele2tag[k] == tag[i]) {
                      elei = k;
                      if (coulcalflag == 1) {
                        m[k] -= q[j]*dudq;
                        break;
                      }
                    }
                  }
                  if (j < nlocal) {
                    if (ele2tag[k] == tag[j]) {
                      elej = k;
                      if (coulcalflag == 1) {
                        m[k] -= q[i]*dudq;
                        break;
                      }
                    }
                  }
                }
                if (coulcalflag == 2 && checksum == 2) {
                  elealli = tag2eleall[tag[i]];
                  eleallj = tag2eleall[tag[j]];
                  if (elei != -1) {
                    idx1d = elei*elenum_all+eleallj;
                    m[idx1d] += dudq;
                  }
                  if (elej != -1) {
                    idx1d = elej*elenum_all+elealli;
                    m[idx1d] += dudq;
                  }
                }
              }
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
double FixConpV3::compute_scalar()
{
  return addv;
}


/* ---------------------------------------------------------------------- */
double FixConpV3::rms(int km, double prd, bigint natoms, double q2)
{
  double value = 2.0*q2*g_ewald/prd *
    sqrt(1.0/(MY_PI*km*natoms)) *
    exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));
  return value;
}

/* ---------------------------------------------------------------------- */
void FixConpV3::coeffs()
{
  int k,l,m;
  double sqk;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0]) * (m*unitk[0]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      kcount++;
    }
    sqk = (m*unitk[1]) * (m*unitk[1]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      kcount++;
    }
    sqk = (m*unitk[2]) * (m*unitk[2]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      kcount++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        kcount++;;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        kcount++;

        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = -m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = -m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) +
          (unitk[2]*m) * (unitk[2]*m);
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          kcount++;
        }
      }
    }
  }
}
/* ---------------------------------------------------------------------- */
void FixConpV3::grow_arrays(int nmax)
{
  memory->grow(i2eleall,nmax,"fix_conpv3:i2eleall");
  memory->grow(arrelesetq,nmax,"fix_conpv3:arrelesetq");
}
/* ---------------------------------------------------------------------- */
void FixConpV3::copy_arrays(int i, int j, int /*delflag*/)
{
  i2eleall[j]=i2eleall[i];
  arrelesetq[j]=arrelesetq[i];
}
/* ---------------------------------------------------------------------- */
int FixConpV3::pack_exchange(int i, double *buf)
{
  buf[0]=static_cast<double>(i2eleall[i]);
  buf[1]=arrelesetq[i];
  return 0;
}
/* ---------------------------------------------------------------------- */
int FixConpV3::unpack_exchange(int nlocal, double *buf)
{
  i2eleall[nlocal]=static_cast<int>(buf[0]);
  arrelesetq[nlocal]=buf[1];
  return 0;
}
/* ---------------------------------------------------------------------- */
int FixConpV3::pack_forward_comm(int n, int *list, double *buf,
                                 int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;
  
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++]=static_cast<double>(i2eleall[j]);
    buf[m++]=arrelesetq[j];
  }
  return m;
}
/* ---------------------------------------------------------------------- */
void FixConpV3::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    i2eleall[i]=static_cast<int>(buf[m++]);
    arrelesetq[i]=buf[m++];
  }
}

