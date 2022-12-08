/*  $Id: disulf_vdw.c,v 1.7 1995/06/20 02:59:40 suchanek Exp $  */
/*  $Log: disulf_vdw.c,v $
 * Revision 1.7  1995/06/20  02:59:40  suchanek
 * function renaming
 *
 * Revision 1.6  1995/06/12  02:03:39  suchanek
 * *** empty log message ***
 *
 * Revision 1.5  1995/06/07  16:25:48  suchanek
 * changed ran2 to rand() function
 *
 * Revision 1.4  1995/06/07  15:55:06  suchanek
 * *** empty log message ***
 *
 * Revision 1.3  1995/06/07  15:32:01  suchanek
 * *** empty log message ***
 *
 * Revision 1.2  1993/12/10  15:40:26  egs
 * bug fixes
 *
 * Revision 1.1  93/12/10  09:43:10  egs
 * Initial revision
 *  */

static char rcsid[] = "$Id: disulf_vdw.c,v 1.7 1995/06/20 02:59:40 suchanek Exp $";



#include    <stdio.h>
#include    <string.h>
#include    <math.h>
/* #include    "nr.h" */

#include    "disulfide.h"
#include    "atom.h"

#define     DB_STRING "/home/suchanek/ss/ss_patterns.dat"
#define     POPSIZE          500
#define     GENELENGTH       5
#define     MAXGEN           2000
#define     MINVAL           -180
#define     MAXVAL            179
#define     BIGNUM           2147483647.0

#define     NORMAL_ARGC      5
#define     MAX_ENTRIES      99

#define     SPHERE_RAD       8.0

/**************************************************************************
 *   Code inspired by "Genetic Algorithms.." by David Goldberg  (1989)
 *   Originally coded by Franz Dill in Turbo Pascal,  8/89
 *   Converted to C and liberally changed, 10/90
 *
 *   Hacked and optimized a bit by Eric G. Suchanek, Ph.D. 11/90
 *
 *    To use: Determine representation, possible values for the entries in
 *     oldpop[GENELENGTH].  Change choosenumber() appropriately.
 *     Determine fitness calculation, change determinefitness()
 *     appropriately.
 *
 ***************************************************************************/


static char  *USEAGE = "\nProgram: disulf_vdw.c 1.12 3/29/91 \n\n\
\nUSAGE: %s <max generations> <protein fname> \n\
\t<proximal resnumber> <distal resnumber>\n\n\
\tProgram uses Genetic Algorithm techniques to minimize the\n\
\tenergy of a five-dimensional energy surface, specifically\n\
\tthe conformational energy of Disulfide Bond, a covalent protein\n\
\tstabilizing structural element. The parameter \n\
\t<max generations> specifies how many generations of mutations \n\
\tand crossovers to cycle before stopping. Parameter <protein fname>\n\
\tis the Brookhaven protein databank input filename. Parameter \n\
\t<proximal resnumber> represents the proximal residue for the\n\
\tdisulfide, while <distal residue> is the terminal residue\n\
\tfor the modelled disulfide. The sorted list of \n\
\tconformations is written to the standard output, so it may be \n\
\tredirected at will.\n\n\
\n\t Author: Eric G. Suchanek, Ph.D.\n\
\n\t Copyright 1990, -egs-, all rights reserved.. \n\n";

DISULFIDE SS_array[100];            /* used to hold database disulfides */
DISULFIDE SS_target;                /* the disulfide we're trying to fit */
DISULFIDE SS_real;                  /* the disulfide we're modelling */
TURTLE THE_TURTLE;                  /* global turtle for building SS bonds */
ATOM *protein_atoms;                /* protein atom list */
ATOM *atom_sphere;                  /* sphere of atoms around model SS     */


static int db_entries       = 0;    /* number of entries in database */

static double oldpop[POPSIZE][GENELENGTH];   /* storage for populations */
static double fitness[POPSIZE];              /* storage for fitness of pop   */


static double pcross        = .6;        /* probability of a cross    */
static double pmutation     = .1;       /* probability of a mutation */
static double pmutation2    = .1;       /* prob. of a sign flip mutation */

static int select(void);
void determinefitness();
void initializepopulation();

void parse_args();
void print_population();
void sort_population();
void print_sorted_population();
int setup_turtle();

double compute_avg_fitness();


/* a structure used for sorting */

struct keys {
  int index;
  double energy;
};

struct keys key_pop[POPSIZE];

long idum = (-1);

#define NUMELEM(a) (sizeof(a)/sizeof(a[0]))
#define range ((int)(MAXVAL - MINVAL + 1))
#define CHOOSE ((rand() % ((int) MAXVAL - MINVAL + 1) + MINVAL))
/* #define CHOOSE (ran2(&idum) * ((int) MAXVAL - MINVAL + 1) + MINVAL) */

/* ========================== MAIN PROGRAM ===============================*/

void main (argc, argv)
     int argc; char *argv[];
{  
   int gen              = 0;      /* generation counter        */
   int maxgen           = 0;      /* maximum generations       */
   int nmutation        = 0;      /* number of mutations       */
   int ncross           = 0;      /* number of xovers to date  */
   int oldpopsize       = 0;      /* total population          */
   int newpopsize       = 0;
   int db_index         = 0;      /* db index of disulfide we're fitting */
   int real_ss          = 0;      /* true if real SS at this position */
   int select1;
   int select2;
   double avgfitness;

   register int i,j,k;		  /* work variables */
   int i2, j2;

   char *pdb_path;
   char pdb_fname[128];
   char full_path[128];
   int natoms = 0;
   int proximal = 0;
   int distal = 0;

   ATOM *prox_ca, *distal_ca;     /* real proximal and distal Ca's */
   ATOM centroid_atom;

   VECTOR centroid;
   VECTOR v1, v2;

   static double newpop[POPSIZE][GENELENGTH];

   /* ------- initialize the population ----------- */


/* never returns if there's trouble */
   parse_args(argc,argv,&maxgen, &pdb_fname[0], &proximal, &distal); 

   pdb_path = (char *)getenv("PDB");
   if (pdb_path != (char *)NULL) 
     {
       strcpy(full_path, pdb_path);
       strcat(full_path,pdb_fname);
     }
   else
     strcpy(full_path, pdb_fname);

   db_entries = DISULFIDE_read_disulfide_database(DB_STRING, &SS_array[0],100);
   if (db_entries <= 0) {
     fprintf(stderr,"\nCan't read disulfide database ss_patterns.dat\n");
     exit(-1);
   }
   else
     printf("Read <%d> entries from SS database.\n",db_entries);

   natoms = ATOM_read_pdb(&protein_atoms,full_path);
   if (natoms <= 0) {
     fprintf(stderr,"Can't read PDB file <%s>!\n", pdb_fname);
     exit(-1);
   }

   printf("Read <%d> atoms from file <%s>\n", natoms, pdb_fname);
   prox_ca = ATOM_list_scan(protein_atoms, proximal, "CA");
   distal_ca = ATOM_list_scan(protein_atoms, distal, "CA");

   if (prox_ca == (ATOM *)NULL || distal_ca == (ATOM *)NULL) {
     fprintf(stderr,"Can't find initial C alphas? \n");
     exit(-2);
   }

   v1 = prox_ca->position;
   v2 = distal_ca->position;
   centroid = average_vectors(&v1,&v2);

   ATOM_init(&centroid_atom,"XXXX","XXX", -1, 0, 
	     centroid.x, centroid.y, centroid.z, 0.0);


   /* find all atoms in the protein within SPHERE_RAD of the 
      approximate centroid of the residue pair.
    */

   atom_sphere = ATOM_sphere(&centroid_atom, protein_atoms, SPHERE_RAD);

   i = ATOM_list_length(atom_sphere);
   printf("Got <%d> atoms in sphere.\n",i);
   if (i <= 0) {
     fprintf(stderr,"Can't get sphere of atoms??\n");
     exit(-3);
   }

   srand(proximal + 100); 
   ran2(&idum);

   /* orient the turtle at the proximal residue in orientation #1 */

   i = setup_turtle(proximal);
   if (i <= 0) {
     fprintf(stderr,"Can't set up the turtle at Residue <%d>?\n",
	     proximal);
     exit(-4);
   }

   i = setup_target_disulfide(proximal,distal);
   if (i <= 0) {
     fprintf(stderr,"Can't set up the target Disulfide at Residue <%d>?\n",
	     distal);
     exit(-5);
   }

   real_ss = setup_real_disulfide(proximal,distal);
   if (real_ss <= 0) {
     fprintf(stderr,"Can't set up the real Disulfide at Residue <%d>?\n",
	     distal);
   }

   /* set exclusion flag on real proximal and distal atoms so that they
      don't figure into the collision checking routines...
   */

   ATOM_exclude_backbone(atom_sphere, proximal);
   ATOM_exclude_backbone(atom_sphere, distal);

   ATOM_exclude_sidechain(atom_sphere, proximal);
   ATOM_exclude_sidechain(atom_sphere, distal);

   if (real_ss)
     DISULFIDE_print(&SS_real);
   else
     DISULFIDE_print(&SS_target);

   initializepopulation();
   avgfitness = compute_avg_fitness();
   newpopsize = POPSIZE;

   while (gen++ < maxgen)    /* outside loop, for n generations           */
     {

       /* print stats for gen.   */
       fprintf(stderr,
	       "Gen: %4d Avg Fitness: %8.4lf  xovers: %5d mut: %5d\r",
	       gen,avgfitness,ncross,nmutation); 

       oldpopsize = newpopsize;
       newpopsize = 0;

       /* -------------- determine fitness --------- */

       avgfitness = compute_avg_fitness();


       /* ------ create new population ----------- */

       while(newpopsize < POPSIZE)
	 {
	   float tocross = 0.0;
	   /* first choose two to cross ---------------- */

	   select1 = select();
	   select2 = select();

	   i = newpopsize;
	   newpopsize = newpopsize + 2;
/*	   tocross = ran2(&idum); */
	   tocross = rand()/BIGNUM;

	   /* -------------- crossover         --------- */

#ifdef DBG
	   fprintf(stderr,"\nX: %f",tocross);
#endif
#undef DBG
	   if (tocross < pcross) /* determine if crossover */
	     {
	       ncross++;
	       k = (rand() % (long)GENELENGTH);	/*choose allele to split gene*/
	       for (j=0;j<k;j++)
		 {
		   newpop[i][j] = oldpop[select1][j]; /* copy first part  */
		 }
	       for (j=k;j<GENELENGTH;j++)
		 {
		   newpop[i][j] = oldpop[select2][j]; /* copy second part */
		 }
	       for (j=0;j<k;j++)
		 {
		   newpop[i+1][j] = oldpop[select2][j];	/* copy first part */
		 }
	       for (j=k;j<GENELENGTH;j++)
		 {
		   newpop[i+1][j] = oldpop[select1][j];	/* copy second part */
		 }
	     }
	   else
	     for (j=0;j<GENELENGTH;j++)	/* copy untouched */
	       { 
		 newpop[i][j]   = oldpop[select1][j];
		 newpop[i+1][j] = oldpop[select2][j];
	       }

	   for (j=0;j<GENELENGTH;j++) /* handle mutation  */
	     {
	       if (rand()/BIGNUM < pmutation2) 
		 {
		   nmutation++;
		   newpop[i][j] = -newpop[i][j];

		   if (rand()/BIGNUM < pmutation2)  
		     {
		       newpop[i+1][j] = -newpop[i+1][j];
		       nmutation++;
		     }
		 }
	       else
		 {
		   if (rand()/BIGNUM < pmutation)
		     {
		       nmutation++;
		       newpop[i][j] =  (rand()/BIGNUM * ((int) MAXVAL - MINVAL + 1) + MINVAL);
		     }

		   if (rand()/BIGNUM < pmutation)
		     {
		       nmutation++;
		       newpop[i+1][j] = (rand()/BIGNUM * ((int) MAXVAL - MINVAL + 1) + MINVAL);
		     }
		 }  /* else */
	     }
	 }			/* end loop over members of generation  */

       for (i2=0;i2<POPSIZE;i2++)
	 for (j2=0;j2<GENELENGTH;j2++)
	   oldpop[i][j] = newpop[i][j];

       oldpopsize = newpopsize;
     }				/* end, loop over generation production */

   sort_population(oldpop,1);   /* sort into ascending order */

   print_sorted_population(oldpop,10); /* print top 10 */

 }			/* end, main */

/************************** auxilliary functions *****************************/

void parse_args(argc,argv,gen,fname,prox,distal)
int argc;
char *argv[];
int *gen;
char *fname;
int *prox, *distal;

{
  if (argc < NORMAL_ARGC) {
    fprintf(stderr,USEAGE,argv[0]);
    exit(-1);
  }

  *gen = atoi(argv[1]);
  if (*gen > MAXGEN) {
    fprintf(stderr,"Generation count of: %d too large, set to: %d\n",
	    *gen, MAXGEN);
    *gen = MAXGEN;
  }
  
  strcpy(fname,argv[2]);

  *prox = atoi(argv[3]);
  if (*prox < 0 ) {
    fprintf(stderr,"Error, proximal <%d> out of range, using 0\n",*prox);
    *prox = 1;
  }

  *distal = atoi(argv[4]);
  if (*distal < 0 ) {
    fprintf(stderr,"Error, distal <%d> out of range, using 2\n",*distal);
    *distal = 2;
  }
    
}

double compute_avg_fitness() {
  register int i;
  double sumfitness = 0;	
   for (i=0;i<POPSIZE;i++)
     { 
       determinefitness(i);
       sumfitness = sumfitness + fitness[i];
     }
   return(sumfitness / (float)(POPSIZE+1));
}

 /* ----------- Choose random number between MINVAL and MAXVAL ------------*/
 /* -----------------------------------------------------------------------*/

/* int choosenumber()
{
  return(rand() % range) + MINVAL;
}
*/

/* ----------- Provide a random uniform float between 0 and 1 ------------*/
/* -----------------------------------------------------------------------*/
/*
float vrandom()
{
  return((float)rand()/BIGNUM);
}
*/
 /* ------------------- Selection routine ---------------------------------*/
 /*              Choose a gene, biased by it's fitness                     */
 /* -----------------------------------------------------------------------*/

static int select()
{

 double max  = -9999999.0;
 int   jmax = 0;
 register int   i;
 double zrandom;

 for (i=0;i<POPSIZE;i++) {
   zrandom  = rand() / BIGNUM * fitness[i];
   if ((zrandom) > max)
     {
       max = zrandom;
       jmax = i;
     }
 }
 return(jmax);
}

static int oselect()
{

 double max  = -9999999.0;
 int   jmax = 0;
 register int   i;
 double zrandom;

 for (i=0;i<POPSIZE;i++) {
   zrandom  = rand()/BIGNUM * fitness[i];
   if ((zrandom) > max)
     {
       max = zrandom;
       jmax = i;
     }
 }
 return(jmax);
}

 /* ---------------------- Determine Fitness ----------------------------- */
 /* Customize this function to reflect the application.  Fitness should
    be some function of each gene in the population                        */
 /* -----------------------------------------------------------------------*/

/* this particular fitness function is based on an empirical energy
   function for a disulfide covalent cross-link. it assumes that the
   gene has five alleles of type double, and each allele ranges from
   -180.0 -- 180.0 degrees. the negative of the energy (in Kcal/mol) 
   is returned, since we want to MAXIMIZE fitness (minimize energy)!
*/

void determinefitness(i)
     int i;
{
  float fitval = 0;
  int j;
  double x1, x2, x3, x4, x5;
  double energy = 0.0;
  double rms_pos = 0.0;
  double rms_conf = 0.0;

  DISULFIDE SStmp;
  TURTLE    turt;

  int bumps = 0;

  DISULFIDE_init(&SStmp,"tmp");
  TURTLE_init(&turt,"turtle");

  TURTLE_copy_coords(&turt,&THE_TURTLE); /* turtle in proper orientation */

  x1 = oldpop[i][0]; x2 = oldpop[i][1]; x3 = oldpop[i][2]; 
  x4 = oldpop[i][3]; x5 = oldpop[i][4];

  /* crn 3 40 */
  /*  x1 = -51; x2 = -75.0; x3 = -79.0; x4 = -75.0; x5 = -63.0; */

  DISULFIDE_set_conformation(&SStmp,x1,x2,x3,x4,x5);
  DISULFIDE_build(&SStmp, turt);

  energy = SStmp.energy;   /* computed by build_disulfide */

  rms_pos = DISULFIDE_compare_backbones(&SStmp,&SS_target);
  rms_pos = rms_pos == 0.0 ? -50.0 : rms_pos;

  bumps = count_SS_bumps(&SStmp);

#ifdef DBG
  fprintf(stderr,"Bumpds: %4d\r",bumps); 
  DISULFIDE_print(&SStmp);
  DISULFIDE_print(&SS_target);
#endif

  fitness[i] = -1.0 * (rms_pos + energy + rms_conf + bumps);

}

int count_SS_bumps(SS)
DISULFIDE *SS;
{
  ATOM tmpatm;
  int bumps;
  int len, i;

  VECTOR cb1, sg1, sg2, cb2;

  cb1 = SS->cb_prox; sg1 = SS->sg_prox;
  cb2 = SS->cb_dist; sg2 = SS->sg_dist;

  bumps = 0;

/*  CB is FIXED, so we can ignore its "Collisions"... */

#ifdef DBG
  ATOM_print(&tmpatm);
  fprintf(stderr,"CB Bumps: %d\n",bumps); 
#endif
  
  ATOM_init(&tmpatm,"XXXX","XXX", -1, 0, sg1.x, sg1.y, sg1.z, 0.0);
  ATOM_find_collisions(&tmpatm, atom_sphere);
  bumps += ATOM_collisions(&tmpatm);

  ATOM_init(&tmpatm,"XXXX","XXX", -1, 0, cb2.x, cb2.y, cb2.z, 0.0);
  ATOM_find_collisions(&tmpatm, atom_sphere);
  bumps += ATOM_collisions(&tmpatm);

  ATOM_init(&tmpatm,"XXXX","XXX", -1, 0, sg2.x, sg2.y, sg2.z, 0.0);
  ATOM_find_collisions(&tmpatm, atom_sphere);
  bumps += ATOM_collisions(&tmpatm);

  return(bumps);
}


/* -------------------- Initialize Population -----------------------------*/
/* ------------------------------------------------------------------------*/
void initializepopulation()
{
  int i,j;

  for (i=0;i<POPSIZE;i++) {
    fitness[i] = 0;
    for (j=0;j<GENELENGTH;j++) oldpop[i][j] = CHOOSE;
    determinefitness(i);
  }
}

/* degrees --> radians factor */

#define degrad_factor 0.01745329252
#define raddeg_factor 57.2957795132

double to_rad(deg)
double deg;
{
  return((double)(deg * degrad_factor));
}


/**************** Population manipulation and I/O stuff *********************/

/* currently only works with oldpop population */

void print_population(population)
double population[][GENELENGTH];
{
  register int i,j;

  for (i = 0; i < POPSIZE; i++) {
    determinefitness(i);
    printf("%4d ",i);
    for (j = 0; j < GENELENGTH; j++) {
      printf("%8.2lf ", population[i][j]);
      if (j == GENELENGTH -1)
	printf("%8.4lf\n", fitness[i]);
    }
  }
}

/* sorting the population by energy */

void sort_population(population,updown)
double population[][GENELENGTH];
int updown;
{
  register int i,j;
  int cmp_keysa();
  int cmp_keysd();

  for (i = 0; i < POPSIZE; i++) {
    determinefitness(i);
    for (j = 0; j < GENELENGTH; j++) {
      key_pop[i].index = i;
      key_pop[i].energy = -fitness[i]; /* change sign back into real units */
    }
  }

  if (updown > 0)
    qsort(&key_pop[0], POPSIZE, sizeof(struct keys), cmp_keysa);
  else
    qsort(&key_pop[0], POPSIZE, sizeof(struct keys), cmp_keysd);
}

/* sort the keys structure into descending order */

int cmp_keysd(key1, key2)
struct keys *key1, *key2;
{
  if (key1->energy > key2->energy)
    return(-1);
  else
    {
      if (key1->energy < key2->energy)
	return(1);
      else
	{
	  return(0);
	}
    }
}

/* sort the keys structure into ascending order */

int cmp_keysa(key1, key2)
struct keys *key1, *key2;
{
/*  printf("1: %lf 2: %lf \n", key1->energy, key2->energy); */
  if (key1->energy > key2->energy)
    return(1);
  else
    {
      if (key1->energy < key2->energy)
	return(-1);
      else
	{
	  return(0);
	}
    }
}


/* print the sorted population */

void print_sorted_population(population,toprint)
double population[][GENELENGTH];
int toprint;
{
  register int i,j;
  int index;
  double energy;
  int printmax = 0;

  printf("\n**************** Sorted Chromosome Array *****************\n\n");
  printf("Member  Chi1     Chi2     Chi3     Chi4     Chi5    Fitness \n");
  printf("----------------------------------------------------------\n");

  if (toprint == 0) printmax = POPSIZE;
  else printmax = toprint;

  for (i = 0; i < printmax; i++) {
    index = key_pop[i].index;
    energy = key_pop[i].energy;
    printf("%4d ", index);
    for (j = 0; j < GENELENGTH; j++) {
      printf("%8.2lf ", population[index][j]);
      if (j == GENELENGTH -1)
	printf("%8.4lf\n", energy);
    }
  }
}


/* 
  routine orients the global turtle correctly to allow disulfides to be made
  by using orientation #1. (At Ca, headed toward Cb, with N on the left...
*/

int setup_turtle(proximal)
     int proximal;
{
  int res = 0;

  TURTLE_init(&THE_TURTLE,"The turtle is here");
  res = orient_at_residue(&THE_TURTLE, protein_atoms, proximal, 1);
  
  return(res);

}

int setup_target_disulfide(proximal,distal)
     int proximal, distal;
{
  int res = 0;
  VECTOR bogus;

  VECTOR n1, ca1, c1, cb1, sg1, sg2, cb2, ca2, n2, c2, o1, o2;
  ATOM *tmpatm;

  VECTOR_init(&bogus, -99.9, -99.9, -99.9);

  DISULFIDE_init(&SS_target,"Target Disulfide");
  DISULFIDE_set_resnum(&SS_target,proximal,distal);

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "N");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom N %d\n",proximal);
    return(-1);
  }
  n1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "CA");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom CA %d\n",proximal);
    return(-1);
  }
  ca1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "C");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom C %d\n",proximal);
    return(-1);
  }
  c1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "O");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom O %d\n",proximal);
    return(-1);
  }
  o1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "CB");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom CB %d\n",proximal);
    return(-1);
  }
  cb1 = tmpatm->position;


  tmpatm = ATOM_list_scan(protein_atoms, distal, "N");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom N %d\n", distal);
    return(-1);
  }
  n2 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, distal, "CA");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom CA %d\n", distal);
    return(-1);
  }
  ca2 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, distal, "C");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom C %d\n", distal);
    return(-1);
  }
  c2 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, distal, "O");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom O %d\n", distal);
    return(-1);
  }
  o2 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, distal, "CB");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom CB %d\n", distal);
    return(-1);
  }
  cb2 = tmpatm->position;

  DISULFIDE_set_positions(&SS_target, n1, ca1, c1, o1, cb1, bogus, n2, ca2,
			  c2, o2, cb2, bogus);

  return(1);
}


int setup_real_disulfide(proximal,distal)
     int proximal, distal;
{
  int res = 0;

  VECTOR n1, ca1, c1, cb1, sg1, sg2, cb2, ca2, n2, c2, o1, o2;
  ATOM *tmpatm;

  DISULFIDE_init(&SS_real,"Real Disulfide");
  DISULFIDE_set_resnum(&SS_real,proximal,distal);

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "N");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom N %d\n",proximal);
    return(-1);
  }
  n1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "CA");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom CA %d\n",proximal);
    return(-1);
  }
  ca1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "C");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom C %d\n",proximal);
    return(-1);
  }
  c1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "O");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom O %d\n",proximal);
    return(-1);
  }
  o1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "CB");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom CB %d\n",proximal);
    return(-1);
  }
  cb1 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, proximal, "SG");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom SG %d\n",proximal);
    return(-1);
  }
  sg1 = tmpatm->position;


  tmpatm = ATOM_list_scan(protein_atoms, distal, "N");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom N %d\n", distal);
    return(-1);
  }
  n2 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, distal, "CA");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom CA %d\n", distal);
    return(-1);
  }
  ca2 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, distal, "C");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom C %d\n", distal);
    return(-1);
  }
  c2 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, distal, "O");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom O %d\n", distal);
    return(-1);
  }
  o2 = tmpatm->position;

  tmpatm = ATOM_list_scan(protein_atoms, distal, "CB");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom CB %d\n", distal);
    return(-1);
  }
  cb2 = tmpatm->position;


  tmpatm = ATOM_list_scan(protein_atoms, distal, "SG");
  if (tmpatm == (ATOM *)NULL) {
    fprintf(stderr,"Can't find atom SG %d\n", distal);
    return(-1);
  }
  sg2 = tmpatm->position;

  DISULFIDE_set_positions(&SS_real, n1, ca1, c1, o1, cb1, sg1, n2, ca2,
			  c2, o2, cb2, sg2);

  DISULFIDE_compute_dihedrals(&SS_real);
  DISULFIDE_compute_torsional_energy(&SS_real);

  return(1);
}

