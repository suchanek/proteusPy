

/*           @(#)dmgen.c	1.3        4/2/91           */
static char SCCS_ID[] = "@(#)dmgen.c	1.3 \t\t4/2/91 \tEGS";

#undef dbg1
#undef dbg2 
#undef dbg3

#define NOISY 1

#include    <stdio.h>
#include    <math.h>

#ifndef AMIGA
#include    <strings.h>
#else
#include    <stdlib.h>
#include    <string.h>
#endif

#include    "geometry.h"
#include    "atom.h"
#include    "turtle.h"
#include    "residue.h"
#include    "distmat.h"
#include    "phipsi.h"
#include    "keypop.h"


#define     MAX_POPSIZE          100
static      int POPSIZE = 0;

#define     MAXGEN           2000
#define     BIGNUM           2147483647.0
#define     MAX_FITNESS      10.0

#define     NORMAL_ARGC      3

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


static char  *USEAGE = "\nProgram: dm_gen.c 1.8 3/28/91 \n\
\nUSAGE: %s <max generations> [-p popsize] [-s seed] [-f PDB file] \n\n\
\tProgram uses Genetic Algorithm techniques to minimize the\n\
\tdifference between a protein backbone Ca distance matrix and\n\
\ta modelled Ca distance matrix. The program does this by mutating\n\
\tand crossing over a list of backbone Phi-Psi-Omega triples,\n\
\tbuilding a protein backbone and comparing the resulting distance\n\
\tmatrix to the target. The parameters are as follows:\n\n\
\t<max generations> the number of generations to cycle.\n\
\t[-p popsize]: the population size.\n\
\t[-s seed]: random number seed.\n\
\t[-f filename]: input Brookhaven databank format file.\n\n\
\tIf you don't specify these flags, the program will use a\n\
\tpopulation size of 50, a seed of 12345 and filename tst.pdb.\n\
\tThe output is written to the standard output, so it may be \n\
\tredirected at will.\n\n\
\tAuthor: Eric G. Suchanek, Ph.D. \n\
\tCopyright 1991, -egs-. all rights reserved.. \n";

TURTLE THE_TURTLE;                  /* global turtle for building structures */


static PHIPSI *oldpop[MAX_POPSIZE];     /* array of pointers to PHIPSIs */
static PHIPSI *newpop[MAX_POPSIZE];
/* target PHIPSI list for structure */
static PHIPSI *phipsi_target = ((PHIPSI *)NULL);


static DISTMAT *dm_target;          /* distmat for the target structure */
static DISTMAT *dm_test;            /* current test distmat */

static ATOM    *atom_list;          /* atom list for the structure */
static ATOM    *backbone;           /* Ca backbone for the structure */

static double fitness[MAX_POPSIZE]; /* storage for fitness of pop   */

static double pcross        = .7;        /* probability of a cross    */
static double pmutation     = .0025;       /* probability of a mutation */

static int GENELENGTH       = 0;

void determinefitness();
void initializepopulation();

void parse_args();
void print_population();
void sort_population();
void print_sorted_population();
void setup_turtle();

double compute_avg_fitness();

void compare_distmats();
void cleanup();



/* a structure used for sorting */

/*
struct keys {
  int index;
  double energy;
};

struct keys key_pop[MAX_POPSIZE];
*/

KEYPOP *key_pop;

#define NUMELEM(a) (sizeof(a)/sizeof(a[0]))
#define _ps_range ((int)(MAXVAL - MINVAL + 1))
#define CHOOSE ((rand() % ((int) MAXVAL - MINVAL + 1) + MINVAL))

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
   int myseed         = 0;      /* db index of disulfide we're fitting */
   int select1;
   int select2;
   double avgfitness;
   char fname[255];

   register int i,j,k;		  /* work variables */
   int lores, hires;              /* residue range vars */
   int phipsi_len;
   double phi, psi, omega;

/* never returns if there's trouble */
   parse_args(argc,argv,&maxgen,&myseed,fname,&POPSIZE); 

   i = ATOM_read_pdb(&atom_list,fname);
#ifdef NOISY
   fprintf(stderr,"-->Read %d atoms from %s\n",i,fname);
#endif
   if (i <= 0)
     {
       fprintf(stderr,"\nCan't read structure <%s>?\n",fname);
       exit(-1);
     }

   if ((i = RESIDUE_phi_psi_list_build(atom_list, &phipsi_target)) <= 0)
   {
     fprintf(stderr,"\nCan't build target phipsi list?\n");
     cleanup();
     exit(-1);
   }

#ifdef NOISY
   fprintf(stderr,"--->Phi-Psi list built.\n");
#endif

   phipsi_len = PHIPSI_list_length(phipsi_target);
   GENELENGTH = phipsi_len;

   if (phipsi_len <= 0)
     {
       fprintf(stderr,"\nCan't get PHIPSI list length for target?\n");
       cleanup();
       exit(-1);
     }

   dm_target = DISTMAT_build_from_atom_list(atom_list, fname, MAX_CUTOFF, 1);
   if (dm_target == (DISTMAT *)NULL)
     {
       fprintf(stderr,"\nCan't build target distance matrix?\n");
       cleanup();
       exit(-1);
     }

#ifdef NOISY
   fprintf(stderr,"--->DistMat 1 built.\n");
#endif

   
   RESIDUE_range(atom_list, &lores, &hires);
   setup_turtle(lores+1, atom_list);

   srand(myseed);

   /* ------- initialize the population ----------- */

   initializepopulation(phipsi_len);


   avgfitness = compute_avg_fitness();

#ifdef NOISY
   fprintf(stderr,"--->Population initialized.\n");
#endif

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
	   /* first choose two to cross ---------------- */

	   select1 = select();
	   select2 = select();

	   i = newpopsize;
	   newpopsize = newpopsize + 2;

	   /* -------------- crossover         --------- */


	   if (rand()/BIGNUM < pcross) /* determine if crossover */
	     {
	       ncross++;
	       k = (rand() % (long)GENELENGTH);	/*choose allele to split gene*/
#ifdef dbg4
	       printf("\nCrossing %d with %d\nNew bef.:\n",select1,select2);
	       PHIPSI_list_print(newpop[i]);
#endif
	       for (j=0;j<k;j++)
		 {
		   newpop[i][j] = oldpop[select1][j];
		 }
	       for (j=k;j<GENELENGTH;j++)
		 {
		   newpop[i][j] = oldpop[select2][j];
		 }
	       for (j=0;j<k;j++)
		 {
		   newpop[i+1][j] = oldpop[select2][j];
		 }
	       for (j=k;j<GENELENGTH;j++)
		 {
		   newpop[i+1][j] = oldpop[select1][j];
		 }
#ifdef dbg4
	       printf("\nNew aft.:\n");
	       PHIPSI_list_print(newpop[i]);
#endif
	     }
	   else
	     for (j=0;j<GENELENGTH;j++)	/* copy untouched */
	       { 
		 newpop[i][j] = oldpop[select1][j];
		 newpop[i+1][j] = oldpop[select2][j];
	       }

	   for (j=0;j<GENELENGTH;j++) /* handle mutation  */
	     {
	       if (rand()/BIGNUM < pmutation)
		 {
		   nmutation++;
		   PHIPSI_choose_random_dihedrals(&phi,&psi,&omega);
		   newpop[i][j].phi = phi;
		   newpop[i][j].psi = psi;
		   newpop[i][j].omega = omega;
		 }

	       if (rand()/BIGNUM < pmutation)
		 {
		   nmutation++;
		   PHIPSI_choose_random_dihedrals(&phi,&psi,&omega);
		   newpop[i+1][j].phi = phi;
		   newpop[i+1][j].psi = psi;
		   newpop[i+1][j].omega = omega;
		 }
	     }
	 }			/* end loop over members of generation  */

       for (i=0;i<POPSIZE;i++)
	 {
	   for (j=0;j<GENELENGTH;j++)
	     {
	       oldpop[i][j] = newpop[i][j];
	     }
#ifdef dbg3
	   printf("Oldpop Member: %d\n",i);
	   PHIPSI_list_print(oldpop[i]);
	   printf("---------------------------------------\n");
	   printf(" Newpop Member: %d\n",i);
	   PHIPSI_list_print(newpop[i]);
	   printf("---------------------------------------\n");
#endif
	 }

       oldpopsize = newpopsize;

     }				/* end, loop over generation production */

#ifdef NOISY
   fprintf(stderr,"\n-->Sorting population.\n");
#endif

   sort_population(oldpop,1);   /* sort into ascending order */
   print_sorted_population(oldpop);

#ifdef NOISY
   fprintf(stderr,"-->Comparing Distmats.\n");
#endif

   compare_distmats(key_pop[0].index);

   cleanup();
}			/* end, main */

/************************** auxilliary functions *****************************/

void parse_args(argc,argv,gen,index,fname,popsize)
int argc;
char *argv[];
int *gen;
int *index;
char *fname;
int *popsize;
{
  int pop = 0;
  char myfile[255];
  int i = 0;

  if (argc == 1) {
    fprintf(stderr,USEAGE,argv[0]);
    exit(-1);
  }

  *gen = atoi(argv[1]);
  if (*gen > MAXGEN) {
    fprintf(stderr,"\nGeneration count of: %d too large, set to: %d\n",
	    *gen, MAXGEN);
    *gen = MAXGEN;
  }
  

  /* default seed */
  *index = (int)123;
    
  /* default filename */
  strcpy(fname,"tst.pdb");

  /* default population size */
  *popsize = (int)(MAX_POPSIZE);

  for (i = 2; i < argc && (*argv[i] == '-'); ++i) {
	switch(*(argv[i]+1)) {
	case 's':                  /* seed */
	  *index = atoi(argv[++i]);
	  break;
	case 'p':                  /* popsize */
	  sscanf(argv[++i]," %d",&pop);
	  if (pop % 2 != 0)
	    {
	      pop -= 1;
	      fprintf(stderr,"\n-->WARNING: Population size changed to %d\n",pop);
	    }

	  if (pop > MAX_POPSIZE) pop = MAX_POPSIZE;
	  *popsize = pop;
	  break;
	case 'f':
	  sscanf(argv[++i], " %s", myfile);
	  strcpy(fname,myfile);
	  break;
	default:
	  fprintf(stderr,"Parse error: Argument %s not understood exiting.\n",
		  argv[i]);
	  fprintf(stderr,USEAGE,argv[0]);
	  exit(-1);
	  break;
        }	/* switch */
      }		/* for    */ 
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
  return(rand() % _ps_range) + MINVAL;
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

int select()
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
  double fitval = 0.0;

  TURTLE    turt;
  DISTMAT   *dm = NULL;

  /*
    aren't using the turtle yet, but would use it if comparing backbone
    structures...
  */

  TURTLE_init(&turt,"turtle");
  TURTLE_copy_coords(&turt,&THE_TURTLE); /* turtle in proper orientation */

  /* remember, this function builds from the second residue, not the 
     first, since this is in general undefined in phi-psi space */

  dm = DISTMAT_build_from_phipsi_list(oldpop[i],"tmp", MAX_CUTOFF);
  fitval = DISTMAT_rms(dm, dm_target);

  fitness[i] = -fitval;
  free_DISTMAT(dm);
}

void compare_distmats(i)
     int i;
{
  double fitval = 0.0;

  TURTLE    turt;
  DISTMAT   *dm = NULL;
  DISTMAT   *dm_diff = NULL;

  /* remember, this function builds from the second residue, not the 
     first, since this is in general undefined in phi-psi space */

  dm = DISTMAT_build_from_phipsi_list(oldpop[i],"Best Model", MAX_CUTOFF);
  fitval = DISTMAT_rms(dm, dm_target);

  dm_diff = DISTMAT_difference(dm_target,dm, "Best - Target");
  if (dm_diff == (DISTMAT *)NULL)
    {
      fprintf(stderr,"\nCan't get distmat difference?\n");
      if (dm != (DISTMAT *)NULL)
	free_DISTMAT(dm);
      return;
    }

  printf("\n------------------------------------------------------\n");
  printf("Distance Matrix Summary Results.\n");
  printf("Original distance matrix:\n");
  DISTMAT_tab_print(dm_target);
  printf("\n------------------------------------------------------\n");
  printf("Best distance matrix: (RMS Error: %8.3lf)\n",fitval);
  DISTMAT_tab_print(dm);
  printf("\n------------------------------------------------------\n");
  printf("Distance matrix difference:\n");
  DISTMAT_tab_print(dm_diff);
  printf("\n------------------------------------------------------\n");

  free_DISTMAT(dm);
  free_DISTMAT(dm_diff);
}

/* -------------------- Initialize Population -----------------------------*/
/* ------------------------------------------------------------------------*/
void initializepopulation(len)
     int len;
{
  int i;

  key_pop = new_KEYPOP(len);

  for (i=0;i<POPSIZE;i++) 
    {
      fitness[i] = -999.9;

      oldpop[i] = new_PHIPSI_list(len);
      if (oldpop[i] == (PHIPSI *)NULL)
	{
	  fprintf(stderr,"\nCan't allocate oldpop PHIPSI %d?\n",i);
	  cleanup();
	  exit(-2);
	}
      PHIPSI_random_init(oldpop[i]);

      newpop[i] = (PHIPSI *)new_PHIPSI_list(len);
      if (newpop[i] == (PHIPSI *)NULL)
	{
	  fprintf(stderr,"\nCan't allocate newpop PHIPSI %d?\n",i);
	  cleanup();
	  exit(-2);
	}
      PHIPSI_random_init(newpop[i]);

      determinefitness(i);

#ifdef NOISY
      fprintf(stderr,"---->Member %d initialized.\r",i);
#endif

    }

#ifdef NOISY
      fprintf(stderr,"\n");
#endif
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
PHIPSI *population[];
{
  register int i;

  for (i = 0; i < POPSIZE; i++) {
    determinefitness(i);
    printf("%4d ",i);
    PHIPSI_list_print(population[i]);
    printf("\n----------------------------------------------\n");
    printf("Fitness: %8.4lf\n--------------------------------\n", 
	   fitness[i]);
  }
}

/* sorting the population by energy */

void sort_population(population,updown)
PHIPSI *population[];
int updown;
{
  register int i;

  for (i = 0; i < POPSIZE; i++) {
    determinefitness(i);
    key_pop[i].index = i;

    /* change sign back into real units */
    key_pop[i].energy = -fitness[i]; 
  }
KEYPOP_sort(key_pop,POPSIZE,updown);

}


/* print the sorted population */

void print_sorted_population(population)
PHIPSI *population[];
{
  register int i;
  int index;
  double energy;

  printf("\n**************** Sorted PHI-Psi Array *****************\n\n");
  for (i = 0; i < POPSIZE; i++) {
    index = key_pop[i].index;
    energy = key_pop[i].energy;
    printf("Index: %4d ", index);
    printf(" Energy %8.4lf\n", energy);
  }

  printf("------------------------------------------------------\n");
  printf("PHI-Psi list for BEST fit (index %d) Fitness: %8.4lf \n",
	 key_pop[0].index,key_pop[0].energy);
  PHIPSI_list_print(oldpop[key_pop[0].index]);

  printf("------------------------------------------------------\n");
  printf("PHI-Psi list for starting structure:\n");
  PHIPSI_list_print(phipsi_target);
}


/* 
  routine orients the global turtle correctly to allow disulfides to be made
  in the same orientation as that found in the disulfide database
  (orientation #2), with position at Ca, heading toward C, with N on
  the left...
*/

void setup_turtle(lores,atoms)
     int lores;
     ATOM *atoms;
{
  int ok = 0;

  ok = orient_at_residue(&THE_TURTLE, atoms, lores, ORIENT_BACKBONE);
  if (!ok)
    {
      fprintf(stderr,"\nCan't orient at residue %d?\n", lores);
      cleanup();
      exit(-1);
    }
}


void cleanup()
{
  int i;

  if (atom_list)
    free_ATOM_list(atom_list);
  if (backbone)
    free_ATOM_list(backbone);
  if (dm_target)
    free_DISTMAT(dm_target);
  if (dm_test)
    free_DISTMAT(dm_test);
  if (phipsi_target)
    free_PHIPSI_list(phipsi_target);
  if (key_pop)
    free_KEYPOP(key_pop);
  
  for (i = 0; i < POPSIZE; i++)
    {
      free_PHIPSI_list(&oldpop[i]);
      free_PHIPSI_list(&newpop[i]);
    }
}


