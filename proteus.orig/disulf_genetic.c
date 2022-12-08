/* --- Source Code: Genetic.c --- */
/* $Id: disulf_genetic.c,v 1.4 1995/06/20 02:59:56 suchanek Exp $ */
static char rcsid[] = "$Id: disulf_genetic.c,v 1.4 1995/06/20 02:59:56 suchanek Exp $";

#include    <stdio.h>
#include    <string.h>
#include    <math.h>
#include    <stdlib.h>

#include    "disulfide.h"

#define     POPSIZE          150
#define     GENELENGTH       5
#define     MAXGEN           2000
#define     MINVAL           -180
#define     MAXVAL            179
#define     BIGNUM           2147483647.0

#define     NORMAL_ARGC      3
#define     MAX_ENTRIES      99

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

#define DB_HOME "/home/suchanek/ss/ss_patterns.dat"
static char  *USEAGE = "\nProgram: disulf_genetic.c 1.5 3/29/91 \n\n\
\nUSAGE: %s <max generations> <index> \n\n\
\tProgram uses Genetic Algorithm techniques to minimize the\n\
\tenergy of a five-dimensional energy surface, specifically\n\
\tthe conformational energy of Disulfide Bond, a covalent protein\n\
\tstabilizing structural element. The optional parameter \n\
\t<max generations> specifies how many generations of mutations \n\
\tand crossovers to cycle before stopping. The sorted list of \n\
\tconformations is written to the standard output, so it may be \n\
\tredirected at will.\n\n\
\n\t Author: Eric G. Suchanek, Ph.D. \n\
\n\t Copyright 1995, -egs-, all rights reserved.. \n\n";

DISULFIDE SS_array[100];            /* used to hold database disulfides */
DISULFIDE SS_target;                /* the disulfide we're trying to fit */
TURTLE THE_TURTLE;                  /* global turtle for building SS bonds */

static int db_entries       = 0;    /* number of entries in database */

static double oldpop[POPSIZE][GENELENGTH];   /* storage for populations */
static double newpop[POPSIZE][GENELENGTH];
static double fitness[POPSIZE];              /* storage for fitness of pop   */


static double pcross        = .6;        /* probability of a cross    */
static double pmutation     = .03;       /* probability of a mutation */

void determinefitness();
void initializepopulation();

void parse_args();
void print_population();
void sort_population();
void print_sorted_population();
void setup_turtle();

double compute_avg_fitness();


/* a structure used for sorting */

struct keys {
  int index;
  double energy;
};

struct keys key_pop[POPSIZE];


#define NUMELEM(a) (sizeof(a)/sizeof(a[0]))
#define range ((int)(MAXVAL - MINVAL + 1))
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
   int db_index         = 0;      /* db index of disulfide we're fitting */
   int select1;
   int select2;
   double avgfitness;

   register int i,j,k;		  /* work variables */

   /* ------- initialize the population ----------- */


/* never returns if there's trouble */
   parse_args(argc,argv,&maxgen,&db_index); 

   db_entries = DISULFIDE_read_disulfide_database(DB_HOME, 
					&SS_array[0],100);
   if (db_entries <= 0) {
     fprintf(stderr,"\nCan't read disulfide database ss_patterns.dat\n");
     exit(-1);
   }
   else
     printf("Read <%d> entries from SS database.\n",db_entries);
     
   SS_target = SS_array[db_index];
   printf("Target Disulfide:\n");
   DISULFIDE_print(&SS_target);

   srand(db_index + 100);

   setup_turtle();

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
	       if (rand()/BIGNUM < pmutation)
		 {
		   nmutation++;
		   newpop[i][j] =  CHOOSE;
		 }

	       if (rand()/BIGNUM < pmutation)
		 {
		   nmutation++;
		   newpop[i+1][j] = CHOOSE;
		 }
	     }
	 }			/* end loop over members of generation  */

       for (i=0;i<POPSIZE;i++)
	 for (j=0;j<GENELENGTH;j++)
	   oldpop[i][j] = newpop[i][j];

       oldpopsize = newpopsize;
     }				/* end, loop over generation production */

   sort_population(oldpop,1);   /* sort into ascending order */

   print_sorted_population(oldpop);

 }			/* end, main */

/************************** auxilliary functions *****************************/

void parse_args(argc,argv,gen,index)
int argc;
char *argv[];
int *gen;
int *index;
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
  
  *index = atoi(argv[2]);
  if (*index < 0 || *index > MAX_ENTRIES) {
    fprintf(stderr,"Error, index <%d> out of range, using 0\n",*index);
    *index = 0;
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
  double energy;
  double rms_pos = 0.0;
  double rms_conf = 0.0;

  DISULFIDE SStmp;
  TURTLE    turt;

  DISULFIDE_init(&SStmp,"tmp");
  TURTLE_init(&turt,"turtle");

  TURTLE_copy_coords(&turt,&THE_TURTLE); /* turtle in proper orientation */

  x1 = oldpop[i][0]; x2 = oldpop[i][1]; x3 = oldpop[i][2]; 
  x4 = oldpop[i][3]; x5 = oldpop[i][4];

  DISULFIDE_set_conformation(&SStmp,x1,x2,x3,x4,x5);
  DISULFIDE_build(&SStmp, turt);
  energy = SStmp.energy;   /* computed by build_disulfide */

  rms_pos = DISULFIDE_compare_backbones(&SStmp,&SS_target);
  rms_pos = rms_pos == 0.0 ? -50.0 : rms_pos;

/*  rms_conf = DISULFIDE_compare_dihedrals(&SStmp,&SS_target); */
  fitness[i] = -1.0 * (rms_pos + energy + rms_conf);

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

void print_sorted_population(population)
double population[][GENELENGTH];
{
  register int i,j;
  int index;
  double energy;

  printf("\n**************** Sorted Chromosome Array *****************\n\n");
  printf("Member  Chi1     Chi2     Chi3     Chi4     Chi5    Fitness \n");
  printf("----------------------------------------------------------\n");

  for (i = 0; i < POPSIZE; i++) {
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
  in the same orientation as that found in the disulfide database
  (orientation #2), with position at Ca, heading toward C, with N on
  the left...
*/

void setup_turtle()
{
  VECTOR n, ca, c;

  TURTLE_init(&THE_TURTLE,"The turtle is here");
  VECTOR_init(&n, -0.486, 1.366, 0.00);
  VECTOR_init(&ca, 0.0, 0.0, 0.0);
  VECTOR_init(&c, 1.53, 0.0, 0.0);
  
  TURTLE_orient(&THE_TURTLE, &ca, &c, &n);
  bbone_to_schain(&THE_TURTLE);
}

