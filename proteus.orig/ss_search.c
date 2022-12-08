/****************************************************************************/
/* Yet another incarnation of the now famous Suchanek & Pabo disulfide      */
/* searching algorithm, written with Eric G. Suchanek's C version of        */
/* Proteus.  This version borrows liberally from the Lisp code, but has a   */
/* number of improvements including better VDW checking, energy calculation,*/
/* and genetic-algorithm based conformational optimization.                 */

/* $Id: ss_search.c,v 1.12 1995/11/07 15:44:13 suchanek Exp suchanek $ */

/* $Log: ss_search.c,v $
 * Revision 1.12  1995/11/07  15:44:13  suchanek
 * changed useage message to get rid of the genetic algo stuff
 *
 * Revision 1.11  1995/06/25  02:44:30  suchanek
 * added init_sincostables call to use the new 'quick' turtle routines
 *
 * Revision 1.10  1995/06/25  00:29:34  suchanek
 * added logic to get SS_DBNAME environ variable and backwards building
 * for disulfides
 *
 * Revision 1.9  1995/06/23  11:26:44  suchanek
 * *** empty log message ***
 *
 * Revision 1.8  1995/06/23  00:10:12  suchanek
 * changed to use DISTMAT_build_from_backbone_list for speedup
 *
 * Revision 1.7  1995/06/22  20:38:15  suchanek
 * added logic to use backbone functions, mucho speedup
 *
 * Revision 1.6  1995/06/20  02:59:26  suchanek
 * function renaming
 *
 * Revision 1.5  1995/06/20  01:11:36  suchanek
 * added more optimizations with distance matrix checks
 *
 * Revision 1.4  1995/06/12  03:56:56  suchanek
 * additional code, error checking
 *
 * Revision 1.3  1995/06/11  05:12:32  suchanek
 * actual working version, first cut
 *
 * Revision 1.2  1995/06/07  15:32:05  suchanek
 * *** empty log message ***
 *
 * Revision 1.1  1995/06/06  20:26:18  suchanek
 * Initial revision
 * */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* include relevant Proteus header files */

#include "residue.h"
#include "distmat.h"
#include "disulfide.h"

/* some genetic algorithm parameters and variables */
#define     POPSIZE          150           /* population size */
#define     GENELENGTH       5             /* one for every dihedral */
#define     MAXGEN           2000          /* maximum number of generations */
#define     DEFAULT_GEN      100           /* default number of generations */
#define     MINVAL           -180          /* -180 degrees */
#define     MAXVAL            179          /* 179 degrees */
#define     BIGNUM           2147483647.0  /* for the random number gen. */

#define     SHOW             0
#define     PAUSE            0

#define     MAX_HITS 500     /* maximum number of potential hits allowed */
#define     SKIP 10          /* print every SKIP residues */

static double oldpop[POPSIZE][GENELENGTH];   /* storage for populations */
static double newpop[POPSIZE][GENELENGTH];
static double fitness[POPSIZE];              /* storage for fitness of pop */


static double pcross        = .6;        /* probability of a cross    */
static double pmutation     = .03;       /* probability of a mutation */

/* prototypes */

void cleanup(int result_code);

void determinefitness(int i);
void initializepopulation(void);

void parse_args(int argc, char *argv[], int *gen, char *pdbfile, double *cutoff);
void print_population(double population[][GENELENGTH]);
void sort_population(double population[][GENELENGTH],int updown);
void print_sorted_population(double population[][GENELENGTH]);
void setup_turtle(void);
void print_potential_ss(SS_HIT *potential_ss, int hits, DISULFIDE *ss_array);
double compute_avg_fitness(void);


/* a structure used for sorting */

struct _keys {
  int index;
  double energy;
};

struct _keys key_pop[POPSIZE];

SS_HIT potential_ss[MAX_HITS];


#define range ((int)(MAXVAL - MINVAL + 1))
#define CHOOSE ((rand() % ((int) MAXVAL - MINVAL + 1) + MINVAL))



/* argument list */
#define     MINIMUM_ARGC      3
#define     NORMAL_ARGC      4

static char  *USEAGE = "\nProgram: ss_search \n\n\
\nUSAGE: %s <pdb filename> cutoff\n\n\
\tProgram uses a database search technique to locate potential\n\
\tsites for placement of disulfide bonds in proteins of known\n\
\tstructure.\n\n\
\tMake sure you have the environment variable SS_DBNAME assigned\n\
\tto the location of the ss_patterns.dat database.\n\
\n\t Author: Eric G. Suchanek, Ph.D. \n\
\n\t Copyright 1995, -egs-, all rights reserved.. \n\n";

/*
static char  *USEAGE = "\nProgram: ss_search \n\n\
\nUSAGE: %s <pdb filename> max_generations cutoff\n\n\
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
*/


DISULFIDE *SS_array = NULL; 	        /* the SS database */
DISULFIDE *SS_hits = NULL;              /* list of hits */
DISULFIDE SS_target;                    /* for building disulfides */
TURTLE    THE_TURTLE;                   /* the turtle - duh. */

DISTMAT *ca_distmat;

static int db_entries = 0;

static ATOM *protein_atoms  = NULL;     /* pointer to the atom list */
static BACKBONE *backbone_atoms = NULL; /* pointer to the backbone atoms */


void print_potential_ss(SS_HIT *potential_ss, int hits, DISULFIDE *ss_array)
{

    int elem = NUMELEM(potential_ss);
    int i = 0;
    int index = 0;

    if (hits == 0)
	return;

    printf("Potential Disulfides\nProximal     Distal          Error     Name\n");
    printf("******************************************************\n");
    for (i = 0; i < hits; i++)
	{
	    index = potential_ss[i].index;
	    printf("%5d        %5d        %7.2lf     %s\n",
		   potential_ss[i].proximal, potential_ss[i].distal,
		   potential_ss[i].rms_error, ss_array[index].name);

	}
}


void genetic_optimize(int maxgen)
{
   int gen              = 0;      /* generation counter        */
   int nmutation        = 0;      /* number of mutations       */
   int ncross           = 0;      /* number of xovers to date  */
   int oldpopsize       = 0;      /* total population          */
   int newpopsize       = 0;
   int db_index         = 0;      /* db index of disulfide we're fitting */
   int select1;
   int select2;

   double avgfitness    = -999.9; /* average 'fitness' of the current population */

}


/* ========================== MAIN PROGRAM ===============================*/

void main (int argc, char *argv[])
{  
    int i = 0;
    int maxgen          = 0;   /* maximum number of generations */

    char *ss_filename;       /* disulfide database name */

    char pdb_filename[80];      /* working PDB filename */
    int natoms           = 0;   /* number of atoms in the protein */
    int total_residues   = 0;   /* total residues in the protein */
    int lores            = 0;   /* starting residue number */
    int hires            = 0;   /* ending residue number */
    int proximal         = 0;   /* current proximal residue */
    int distal           = 0;   /* current distal residue */
    int hits             = 0;   /* number of potential disulfides */

    DISULFIDE *ss_model = NULL; /* current disulfide model */
    
    double error;
    double cutoff;
    double min_dist      = 0.0; /* minimum SS ca-ca distance */
    double max_dist      = 0.0; /* maximum SS ca-ca distance */
    double current_dist  = 0.0; /* current ca-ca distance */

    int ss_index         = 0;   /* current disulfide index */
    
    ATOM *n, *ca, *c, *o, *cb;  /* working atomic positions */

    int do_cleanup = 0;         /* true if we need to free the cb atom */

    int res_done = 0;           /* number of residues done so far */

    double j;
    int myindex;

    /* never returns if there's trouble */
    parse_args(argc,argv,&maxgen,pdb_filename, &cutoff); 

    init_sincostables();

#ifdef NOISY2
    for (j = -180.0; j < 179.9; j+= .15)
	{
	    myindex = get_sincos_index(j);
	    printf("J: %lf, myindex: %d, sin: %lf, realsin: %lf \n",
		   j, myindex, qsin(myindex), sin((DEGRAD(j))));
	}
#endif

    SS_array = new_DISULFIDE_list(MAX_SS_NUMBER);
    if (SS_array == (DISULFIDE *)NULL)
	{
	    fprintf(stderr,">>>>>>>> ss_search: Error! Can't allocate SS_array! \n");
	    cleanup(5);
	}

    ss_filename = getenv("SS_DBNAME");
    printf("name: %s\n",ss_filename);

    if (ss_filename)
	db_entries = DISULFIDE_read_disulfide_database(ss_filename, 
						       &SS_array[0],MAX_SS_NUMBER);
    else
	{
	    printf("You need to set the envrionment variable SS_DBNAME to point to \
                    the disulfide database. \n");
	    cleanup(5);
	}

	
    if (db_entries <= 0) {
	fprintf(stderr,">>>>>>>> ss_search: Error! Can't read disulfide database ss_patterns.dat\n");
	exit(-1);
    }
    else
	fprintf(stderr,"Read %d entries from SS database.\n",db_entries);
    
    
    DISULFIDE_scan_database(&SS_array[0], db_entries, &min_dist, &max_dist);

#ifdef NOISY
    fprintf(stderr,"Min dist: %lf, Max dist: %lf \n", min_dist, max_dist);
#endif

    /* read the protein file */
    natoms = ATOM_read_pdb(&protein_atoms,pdb_filename);
    if (natoms <= 0) 
	{
	    fprintf(stderr,">>>>>>>> ss_search: Error! Can't read file <%s>! Exiting! \n", pdb_filename);
	    cleanup(20);
	}

#ifdef NOISY1
    ATOM_list_print(protein_atoms);
#endif
    
    /* now get the residue range for subsequent searches */
    total_residues = RESIDUE_range(protein_atoms, &lores, &hires);

    if (total_residues <= 0)
	{
	    fprintf(stderr,">>>>>>>> ss_search: Error! Can't parse residue range! Exiting!\n");
	    cleanup(30);
	}
    
    printf("Totals: Residues: %d, Low: %d High: %d Atoms: %d \n", total_residues, lores, hires, natoms);
    
#ifdef NOISY2
    DISTMAT_print(ca_distmat);
#endif    
    

    if ((backbone_atoms = BACKBONE_create(protein_atoms)) == (BACKBONE *)NULL)
	{
	    fprintf(stderr,"Can't build backbone??\n");
	    cleanup(40);
	}

    
#ifdef NOISY2
    BACKBONE_print(backbone_atoms, lores, hires);
#endif

    printf("Building distance matrix...\n");

    ca_distmat = DISTMAT_build_from_backbone_list(backbone_atoms, pdb_filename, max_dist+.5, lores, hires);
    if (ca_distmat == (DISTMAT *)NULL)
	{
	    fprintf(stderr,"\nCan't build target distance matrix?\n");
	    cleanup(30);
	}

    /* do the search */
    
    for (proximal = lores; proximal <= hires; proximal++)
	{
	    if (++res_done % SKIP == 0)
		printf("Proximal: %d \n", proximal);
	    for (distal = lores; distal <= hires; distal++)
		{

		    if (proximal == distal)
			continue;

		    current_dist = DISTMAT_get(ca_distmat,proximal, distal);

		    if (current_dist > max_dist || current_dist < min_dist)
			{
#ifdef NOISY2
			    fprintf(stderr,"Skipped %d, %d, distance: %lf \n",
				    proximal, distal, current_dist);
#endif
			    continue;
			}


		    BACKBONE_get_backbone(backbone_atoms, distal, lores,
					 &n, &ca, &c, &o, &cb);


		    for (ss_index = 0; ss_index < db_entries; ss_index++)

			{
#ifdef NOISY2
			    printf("Proximal: %d Distal: %d Index: %d \n",
				   proximal, distal, ss_index);
#endif

			    /* build it forwards */
			    if ((ss_model = DISULFIDE_build_at_residue(protein_atoms,
								       backbone_atoms,
								       proximal, 
								       lores,
								       &SS_array[ss_index]))
				== (DISULFIDE *)NULL)
				{
				    fprintf(stderr,"Can't build model disulfide?");
				    cleanup(15);
				}

			    error = 0.0;
			    error = DISULFIDE_compare_to_backbone_atoms(ss_model, 
								n, ca, c, cb);

			    free_DISULFIDE(ss_model);


			    if (error <= cutoff)
				{
				    potential_ss[hits].proximal = proximal;
				    potential_ss[hits].distal = distal;
				    potential_ss[hits].index = ss_index;
				    potential_ss[hits].rms_error = error;
				    if (++hits >= MAX_HITS)
					{
					    fprintf(stderr,"Max SS hits found.  Breaking out of search \n");
					    break;
					}
				}

			    /* build it backwards */
			    if ((ss_model = DISULFIDE_build_at_residue_backwards(protein_atoms,
								       backbone_atoms,
								       proximal, 
								       lores,
								       &SS_array[ss_index]))
				== (DISULFIDE *)NULL)
				{
				    fprintf(stderr,"Can't build model disulfide?");
				    cleanup(15);
				}

			    error = 0.0;
			    error = DISULFIDE_compare_to_backbone_atoms(ss_model, 
								n, ca, c, cb);

			    free_DISULFIDE(ss_model);


			    if (error <= cutoff)
				{
				    potential_ss[hits].proximal = proximal;
				    potential_ss[hits].distal = distal;
				    potential_ss[hits].index = ss_index;
				    potential_ss[hits].rms_error = error;
				    if (++hits >= MAX_HITS)
					{
					    fprintf(stderr,"Max SS hits found.  Breaking out of search \n");
					    break;
					}
				}


			} /* for ss_index */

		} /* for distal */
	} /* for proximal */

    if (hits > 0)
	print_potential_ss(potential_ss, hits, &SS_array[0]);
    /* all done, clean up global resources */
    cleanup(0);
    
}


void cleanup(int result_code)
{
    /* clean up after ourselves */
    
   if (SS_array)
	free_DISULFIDE_list(SS_array);

    if (protein_atoms)
	free_ATOM_list(protein_atoms);

    if (backbone_atoms)
	free_BACKBONE_list(backbone_atoms);

    if (ca_distmat)
	free_DISTMAT(ca_distmat);
    
    exit(result_code);
    
}


/************************** auxilliary functions *****************************/

void parse_args(int argc, char *argv[], int *gen, char *pdbfile, double *cutoff)
{
    if (argc < MINIMUM_ARGC) 
	{
	    fprintf(stderr,USEAGE,argv[0]);
	    exit(-1);
	}
    
    strcpy(pdbfile,argv[1]);
/*    *cutoff = atof(argv[3]); */
    *cutoff = atof(argv[2]);

    if (argc == NORMAL_ARGC) 
	{
/*	    *gen = atoi(argv[2]); */
	    if (*gen > MAXGEN) 
		{
		    fprintf(stderr,">>>>>>>> ss_search: Warning! Generation count of: %d too large, set to: %d\n",
			    *gen, MAXGEN);
		    *gen = MAXGEN;
		}
    	}
    else
	*gen = DEFAULT_GEN;   /* use some reasonable default */
    
}

/************************** genetic algorithm auxilliary functions *********************/

double compute_avg_fitness(void) {
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

void determinefitness(int i)
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
void initializepopulation(void)
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

void print_population(double population[][GENELENGTH])
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

void sort_population(double population[][GENELENGTH],int updown)
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
    qsort(&key_pop[0], POPSIZE, sizeof(struct _keys), cmp_keysa);
  else
    qsort(&key_pop[0], POPSIZE, sizeof(struct _keys), cmp_keysd);
}

/* sort the keys structure into descending order */

int cmp_keysd(key1, key2)
struct _keys *key1, *key2;
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
struct _keys *key1, *key2;
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

void print_sorted_population(double population[][GENELENGTH])
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

void setup_turtle(void)
{
  VECTOR n, ca, c;

  TURTLE_init(&THE_TURTLE,"The turtle is here");
  VECTOR_init(&n, -0.486, 1.366, 0.00);
  VECTOR_init(&ca, 0.0, 0.0, 0.0);
  VECTOR_init(&c, 1.53, 0.0, 0.0);
  
  TURTLE_orient(&THE_TURTLE, &ca, &c, &n);
  bbone_to_schain(&THE_TURTLE);
}




