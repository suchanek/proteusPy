/*  $Id: disulfide.c,v 1.9 1995/06/25 00:28:44 suchanek Exp $  */
/*  $Log: disulfide.c,v $
 * Revision 1.9  1995/06/25  00:28:44  suchanek
 * added DISULFIDE_build_at_residue_backwards function
 *
 * Revision 1.8  1995/06/23  00:09:32  suchanek
 * changed to use DISTMAT_Build_from_backbone
 *
 * Revision 1.7  1995/06/22  20:37:55  suchanek
 * additional logic to handle the backbone functions
 *
 * Revision 1.6  1995/06/20  02:59:50  suchanek
 * function renaming
 *
 * Revision 1.5  1995/06/20  01:12:31  suchanek
 * added DISULFIDE_scan_database function to get minimum and maximum distances
 *
 * Revision 1.4  1995/06/12  03:56:22  suchanek
 * added code to handle building cb atoms
 *
 * Revision 1.3  1995/06/11  05:12:48  suchanek
 * many changes, still has problem with gly
 *
 * Revision 1.2  1995/06/07  15:36:56  suchanek
 * *** empty log message ***
 *
 * Revision 1.1  1993/12/10  09:23:26  egs
 * Initial revision
 * */

static char rcsid[] = "$Id: disulfide.c,v 1.9 1995/06/25 00:28:44 suchanek Exp $";

#include "protutil.h"
#include "disulfide.h"

#define MAXVAL 179
#define MINVAL -180
#define range ((int)(MAXVAL - MINVAL + 1))

void DISULFIDE_print(DISULFIDE *SS)
{
  printf("\nDisulfide: <%s>\n\tProximal: %d \n\tDistal: %d\n",
	 SS->name, SS->prox_resnum, SS->dist_resnum);
  printf("\tConformation: %8.3lf %8.3lf %8.3lf %8.3lf %8.3lf \n",
	 SS->chi1, SS->chi2, SS->chi3, SS->chi4, SS->chi5);
  printf("\tEnergy: %8.3lf\n",SS->energy);
  printf("\tAtoms:\n");
  printf("\t\tN prox: ");
  VECTOR_print(&SS->n_prox);
  printf("\t\tCa prox: ");
  VECTOR_print(&SS->ca_prox);
  printf("\t\tC prox: ");
  VECTOR_print(&SS->c_prox);
  printf("\t\tO prox: ");
  VECTOR_print(&SS->o_prox);
  printf("\t\tCb prox: ");
  VECTOR_print(&SS->cb_prox);
  printf("\t\tSg prox: ");
  VECTOR_print(&SS->sg_prox);
  printf("\t\tSg dist: ");
  VECTOR_print(&SS->sg_dist);
  printf("\t\tCb dist: ");
  VECTOR_print(&SS->cb_dist);
  printf("\t\tCa dist: ");
  VECTOR_print(&SS->ca_dist);
  printf("\t\tN dist: ");
  VECTOR_print(&SS->n_dist);
  printf("\t\tC dist: ");
  VECTOR_print(&SS->c_dist);
  printf("\t\tO dist: ");
  VECTOR_print(&SS->o_dist);

}

/* --------------- Memory Allocation and Initialization -------------- */

/*
 * Function allocates space for an atom list of NH elements.
 * It also calls ATOM_list_init to fill in default information
 * in the structure
 */


/*
 * Function initializes an entire list of disulfide to default values.
 *
 */

void DISULFIDE_list_init(DISULFIDE *disulfide, int nh)
{
  int nl = 0;
  int i, j;

}

DISULFIDE *new_DISULFIDE_list(int nh)
{
	DISULFIDE *v = NULL;
	int nl = 0;
	v=(DISULFIDE *)malloc((unsigned) (nh-nl+1)*sizeof(DISULFIDE));
	if (!v) Proteus_error("allocation failure in DISULFIDE_list()");
	DISULFIDE_list_init(v,nh);
	return v;
}

/*
 * Function frees space allocated to an DISULFIDE list.
 *
 */

void free_DISULFIDE_list(DISULFIDE *v)
{
  if (v != (DISULFIDE *)NULL)
    free((char*) (v));
}

DISULFIDE *new_DISULFIDE(char *name)
{
	DISULFIDE *v = NULL;

	v=(DISULFIDE *)malloc((unsigned)sizeof(DISULFIDE));
	if (v == (NULL)) 
	    {
		fprintf(stderr,"allocation failure in new_DISULFIDE()");
		return ((DISULFIDE *)NULL);
	    }
	DISULFIDE_init(v,name);
	return v;
}

/*
 * Function frees space allocated to an DISULFIDE list.
 *
 */

void free_DISULFIDE(v)
DISULFIDE *v;
{
  if (v != (DISULFIDE *)NULL)
    free((char*) (v));
}



void DISULFIDE_init(DISULFIDE *SS, char *name)
{
  VECTOR orig;

  VECTOR_init(&orig, -99.0, -99.0, -99.0);

  SS->n_prox = orig;
  SS->ca_prox = orig;
  SS->c_prox = orig;
  SS->o_prox = orig;
  SS->cb_prox = orig;
  SS->sg_prox = orig;
  SS->sg_dist = orig;
  SS->cb_dist = orig;
  SS->ca_dist = orig;
  SS->n_dist = orig;
  SS->c_dist = orig;
  SS->o_dist = orig;

  SS->chi1 = SS->chi2 = SS->chi3 = SS->chi4 = SS->chi5 = -99.9;
  SS->energy = -99.9;
  SS->prox_resnum = -1;
  SS->dist_resnum = -1;

  if (name != (char *) NULL)
    strcpy(SS->name,name);
  else
    strcpy(SS->name,"Unnamed");
      
}

void DISULFIDE_set_conformation(DISULFIDE *SS, double chi1,
				double chi2, double chi3, double chi4,
				double chi5)
{
  SS->chi1 = chi1; SS->chi2 = chi2; SS->chi3 = chi3; SS->chi4 = chi4;
  SS->chi5 = chi5;
}

void DISULFIDE_set_resnum(DISULFIDE *SS, int prox, int distal)
{
  SS->prox_resnum = prox;
  SS->dist_resnum = distal;
}

void DISULFIDE_set_positions(DISULFIDE *SS, VECTOR n1, VECTOR ca1, 
			     VECTOR c1, VECTOR o1, VECTOR cb1, 
			     VECTOR sg1, VECTOR n2, VECTOR ca2,
			     VECTOR c2, VECTOR o2,VECTOR cb2, VECTOR sg2)
{
  SS->n_prox = n1; SS->ca_prox = ca1; SS->c_prox = c1; SS->o_prox = o1;
  SS->cb_prox = cb1; SS->sg_prox = sg1;

  SS->n_dist = n2; SS->ca_dist = ca2; SS->c_dist = c2; SS->o_dist = o2;
  SS->cb_dist = cb2; SS->sg_dist = sg2;

}

/*
  subroutine takes an atom list, residue number, initialized disulfide
  array and model disulfide orients at the residue, and returns a
  disulfide structure in the global coordinates of the atom list
*/

DISULFIDE *DISULFIDE_build_at_residue(ATOM *atoms, 
				      BACKBONE *bb,
				      int resnumb, 
				      int lowres,
				      DISULFIDE *model_ss)
{
    TURTLE turtle;
    DISULFIDE *ss;
    VECTOR n, ca, c, cb;
    ATOM *n_atom, *ca_atom, *c_atom, *cb_atom, *o_atom;
    ATOM *cb_tmp;

    int res = 0;
    int cleanup = 0;  /* true if we built an atom */

    TURTLE_init(&turtle,"tmp");

    /* first make a disulfide */

    if ((ss = new_DISULFIDE(model_ss->name)) == (DISULFIDE *)NULL)
	return((DISULFIDE *)NULL);

    DISULFIDE_init(ss, model_ss->name);
    /* copy conformation */
    ss->chi1 = model_ss->chi1;
    ss->chi2 = model_ss->chi2;
    ss->chi3 = model_ss->chi3;
    ss->chi4 = model_ss->chi4;
    ss->chi5 = model_ss->chi5;


    BACKBONE_get_backbone(bb, resnumb, lowres, &n_atom, &ca_atom, &c_atom, &o_atom, &cb_atom);

    ss->n_prox = n_atom->position;
    ss->ca_prox = ca_atom->position;
    ss->c_prox = c_atom->position;
    ss->o_prox = o_atom->position;
    ss->cb_prox = cb_atom->position;


    /* orient at residue for the build. if error, return NULL */


    if (n_atom == (ATOM *)NULL || ca_atom == (ATOM *)NULL ||
	c_atom == (ATOM *)NULL) 
	{
	    fprintf(stderr,
		    "DISULFIDE_build_at_residue() - error <BACKBONE> - missing atoms?\n");
	    return((DISULFIDE *)NULL);

	}

    TURTLE_orient(&turtle, &ca_atom->position, &cb_atom->position, &n_atom->position);

    /* now build it */
    TURTLE_move(&turtle, 1.53);
    TURTLE_roll(&turtle, ss->chi1);
    TURTLE_yaw(&turtle, 112.8);
/*    ss->cb_prox = turtle.position; */
    
    TURTLE_move(&turtle, 1.86);
    TURTLE_roll(&turtle, ss->chi2);
    TURTLE_yaw(&turtle, 103.8);
    ss->sg_prox = turtle.position;
    
    TURTLE_move(&turtle, 2.044);
    TURTLE_roll(&turtle, ss->chi3);
    TURTLE_yaw(&turtle, 103.8);
    ss->sg_dist = turtle.position;
    
    TURTLE_move(&turtle, 1.86);
    TURTLE_roll(&turtle, ss->chi4);
    TURTLE_yaw(&turtle, 112.8);
    ss->cb_dist = turtle.position;
    
    TURTLE_move(&turtle, 1.53);
    TURTLE_roll(&turtle, ss->chi5);
    TURTLE_pitch(&turtle, 180.0); 
    schain_to_bbone(&turtle);

    build_residue(&turtle, &n, &ca, &cb, &c);
    
    ss->n_dist = n;
    ss->ca_dist = ca;
    ss->c_dist = c;
    
/*   DISULFIDE_compute_torsional_energy(ss); */

    /* done */
    return(ss);
}

/*
  subroutine takes an atom list, residue number, initialized disulfide
  array and model disulfide orients at the residue, and returns a
  disulfide structure in the global coordinates of the atom list.
  this builds it with chi5-chi1 orientation (backwards)

*/

DISULFIDE *DISULFIDE_build_at_residue_backwards(ATOM *atoms, 
				      BACKBONE *bb,
				      int resnumb, 
				      int lowres,
				      DISULFIDE *model_ss)
{
    TURTLE turtle;
    DISULFIDE *ss;
    VECTOR n, ca, c, cb;
    ATOM *n_atom, *ca_atom, *c_atom, *cb_atom, *o_atom;
    ATOM *cb_tmp;

    int res = 0;
    int cleanup = 0;  /* true if we built an atom */

    TURTLE_init(&turtle,"tmp");

    /* first make a disulfide */

    if ((ss = new_DISULFIDE(model_ss->name)) == (DISULFIDE *)NULL)
	return((DISULFIDE *)NULL);

    DISULFIDE_init(ss, model_ss->name);
    /* copy conformation */
    ss->chi1 = model_ss->chi5;
    ss->chi2 = model_ss->chi4;
    ss->chi3 = model_ss->chi3;
    ss->chi4 = model_ss->chi2;
    ss->chi5 = model_ss->chi1;


    BACKBONE_get_backbone(bb, resnumb, lowres, &n_atom, &ca_atom, &c_atom, &o_atom, &cb_atom);

    ss->n_prox = n_atom->position;
    ss->ca_prox = ca_atom->position;
    ss->c_prox = c_atom->position;
    ss->o_prox = o_atom->position;
    ss->cb_prox = cb_atom->position;


    /* orient at residue for the build. if error, return NULL */


    if (n_atom == (ATOM *)NULL || ca_atom == (ATOM *)NULL ||
	c_atom == (ATOM *)NULL) 
	{
	    fprintf(stderr,
		    "DISULFIDE_build_at_residue() - error <BACKBONE> - missing atoms?\n");
	    return((DISULFIDE *)NULL);

	}

    TURTLE_orient(&turtle, &ca_atom->position, &cb_atom->position, &n_atom->position);

    /* now build it */
    TURTLE_move(&turtle, 1.53);
    TURTLE_roll(&turtle, ss->chi1);
    TURTLE_yaw(&turtle, 112.8);
/*    ss->cb_prox = turtle.position; */
    
    TURTLE_move(&turtle, 1.86);
    TURTLE_roll(&turtle, ss->chi2);
    TURTLE_yaw(&turtle, 103.8);
    ss->sg_prox = turtle.position;
    
    TURTLE_move(&turtle, 2.044);
    TURTLE_roll(&turtle, ss->chi3);
    TURTLE_yaw(&turtle, 103.8);
    ss->sg_dist = turtle.position;
    
    TURTLE_move(&turtle, 1.86);
    TURTLE_roll(&turtle, ss->chi4);
    TURTLE_yaw(&turtle, 112.8);
    ss->cb_dist = turtle.position;
    
    TURTLE_move(&turtle, 1.53);
    TURTLE_roll(&turtle, ss->chi5);
    TURTLE_pitch(&turtle, 180.0); 
    schain_to_bbone(&turtle);

    build_residue(&turtle, &n, &ca, &cb, &c);
    
    ss->n_dist = n;
    ss->ca_dist = ca;
    ss->c_dist = c;
    
/*   DISULFIDE_compute_torsional_energy(ss); */

    /* done */
    return(ss);
}


/*
  subroutine assumes turtle is in orientation #1 (at cA, headed toward
  Cb, with N on left), builds disulfide, and puts coordinates back into
  the disulfide structure passed. It also adds the distal protein backbone,
  and computes the disulfide conformational energy.
*/

void DISULFIDE_build(DISULFIDE *SS, TURTLE turtle)
{
  TURTLE tmp;
  VECTOR n, ca, cb, c;

  TURTLE_copy_coords(&tmp, &turtle);

  SS->ca_prox = tmp.position;

  /* switch to orientation #2 to build proximal backbone */
  schain_to_bbone(&tmp);
  build_residue(&tmp, &n, &ca, &cb, &c);

  SS->n_prox = n;
  SS->ca_prox = ca;
  SS->c_prox = c;

  bbone_to_schain(&tmp);
  TURTLE_move(&tmp, 1.53);
  TURTLE_roll(&tmp, SS->chi1);
  TURTLE_yaw(&tmp, 112.8);
  SS->cb_prox = tmp.position;

  TURTLE_move(&tmp, 1.86);
  TURTLE_roll(&tmp, SS->chi2);
  TURTLE_yaw(&tmp, 103.8);
  SS->sg_prox = tmp.position;

  TURTLE_move(&tmp, 2.044);
  TURTLE_roll(&tmp, SS->chi3);
  TURTLE_yaw(&tmp, 103.8);
  SS->sg_dist = tmp.position;

  TURTLE_move(&tmp, 1.86);
  TURTLE_roll(&tmp, SS->chi4);
  TURTLE_yaw(&tmp, 112.8);
  SS->cb_dist = tmp.position;

  TURTLE_move(&tmp, 1.53);
  TURTLE_roll(&tmp, SS->chi5);
  TURTLE_pitch(&tmp, 180.0);
  schain_to_bbone(&tmp);
  build_residue(&tmp, &n, &ca, &cb, &c);

  SS->n_dist = n;
  SS->ca_dist = ca;
  SS->c_dist = c;

  DISULFIDE_compute_torsional_energy(SS);
}



/* function computes and returns disulfide torsional energy for the input
   disulfide. It also sets the disulfide energy slot.
*/


double DISULFIDE_compute_torsional_energy(DISULFIDE *SS_input) 
{

  double x1, x2, x3, x4, x5;
  double energy = 0.0;

  x1 = SS_input->chi1;
  x2 = SS_input->chi2;
  x3 = SS_input->chi3;
  x4 = SS_input->chi4;
  x5 = SS_input->chi5;

  energy = 2.0 * (cos((double)DEGRAD(3.0 * x1)) + 
                  cos((double)DEGRAD(3.0 * x5)));

  energy += cos((double)DEGRAD(3.0 * x2)) + 
    cos((double)DEGRAD(3.0 * x4));

  energy += 3.5 * cos((double)DEGRAD(2.0 * x3)) + 
    0.6 * cos((double)DEGRAD(3.0 * x3)) + 10.1;

  SS_input->energy = energy;
  return(energy);
}

/* function computes disulfide dihedral angles from atomic positions */

void DISULFIDE_compute_dihedrals(DISULFIDE *SS)
{

  VECTOR n, ca, cb, sg, sg2, cb2, ca2, n2;
  double ang;

  n = SS->n_prox;
  ca = SS->ca_prox;
  cb = SS->cb_prox;
  sg = SS->sg_prox;
  sg2 = SS->sg_dist;
  cb2 = SS->cb_dist;
  ca2 = SS->ca_dist;
  n2 = SS->n_dist;

  /* chi1 */
  ang = find_dihedral(&n, &ca, &cb, &sg);
  SS->chi1 = ang;

  /* chi2 */
  ang = find_dihedral(&ca, &cb, &sg, &sg2);
  SS->chi2 = ang;

  /* chi3 */
  ang = find_dihedral(&cb, &sg, &sg2, &cb2);
  SS->chi3 = ang;

  /* chi4 */
  ang = find_dihedral(&sg, &sg2, &cb2, &ca2);
  SS->chi4 = ang;

  /* chi5 */
  ang = find_dihedral(&sg2, &cb2, &ca2, &n2);
  SS->chi5 = ang;



}
/* sorting the population by energy */

void DISULFIDE_sort_array(DISULFIDE *disulfide_array, int len, int updown)
{
  int cmp_SSa();
  int cmp_SSd();

  if (updown > 0)
    qsort(&disulfide_array[0], len, sizeof(DISULFIDE), cmp_SSa);
  else
    qsort(&disulfide_array[0], len, sizeof(DISULFIDE), cmp_SSd);

}

/* sort the keys structure into descending order */

int cmp_SSd(key1, key2)
DISULFIDE *key1, *key2;
{
  return (key1->energy < key2->energy);
}

/* sort the keys structure into ascending order */

int cmp_SSa(key1, key2)
DISULFIDE *key1, *key2;
{
  return (key1->energy > key2->energy);
}

/* print a disulfide_array */

void DISULFIDE_print_dihedral_array(DISULFIDE *disulfide_array, int len)
{
  register int i,j;
  int index;
  double energy;

  printf("\n ******************************* Disulfide Array ***************************\n\n");
  printf("Name                        Chi1     Chi2     Chi3     Chi4     Chi5    Energy\n");
  printf("------------------------------------------------------------------------------\n");

  for (i = 0; i < len; i++) {
    energy = disulfide_array[i].energy;
    printf("%-24s ", disulfide_array[i].name);
    printf("%8.2lf ", disulfide_array[i].chi1);
    printf("%8.2lf ", disulfide_array[i].chi2);
    printf("%8.2lf ", disulfide_array[i].chi3);
    printf("%8.2lf ", disulfide_array[i].chi4);
    printf("%8.2lf ", disulfide_array[i].chi5);
    printf("%8.3lf\n", energy);
  }
}

/* ----------- Choose random dihedral between MINVAL and MAXVAL ----------*/
/* -----------------------------------------------------------------------*/

double DISULFIDE_choose_random_dihedral(void)
{
  return ((double)(rand() % range) + MINVAL);
}

void DISULFIDE_random_build(DISULFIDE *SS, TURTLE turtle)
{
  double chi1, chi2, chi3, chi4, chi5;

/*  srand(); */
  chi1 = DISULFIDE_choose_random_dihedral();
  chi2 = DISULFIDE_choose_random_dihedral();
  chi3 = DISULFIDE_choose_random_dihedral();
  chi4 = DISULFIDE_choose_random_dihedral();
  chi5 = DISULFIDE_choose_random_dihedral();

  DISULFIDE_set_conformation(SS, chi1, chi2, chi3, chi4, chi5);
  DISULFIDE_build(SS, turtle);
}

/* 
   subroutine scans disulfide database array passed and computes minimum and maximum
   c-alpha c-alpha distances 
*/

void DISULFIDE_scan_database(DISULFIDE SS_array[], int max_entries, double *min, double *max)
{
    double min_dist = 999.9;
    double max_dist = -999.9;
    double distance = 0.0;

    VECTOR *prox, *distal;

    int i = 0;

    for (i = 0; i < max_entries; i++)
	{
	    prox = &(SS_array[i].ca_prox);
	    distal = &(SS_array[i].ca_dist);
	    distance = dist(prox, distal);

	    if (distance > max_dist)
		max_dist = distance;
	    else
		if (distance < min_dist)
		    min_dist = distance;
	}

    *min = min_dist;
    *max = max_dist;

}




int DISULFIDE_read_disulfide_database(char *fname,
				      DISULFIDE SS_array[],
				      int max_entries)
{

  FILE *f;
  int nentries;
  f = fopen(fname, "r");
  if(f==NULL) {
    fprintf(stderr,"can't open file %s\n",fname);
    return -1;
  }

  nentries = fget_SS_pdb(f, SS_array, max_entries);
  if (nentries <=0)
    fprintf(stderr,"\nREAD_DISULFIDE_DATABASE: no entries in <%s> ?\n",
	    fname);
  fclose(f);
  return(nentries);
}

/*
 * Read a Brookhaven file putting the coords into apos and the char
 *  string names into atype. Return number of atoms.  M Pique
 * Stop reading on End-of-file, or 'END' line, or line with just a ".".
 * Here is some sample input: 
NOTE   PROTK-178-PROTK-249
ATOM      1  N   CYS A   2      -5.442  -3.041  -2.056   1.000   0.000
ATOM      2  CA  CYS A   2      -4.319  -3.426  -1.191   1.000   0.000
ATOM      3  C   CYS A   2      -4.779  -4.516  -0.205   1.000   0.000
ATOM      4  CB  CYS A   2      -3.724  -2.199  -0.478   1.000   0.000
ATOM      5  SG  CYS A   2      -2.203  -2.642   0.447   1.000   0.000
ATOM      6  SG  CYS A   1      -0.746  -2.496  -1.019   1.000   0.000
ATOM      7  CB  CYS A   1      -0.593  -0.710  -1.199   1.000   0.000
ATOM      8  CA  CYS A   1       0.000   0.000   0.000   1.000   0.000
ATOM      9  C   CYS A   1       1.527  -0.000  -0.000   1.000   0.000
ATOM     10  N   CYS A   1      -0.438   1.418  -0.000   1.000   0.000
TER       
END
                                                         These charge values
							 are non-standard
							 and are not genuine
							 PDB. They are optional
*/

int fget_SS_pdb(FILE *f, DISULFIDE *SS_array, int max_entries)
{
  char linebuf[200];
  char name[128];
  char hdr[5], atype[4],amino[4], chain[4];
  int type, atnum, chnum;
  double x, y, z, temp, occ;
  register int n; /* number of atoms */

  VECTOR tvec;
  int res;

  n=0; res = 0;

  while (!feof(f)) {
    while (fgets(linebuf, sizeof linebuf, f)!=NULL && n < max_entries &&
	   0!=strcmp(".", linebuf) &&
	   0!=strncmp("END", linebuf,3) &&
	   0!=strncmp("end", linebuf,3) &&
	   0!=strncmp("TER", linebuf,3)) {

      if (( 0 == strncmp("NOTE", linebuf,4))) {
	sscanf(&linebuf[5], "%s", name);
	DISULFIDE_init(&(SS_array[n]), &name[0]); 
	DISULFIDE_set_resnum(&(SS_array[n]), 1, 2);

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].n_dist = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].ca_dist = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].c_dist = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].cb_dist = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].sg_dist = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].sg_prox = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].cb_prox = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].ca_prox = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].c_prox = tvec;

	fscanf(f, "%s %d %s %s %s %d %lf %lf %lf %lf %lf %lf \n",
	       hdr, &atnum, atype, amino, chain, &chnum, &x, &y, &z, &temp,
	       &occ);
	VECTOR_init(&tvec, x, y, z);
	SS_array[n].n_prox = tvec;
      }
      
      /* compute dihedrals and conformational energy for this SS */

      DISULFIDE_compute_dihedrals(&SS_array[n]);
      DISULFIDE_compute_torsional_energy(&SS_array[n]);

      n++;
    }
  }
  return n;
}

/*
  function compares the dihedrals of the two disulfides and returns the RMS
  difference across the 5 angles.
*/

double DISULFIDE_compare_dihedrals(DISULFIDE *SS1, DISULFIDE *SS2)
{

  double diff = 0.0;
  double chi1, chi2, chi3, chi4, chi5;
  double chi1a, chi2a, chi3a, chi4a, chi5a;

  chi1 = SS1->chi1;
  chi2 = SS1->chi2;
  chi3 = SS1->chi3;
  chi4 = SS1->chi4;
  chi5 = SS1->chi5;
  
  chi1 = chi1 < 0.0 ? 360.0 + chi1 : chi1;
  chi2 = chi2 < 0.0 ? 360.0 + chi2 : chi2;
  chi3 = chi3 < 0.0 ? 360.0 + chi3 : chi3;
  chi4 = chi4 < 0.0 ? 360.0 + chi4 : chi4;
  chi5 = chi5 < 0.0 ? 360.0 + chi5 : chi5;

  chi1a = SS2->chi1;
  chi2a = SS2->chi2;
  chi3a = SS2->chi3;
  chi4a = SS2->chi4;
  chi5a = SS2->chi5;
  
  chi1a = chi1a < 0.0 ? 360.0 + chi1a : chi1a;
  chi2a = chi2a < 0.0 ? 360.0 + chi2a : chi2a;
  chi3a = chi3a < 0.0 ? 360.0 + chi3a : chi3a;
  chi4a = chi4a < 0.0 ? 360.0 + chi4a : chi4a;
  chi5a = chi5a < 0.0 ? 360.0 + chi5a : chi5a;

  diff = (chi1 - chi1a) * (chi1 - chi1a);
  diff += (chi2 -chi2a) * (chi2 - chi2a);
  diff += (chi3 - chi3a) * (chi3 - chi3a);
  diff += (chi4 - chi4a) * (chi4 - chi4a);
  diff += (chi5 - chi5a) * (chi5 - chi5a);

/*
  diff = (SS1->chi1 - SS2->chi1) * (SS1->chi1 - SS2->chi1);
  diff += (SS1->chi2 - SS2->chi2) * (SS1->chi2 - SS2->chi2);
  diff += (SS1->chi3 - SS2->chi3) * (SS1->chi3 - SS2->chi3);
  diff += (SS1->chi4 - SS2->chi4) * (SS1->chi4 - SS2->chi4);
  diff += (SS1->chi5 - SS2->chi5) * (SS1->chi5 - SS2->chi5);
*/

  if (diff != 0.0)
    diff = sqrt((diff / 5.0));

  return diff;

}

/* 
  function returns the RMS deviation in the backbone and sidechain atomic
  positions for the two disulfides input.
*/

double DISULFIDE_compare_positions(DISULFIDE *SS1, DISULFIDE *SS2)
{

  double d = 0.0;
  
  d = dist(&SS1->n_prox, &SS2->n_prox);
  d += dist(&SS1->ca_prox, &SS2->ca_prox);
  d += dist(&SS1->cb_prox, &SS2->cb_prox);
  d += dist(&SS1->sg_prox, &SS2->sg_prox);
  d += dist(&SS1->sg_dist, &SS2->sg_dist);
  d += dist(&SS1->cb_dist, &SS2->cb_dist);
  d += dist(&SS1->ca_dist, &SS2->ca_dist);
  d += dist(&SS1->n_dist, &SS2->n_dist);

  if (d != 0.0)
    d = sqrt((d / 8.0));

  return(d);

}

/* 
  function returns the RMS deviation between the passed n, ca, c, o,
  and cb (which might be null) and the given disulfide
*/

double DISULFIDE_compare_to_backbone_atoms(DISULFIDE *ss, ATOM *n,
					   ATOM *ca, ATOM *c,
					   ATOM *cb)
{
    double d = 0.0;
    int numb = 0;
    VECTOR pos;

    numb = (cb == (ATOM *)NULL) ? 3 : 4;  /* for the RMS calculation */


/*    pos = n->position;
      VECTOR_print((&pos));
*/

    d += dist(&ss->n_dist, &n->position);
    d += dist(&ss->ca_dist, &ca->position);
    d += dist(&ss->c_dist, &c->position);

    if (numb == 4)
	{
	    d += dist(&ss->cb_dist, &cb->position);
	}


    if (d != 0.0)
	d = sqrt((d / numb));

    return(d);
}


/* 
  function returns the RMS deviation in the backbone atomic
  positions for the two disulfides input.
*/

double DISULFIDE_compare_backbones(DISULFIDE *SS1, DISULFIDE *SS2)
{

  double d = 0.0;
  
/*
  d += dist(&SS1->cb_prox, &SS2->cb_prox);
  d += dist(&SS1->cb_dist, &SS2->cb_dist);
*/
  d += dist(&SS1->ca_dist, &SS2->ca_dist);
  d += dist(&SS1->n_dist, &SS2->n_dist);
  d += dist(&SS1->c_dist, &SS2->c_dist);

  if (d != 0.0)
    d = sqrt((d / 3.0));

  return(d);

}
  
