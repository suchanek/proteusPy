/*  $Id: protein.c,v 1.2 1995/06/07 15:16:51 suchanek Exp $  */
/*  $Log: protein.c,v $
 * Revision 1.2  1995/06/07  15:16:51  suchanek
 * added prototypes and some cleanup
 *
 * Revision 1.1  1993/12/10  09:23:30  egs
 * Initial revision
 * */

static char rcsid[] = "$Id: protein.c,v 1.2 1995/06/07 15:16:51 suchanek Exp $";

#include "protutil.h"
#include "protein.h"


#undef DBG

/* ---------------  -------------- */

/* --------------- Memory Allocation and Initialization -------------- */

PROTEIN *new_PROTEIN(char *name, int numb)
{
	PROTEIN *v;

	v=(PROTEIN *)malloc((unsigned)sizeof(PROTEIN));
	if (!v) Proteus_error("allocation failure in new_PROTEIN()");
	PROTEIN_init(v,name,numb,1,2);
	return v;
}


/*
 * Function allocates space for an protein list of NH elements.
 * It also calls PROTEIN_list_init to fill in default information
 * in the structure
 */

PROTEIN *new_PROTEIN_list(int nh)
{
	PROTEIN *v;
	int nl = 0;
	char name[80];

	v=(PROTEIN *)malloc((unsigned) (nh-nl+1)*sizeof(PROTEIN));
	if (!v) Proteus_error("allocation failure in PROTEIN_list()");
	for (nl = 0; nl < nh; nl++)
	  {
	    sprintf(name,"empty_%d", nl);
	    PROTEIN_init(v, name, 2, 1, 2);
	  }
	return v;
}

/*
 * Function frees space allocated to an PROTEIN list.
 *
 */

void free_PROTEIN_list(PROTEIN *protein)
{
  int numb = 0;
  int i = 0;

  if (protein != (PROTEIN *)NULL)
    {
      numb = PROTEIN_list_length(protein);
      for (i = 0; i < numb; i++)
	free_PROTEIN((PROTEIN *)&protein[i]);
    }
}

void free_PROTEIN(PROTEIN *v)
{
  int numb = 0;
  if (v != (PROTEIN *)NULL)
    {
      free_ATOM_list(v->atom_list);
      free_PHIPSI_list(v->phipsi_list);
      free_DISTMAT(v->distmat);
      free((char *)v);
    }
}

/*
 * Function initializes a specific PROTEIN with its name, number, and
 * residue info.
 */

void PROTEIN_init(PROTEIN *protein, char *name, int numb, 
		  int low_res, int hi_res)
{
  strcpy(protein->name,name);
  protein->nh = numb;
  protein->low_residue = low_res;
  protein->high_residue = hi_res;
  protein->selected_low = low_res;
  protein->selected_high = hi_res;
  protein->residues = 0;
  protein->molwt = 0.0;
  protein->charge = 0.0;
  protein->pH = 7.0;
}

/*
 * Function initializes an entire list of atoms to default values.
 *
 */

void PROTEIN_list_init(PROTEIN *protein, int nh)
{
  int nl = 0;
  int i;

  char name[80];

  for(i = nl; i < nh; i++) {
    sprintf(name,"Empty_%d", i);
    PROTEIN_init((PROTEIN *)&protein[i], name, i, 1, 2);
  }
}

/*
 * Function returns the current protein list length. The list must have been
 * initialized first!
 */
#define NUMELEM(a) (sizeof(a)/sizeof(a[0]))

int PROTEIN_list_length(PROTEIN *proteins)
{
  int len = NUMELEM(proteins);
  return(len);
}


/* --------------- Writing the Structure Contents  -------------- */

/*
 * Function prints out the properties of a specific atom.
 *
 */

void PROTEIN_print(PROTEIN *v)
{

    printf("\tProtein: [%s]\n\tResidues: \t\t\t%4d\n\tMolecular Weight: \t%8.2lf\n\t\
Charge @ pH %4.1lf: \t%8.2lf\n\tInitial Residue: \t\t%4d\n\tFinal Residue: \t\t\t%4d\n\t\
Initial Selected Residue: \t%4d\n\tFinal Selected Residue: \t%4d\n",
	   v->name, v->residues, v->molwt, v->pH, v->charge, v->low_residue,
	   v->high_residue, v->selected_low, v->selected_high);

    printf("\n*****************************************************\n");
    printf("\tAtom List:\n");
    printf("*****************************************************\n");
    ATOM_list_print(v->atom_list);
    printf("\n*****************************************************\n");

    printf("\tPhi-Psi List:\n");
    printf("*****************************************************\n");
    PHIPSI_list_print(v->phipsi_list);

    printf("\n*****************************************************\n");
    printf("\tDistance Matrix:\n");
    printf("*****************************************************\n");
    DISTMAT_print(v->distmat);
    printf("*****************************************************\n");
}

/*
 * Function prints out the stats for an entire list of atoms.
 *
 */

void PROTEIN_list_print(PROTEIN *v)
{
  int i, j;
  int nl = 0;
  int nh = 0;

  nh = PROTEIN_list_length(v);

  for (i = 0; i < nh; i++) {
    PROTEIN_print(&v[i]);
  }

}

/*
 * Function prints out the stats for a specific range of PROTEINs in an
 * PROTEIN list.
 */

void PROTEIN_range_print(PROTEIN *v, int nl, int nh)
{
  int i, j;

  int atnum_max = PROTEIN_list_length(v);

  if (nl < 0 || nl >= nh) {
    fprintf(stderr,"PROTEIN_range_print - error, NL <%d> out of range...\n",nl);
    nl = 0;
  }

  if (nh > atnum_max) {
    fprintf(stderr,"PROTEIN_range_print - error, NH <%d> out of range...\n",nh);
    nh = atnum_max;
  }

  for (i = nl; i < nh; i++) {
    PROTEIN_print(&v[i]);
  }

}


/* 
 * Function returns a pointer to a protein within the input protein_list
 * specified by the protein name string.
 *
 */

PROTEIN *PROTEIN_list_scan(PROTEIN *protein_list, char *name)
{

  int len;
  int i;
  char *current;

  len = PROTEIN_list_length(protein_list);

  for (i = 0; i < len; i++ ) {
    current = protein_list[i].name;
    if (strcmp(name, current) == 0)
      return(&protein_list[i]);
  }
  return((PROTEIN *)NULL);

}
/*
 *
 * Function creates and initializes a protein structure object
 * by reading in a PDB format file. It fills in the atom-list slot
 * with the atoms read, and will eventually be the main entry point
 * for computing physical properties at read time. Right now it
 * basically computes the residue counts, low residue and high residue.
 * On a fast machine, it could file in the distance matrix and phi-psi
 * list at this point as well.
 *
 *
*/

PROTEIN *PROTEIN_read_pdb(char *filename, char *prot_name)
{
  PROTEIN *res = (PROTEIN *)NULL;
  ATOM *atomlist = NULL;
  DISTMAT *dm = NULL;
  PHIPSI *phipsi = NULL;

  int count= 0;
  int lores, hires;
  int tot_residues;

  if ((res = new_PROTEIN(prot_name,0)) == (PROTEIN *)NULL)
    return(res);

  count = ATOM_read_pdb(&atomlist, filename);

  if (count <= 0) {
    Proteus_warning("Can't read protein file?\n");
    free_PROTEIN(res);
    return((PROTEIN *)NULL);
  }

  tot_residues = RESIDUE_range(atomlist, &lores, &hires);

  res->low_residue = lores;
  res->high_residue = hires;
  res->selected_low = lores;
  res->selected_high = hires;
  res->residues = tot_residues;
  res->atom_list = atomlist;

  count = RESIDUE_phi_psi_list_build(atomlist, &phipsi);
  if (count <=0)
    Proteus_warning("Can't build PHIPSI list?\n");
  else
    res->phipsi_list = phipsi;

  dm = DISTMAT_build_from_atom_list(atomlist, prot_name, MAX_CUTOFF, 0);
  if (dm == (DISTMAT *)NULL)
    Proteus_warning("Can't build DISTMAT list?\n");
  else
    res->distmat = dm;

  return(res);
}

