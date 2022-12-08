/* $Id: protein.h,v 1.2 1995/06/07 15:17:02 suchanek Exp $ */
/* $Log: protein.h,v $
 * Revision 1.2  1995/06/07  15:17:02  suchanek
 * added prototypes and some cleanup
 *
 * Revision 1.1  1995/06/07  14:44:37  suchanek
 * Initial revision
 * */


#ifndef PROTEIN_H
#include "geometry.h"
#include "atom.h"
#include "phipsi.h"
#include "residue.h"
#include "distmat.h"

#define TRUE 1
#define FALSE 0

typedef struct _tagPROTEIN {
  char name[80];
  int nh;             /* used for lists of proteins */
  int residues;
  double molwt;
  double charge;
  double pH;
  int low_residue;
  int high_residue;
  int selected_low;
  int selected_high;
  ATOM *atom_list;
  PHIPSI *phipsi_list;
  DISTMAT *distmat;
} PROTEIN, *pPROTEIN;


PROTEIN   *new_PROTEIN(char *name, int numb);
PROTEIN   *new_PROTEIN_list(int nh);
PROTEIN   *PROTEIN_read_pdb(char *filename, char *prot_name);
PROTEIN   *PROTEIN_scan(PROTEIN *protein_list, char *name);
PROTEIN   *PROTEIN_list_scan(PROTEIN *protein_list, char *name);

void  PROTEIN_init(PROTEIN *protein, char *name, int numb, 
		  int low_res, int hi_res);
void  free_PROTEIN(PROTEIN *protein);
void  PROTEIN_print(PROTEIN *protein);
void  PROTEIN_compute_props();
void  PROTEIN_list_init(PROTEIN *protein, int nh);
void  free_PROTEIN_list(PROTEIN *protein);
void  PROTEIN_list_print(PROTEIN *protein);
void PROTEIN_range_print(PROTEIN *v, int nl, int nh);



int  PROTEIN_load_pdb();
int PROTEIN_list_length(PROTEIN *proteins);


#define PROTEIN_H 1
#endif
