/* $Id: distmat.h,v 1.3 1995/06/23 00:10:55 suchanek Exp $ */
/* $Log: distmat.h,v $
 * Revision 1.3  1995/06/23  00:10:55  suchanek
 * added prototypes
 *
 * Revision 1.2  1995/06/20  01:13:00  suchanek
 * added prototypes
 *
 * Revision 1.1  1995/06/07  14:44:04  suchanek
 * Initial revision
 * */

#ifndef DISTMAT_H
#include "atom.h"
#include "geometry.h"
#include "residue.h"
#include "protutil.h"

#define TRUE 1
#define FALSE 0

typedef struct _tagDISTMAT {
  int lores;
  int hires;
  double cutoff;
  double **matrix;
  char *name;
} DISTMAT, *pDISTMAT;


DISTMAT 
   *new_DISTMAT(),
   *DISTMAT_build_from_atom_list(),
   *DISTMAT_build_from_phipsi_list(),
   *DISTMAT_difference();

DISTMAT *DISTMAT_build_from_backbone_list(BACKBONE *bb,
					  char *name,
					  double cutoff,
					  int lores,
					  int hires);

void 
  DISTMAT_init(),
  DISTMAT_print(),
  DISTMAT_tab_print(),
  DISTMAT_range_print(),
  free_DISTMAT();

double DISTMAT_rms();

double DISTMAT_get(DISTMAT *v, int row, int column);

int
  DISTMAT_cutoff();

#define MAX_CUTOFF ((double) 99.9)
#define DISTMAT_H 1
#endif
