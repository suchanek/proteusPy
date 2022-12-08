/* $Id: residue.h,v 1.4 1995/06/23 14:17:17 suchanek Exp $ */
/* $Log: residue.h,v $
 * Revision 1.4  1995/06/23  14:17:17  suchanek
 * added a free_cb flag to the BACKBONE structure
 *
 * Revision 1.3  1995/06/22  20:37:29  suchanek
 * code for the residue functions
 *
 * Revision 1.2  1995/06/11  05:13:56  suchanek
 * added prototypes
 *
 * Revision 1.1  1995/06/07  14:44:43  suchanek
 * Initial revision
 * */


#ifndef RESIDUE_H

#include "geometry.h"
#include "turtle.h"
#include "atom.h"
#include "phipsi.h"
#include "protutil.h"

/* orientation symbols for orient_at_residue */

#define ORIENT_BACKBONE  2
#define ORIENT_SIDECHAIN 1

typedef struct {
    int resnumb;
    int nres;
    int lowres;
    int hires;
    int free_cb;
    ATOM *n_atom;
    ATOM *ca_atom;
    ATOM *c_atom;
    ATOM *o_atom;
    ATOM *cb_atom;
} BACKBONE;

void bbone_to_schain(TURTLE *turtle);
void schain_to_bbone(TURTLE *turtle);
void build_residue(TURTLE *turtle, VECTOR *n, VECTOR *ca, 
		   VECTOR *cb, 
		   VECTOR *c);
void add_oxygen(TURTLE *turtle, ATOM *O);

void RESIDUE_get_backbone(ATOM *atoms, int resnum, 
			  ATOM **n,
			  ATOM **ca,
			  ATOM **c,
			  ATOM **o,
			  ATOM **cb,
			  int build_cb,
			  int *do_cleanup);

VECTOR to_alpha(TURTLE *turtle, double phi);
VECTOR to_carbonyl(TURTLE *turtle, double psi);
VECTOR to_nitrogen(TURTLE *turtle, double omega);
VECTOR to_oxygen(TURTLE *turtle);


int orient_at_residue(TURTLE *turt, ATOM *atoms, int resnum, int orientation);
int RESIDUE_phi_psi_omega(ATOM *atoms, int resnum, 
			  double *phi, double *psi, double *omega);
int RESIDUE_range(ATOM *atoms, int *lores, int *hires);
int RESIDUE_phi_psi_list_build(ATOM *atoms, PHIPSI **phipsi_out);

ATOM *RESIDUE_build_backbone_from_phipsi(PHIPSI *ps_list, TURTLE *turtle);

BACKBONE *new_BACKBONE_list(int nl, int nh);
BACKBONE *BACKBONE_create(ATOM *atoms);

void free_BACKBONE_list(BACKBONE *bb);
void BACKBONE_init(BACKBONE *backbone, int resnumb, ATOM *n, 
		   ATOM *ca, ATOM *c, ATOM *o, ATOM *cb, int free_cb);

void BACKBONE_list_init(BACKBONE *backbone, int nl, int nh);
int BACKBONE_list_length(BACKBONE *backbone);
void BACKBONE_print(BACKBONE *bb, int lowres, int hires);






#define RESIDUE_H 1

#endif
