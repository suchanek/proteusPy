/* $Id: disulfide.h,v 1.5 1995/06/25 00:28:44 suchanek Exp $ */
/* $Log: disulfide.h,v $
 * Revision 1.5  1995/06/25  00:28:44  suchanek
 * added DISULFIDE_build_at_residue_backwards function
 *
 * Revision 1.4  1995/06/22  20:37:05  suchanek
 * additional logic for using the backbone arrays
 *
 * Revision 1.3  1995/06/20  01:13:09  suchanek
 * added prototypes
 *
 * Revision 1.2  1995/06/11  05:14:05  suchanek
 * added more functions, prototypes
 *
 * Revision 1.1  1995/06/07  14:44:08  suchanek
 * Initial revision
 * */


#ifndef DISULFIDE_H

#include "geometry.h"
#include "turtle.h"
#include "residue.h"

typedef struct {
  VECTOR n_prox;                /* coordinates for proximal N */
  VECTOR ca_prox;               /* coordinates for proximal Ca */
  VECTOR c_prox;                /* coordinates for proximal C */
  VECTOR o_prox;                /* coordinates for proximal O */
  VECTOR cb_prox;               /* coordinates for proximal Cb */
  VECTOR sg_prox;               /* coordinates for proximal Sg */
  VECTOR sg_dist;               /* coordinates for distal Sg */
  VECTOR cb_dist;               /* coordinates for distal Cb */
  VECTOR ca_dist;               /* coordinates for distal Ca */
  VECTOR n_dist;                /* coordinates for distal N */
  VECTOR c_dist;                /* coordinates for distal C */
  VECTOR o_dist;                /* coordinates for distal O */
  double chi1;                  /* Chi1 dihedral angle */
  double chi2;                  /* Chi2 dihedral angle */
  double chi3;                  /* Chi3 dihedral angle */
  double chi4;                  /* Chi2' dihedral angle */
  double chi5;                  /* Chi1' dihedral angle */
  double energy;                /* Approximate torsional energy */
  int prox_resnum;              /* proximal residue number */
  int dist_resnum;              /* distal residue number */
  char name[128];               /* a string identifier */
} DISULFIDE;

/* a structure for potential disulfide hits */

typedef struct tag_ss_hit {
    int backwards;
    int index;
    int proximal;
    int distal;
    int bumps;          /* number of bumps for this model */
    double rms_error;   /* error in building this */
    DISULFIDE model_ss; /* current modelled disulfide at this position */
} SS_HIT;


double DISULFIDE_compute_torsional_energy(DISULFIDE *ss);
double DISULFIDE_choose_random_dihedral(void);
double DISULFIDE_compare_dihedrals(DISULFIDE *SS1, DISULFIDE *SS2);
double DISULFIDE_compare_positions(DISULFIDE *SS1, DISULFIDE *SS2);
double DISULFIDE_compare_backbones(DISULFIDE *SS1, DISULFIDE *SS2);
double DISULFIDE_compare_to_backbone_atoms(DISULFIDE *ss, ATOM *n,
					   ATOM *ca, ATOM *c,
					   ATOM *cb);

void DISULFIDE_print(DISULFIDE *ss);
void DISULFIDE_init(DISULFIDE *ss, char *name);
void DISULFIDE_set_conformation(DISULFIDE *SS, double chi1,
				double chi2, double chi3, double chi4,
				double chi5);
void DISULFIDE_set_resnum(DISULFIDE *SS, int prox, int distal);
void DISULFIDE_build(DISULFIDE *SS, TURTLE turtle);
void DISULFIDE_random_build(DISULFIDE *SS, TURTLE turtle);
void DISULFIDE_sort_array(DISULFIDE *disulfide_array, int len, int updown);
void DISULFIDE_print_dihedral_array(DISULFIDE *disulfide_array, int len);
void DISULFIDE_compute_dihedrals(DISULFIDE *SS);

void DISULFIDE_set_positions(DISULFIDE *SS, VECTOR n1, VECTOR ca1, 
			     VECTOR c1, VECTOR o1, VECTOR cb1, 
			     VECTOR sg1, VECTOR n2, VECTOR ca2,
			     VECTOR c2, VECTOR o2,VECTOR cb2, VECTOR sg2);

void free_DISULFIDE_list(DISULFIDE *ss);
void free_DISULFIDE(DISULFIDE *ss);

void DISULFIDE_scan_database(DISULFIDE SS_array[], int max_entries, double *min, double *max);

int DISULFIDE_read_disulfide_database(char *fname,
				      DISULFIDE SS_array[],
				      int max_entries);

DISULFIDE *new_DISULFIDE_list(int nh);
DISULFIDE *new_DISULFIDE(char *name);
DISULFIDE *DISULFIDE_build_at_residue(ATOM *atoms, 
				      BACKBONE *bb,
				      int resnumb, 
				      int lowres,
				      DISULFIDE *model_ss);

DISULFIDE *DISULFIDE_build_at_residue_backwards(ATOM *atoms, 
				      BACKBONE *bb,
				      int resnumb, 
				      int lowres,
				      DISULFIDE *model_ss);

#define DISULFIDE_DB "/home/suchanek/ss/ss_patterns.dat"
#define MAX_SS_NUMBER 100

#define DISULFIDE_H 1
#endif
