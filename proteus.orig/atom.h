/* $Id: atom.h,v 1.3 1995/06/22 20:37:01 suchanek Exp $ */
/* $Log: atom.h,v $
 * Revision 1.3  1995/06/22  20:37:01  suchanek
 * *** empty log message ***
 *
 * Revision 1.2  1995/06/07  20:19:01  suchanek
 * added a proto
 *
 * Revision 1.1  1995/06/07  14:43:46  suchanek
 * Initial revision
 * */

#ifndef ATOM_H
#include "stdlib.h"
#include "geometry.h"
#include "protutil.h"


#define H_TYPE 1
#define N_TYPE 2
#define C_TYPE 3
#define O_TYPE 4
#define S_TYPE 5
#define P_TYPE 6
#define NO_TYPE 7
#define CA_TYPE 8
#define FE_TYPE 9

#define H_COLOR 1
#define N_COLOR 2
#define C_COLOR 3
#define S_COLOR 4
#define P_COLOR 4
#define O_COLOR 5
#define OTHER_COLOR 5

/* these are CPK radii
#define C_RAD		2.0
#define N_RAD		1.8
#define O_RAD		2.2
#define H_RAD		1.0
#define S_RAD		1.9
#define P_RAD		2.1
#define FACTOR		1.0
#define MAX_RAD		2.2
*/

/* 
  these are rough radii based on empirical lower limits for nonbonded
  contacts. Principles of Protein Structure, p. 33 (Bondi)
*/

#define C_RAD		1.7
#define N_RAD		1.6
#define O_RAD		1.5
#define H_RAD		1.0
#define S_RAD		1.8
#define P_RAD		1.7   /* a guess */
#define FACTOR		1.0
#define MAX_RAD		2.0

#define VDW_BUMP 1.0  /* atom radius scaling factor for collisions */

#define MAX_LINE 512
#define TRUE 1
#define FALSE 0

typedef struct _tagAtom {
  VECTOR position;         /* 3D coordinates of this atom */
  int atnumb;              /* Atom number                 */
  char atnam[5];           /* String name of this atom    */
  int atype;               /* Integer representation of atom type */
  char resnam[4];          /* amino acid type */
  int resnumb;             /* residue number              */
  int collisions;          /* Number of collisions for atom */
  int color;               /* atom color                  */
  double radius;           /* rough atomic radius         */
  double charge;           /* approximate atomic charge   */
  int natoms;              /* number of atoms in the list */
  int excluded;            /* 1 if excluded, 0 otherwise  */
} ATOM, *pATOM;


ATOM 
   *new_ATOM(void),
   *new_ATOM_list(),
   *ATOM_sphere(),
   *ATOM_list_scan();

void 
  ATOM_list_init(),
  ATOM_list_print(),
  ATOM_list_pprint(),
  ATOM_range_print(),
  ATOM_init(),
  ATOM_print(),
  ATOM_pprint(),
  ATOM_make_atom_types(),
  ATOM_make_radii(),
  ATOM_make_charges(),
  ATOM_make_colors(),
  ATOM_find_collisions(),
  ATOM_write_pdb(),
  free_ATOM_list(),
  free_ATOM_sphere(),
  ATOM_collisions_reset(),
  ATOM_include_all(),
  ATOM_exclude_all();

int 
   ATOM_list_length(),
   ATOM_pair_bumpsP(),
   ATOM_read_pdb(),
   ATOM_collisions(),
   ATOM_exclude_backbone(),
   ATOM_exclude_sidechain(),
   ATOM_include_backbone(),
   ATOM_include_sidechain();

double
   ATOM_dist();

#define ATOM_H 1
#endif
