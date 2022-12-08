/* $Id: phipsi.h,v 1.1 1995/06/07 14:44:29 suchanek Exp $ */
/* $Log: phipsi.h,v $
 * Revision 1.1  1995/06/07  14:44:29  suchanek
 * Initial revision
 * */

#ifndef PHIPSI_H

#include "geometry.h"
#include "protutil.h"

#define TRUE 1
#define FALSE 0
#define PHIPSI_UNDEFINED ((double) -999.9)

#define MAXVAL 179
#define MINVAL -180

#define MAXOMEGA 200
#define MINOMEGA 160

#define _ps_range ((int)(MAXVAL - MINVAL + 1))
#define _omega_range ((int)(MAXOMEGA - MINOMEGA + 1))

typedef struct _tagPhiPsi {
  int resnumb;
  int numelem;
  double phi;
  double psi;
  double omega;
} PHIPSI, *pPHIPSI;


PHIPSI 
   *new_PHIPSI_list(),
   *PHIPSI_list_scan();

void 
  free_PHIPSI_list(),
  PHIPSI_list_init(),
  PHIPSI_list_print(),
  PHIPSI_range_print(),
  PHIPSI_init(),
  PHIPSI_pprint(),
  PHIPSI_print(),
  PHIPSI_choose_random_dihedrals(),
  PHIPSI_choose_random_allowed_dihedrals();


int 
  PHIPSI_list_length(),
  PHIPSI_random_init();


#define PHIPSI_H 1
#endif
