/* $Id: keypop.h,v 1.1 1995/06/07 14:44:24 suchanek Exp $ */
/* $Log: keypop.h,v $
 * Revision 1.1  1995/06/07  14:44:24  suchanek
 * Initial revision
 * */

#include <stdio.h>

#define     CORRECT          ((double) 1.0)
#define     INCORRECT        ((double) -1.0)
#define     MISSING          ((double) -999.9)
#define     WRONG            ((double) -666.6)

typedef struct {
  int index;
  int nelem;
  double energy;
} KEYPOP;


KEYPOP 
  *new_KEYPOP(),
  *merge_keys(),
  *KEYPOP_find_key();

void 
  free_KEYPOP(),
  KEYPOP_init(),
  KEYPOP_sort(),
  KEYPOP_print(),
  KEYPOP_print_keys(),
  KEYPOP_print_subgroup();

