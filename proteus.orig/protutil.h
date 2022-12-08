/* $Id: protutil.h,v 1.3 1995/06/23 14:28:33 suchanek Exp $ */
/* $Log: protutil.h,v $
 * Revision 1.3  1995/06/23  14:28:33  suchanek
 * added prototypes, NUMELEM macro
 *
 * Revision 1.2  1995/06/22  20:37:18  suchanek
 * removed the exec/types.h include
 *
 * Revision 1.1  1995/06/07  14:44:56  suchanek
 * Initial revision
 * */



#ifndef PROTEUS_H

#ifndef AMIGA
/*#include <malloc.h>*/
#include <strings.h>
#else
/*#include <exec/types.h> */
#include <memory.h>
#include <string.h>
#include <stdlib.h>
#endif

#include <stdio.h>
#include <math.h>

void Proteus_warning(char *error_text);
void Proteus_error(char *error_text);

#define NUMELEM(a) (sizeof(a)/sizeof(a[0]))

#define PROTEUS_H 1
#endif
