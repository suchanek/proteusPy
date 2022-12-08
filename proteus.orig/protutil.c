/*  $Id: protutil.c,v 1.3 1995/06/23 14:28:44 suchanek Exp $  */
/*  $Log: protutil.c,v $
 * Revision 1.3  1995/06/23  14:28:44  suchanek
 * added prototypes
 *
 * Revision 1.2  1993/12/10  09:23:04  egs
 * *** empty log message ***
 *
 * Revision 1.1  93/12/10  09:22:58  egs
 * Initial revision
 *  */

static char rcsid[] = "$Id: protutil.c,v 1.3 1995/06/23 14:28:44 suchanek Exp $";



#include <stdio.h>
#include "protutil.h"


void Proteus_error(char *error_text)
{
	void exit();

	fprintf(stderr,"PROTEUS run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

#ifdef AMIGA
void Proteus_warning(char *error_text)
{
  extern LONG PostMsg(); /*in project.c */

  PostMsg("PROTEUS run-time error...\n%s\n",error_text);

}

#else

void Proteus_warning(char *error_text)
{
  fprintf(stderr,"PROTEUS run-time warning...\n");
  fprintf(stderr,"%s\n",error_text);
}

#endif
