/* $Id: turtle.h,v 1.3 1995/06/25 02:44:55 suchanek Exp $ */
/* $Log: turtle.h,v $
 * Revision 1.3  1995/06/25  02:44:55  suchanek
 * added prototypes.
 *
 * Revision 1.2  1995/06/20  03:00:55  suchanek
 * function renaming
 *
 * Revision 1.1  1995/06/07  14:44:49  suchanek
 * Initial revision
 * */

#ifndef TURTLE_H
#include "geometry.h"

#define UP 0
#define DOWN 1

typedef struct {
	VECTOR position;
	VECTOR heading;
	VECTOR left;
	VECTOR up;
	char name[128];
	int pen;
	} TURTLE;



void 
  TURTLE_copy_coords(),
  TURTLE_init(),
  TURTLE_reset_coords (),
  TURTLE_print(),
  TURTLE_move(),
  TURTLE_roll(),
  TURTLE_yaw(),
  TURTLE_pitch(),
  TURTLE_turn(),
  TURTLE_orient(),
  TURTLE_pen_up(),
  TURTLE_pen_down();

	

VECTOR 
  TURTLE_to_local(),
  TURTLE_to_global();

int get_sincos_index(double deg);


double my_qsin(double deg);
double my_qcos(double deg);


#define TURTLE_H 1
#endif
