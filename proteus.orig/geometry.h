/* $Id: geometry.h,v 1.3 1995/06/23 00:11:02 suchanek Exp $ */
/* $Log: geometry.h,v $
 * Revision 1.3  1995/06/23  00:11:02  suchanek
 * *** empty log message ***
 *
 * Revision 1.2  1995/06/20  03:01:00  suchanek
 * function renaming
 *
 * Revision 1.1  1995/06/07  14:44:16  suchanek
 * Initial revision
 * */


/* Header file to be included in all sections of PROTEUS requiring the */
/* Geometry primitives       */
 
/* The functions returning VECTORS are:
*
* VECTOR cross_prod(),VECTOR_Difference(),scalar_times_vector();
* VECTOR VECTOR_Sum(),unit(),apply_rotmat(),matrix_times_vector();
* VECTOR average_vectors();
*
* Functions returning DOUBLE are:
* DOUBLE dot_prod(),VECTOR_Length(),dist(),dist_to_line(),find_angle();
* DOUBLE find_bond_angle(),find_dihedral(), determinant();
*
* Functions of type VOID are:
* print_rotmat(), make_rotmat(),matrix_times_matrix();
*
*/ 
#ifndef GEOMETRY_H

#include "protutil.h"

#ifndef PI
#define PI 3.14159267
#endif

#define DEGRAD(degrees) ((degrees) * (0.017453293))
#define RADDEG(rad) ((rad) * (57.29578))


/* let's define a new type for Vector Geometry operators */

typedef struct {  
double x;
double y;
double z;
        } VECTOR;

typedef struct {
double r;
double theta;
double z;
       } CYLINDRICAL_COORDS;

typedef struct {
double rho;
double theta;
double phi;
       } SPHERICAL_COORDS;



/*            Here are the geometry function primitives       */

void 
  VECTOR_init(),
  VECTOR_print(),
  print_rotmat(),
  make_rotmat(),
  matrix_times_matrix();

VECTOR 
  cross_prod(), 
  VECTOR_difference(), 
  scalar_times_vector(), 
  VECTOR_sum(),
  unit(), 
  apply_rotmat(), 
  matrix_times_vector(), 
  average_vectors(),
  cylindrical_to_cartesian(),
  apply_rotmat(),
  matrix_times_vector();


CYLINDRICAL_COORDS  
  cartesian_to_cylindrical();

double 
  dot_prod(), 
  VECTOR_length(), 
  dist(), 
  dist_to_line(), 
  find_angle(),
  find_bond_angle(), 
  find_dihedral(), 
  determinant(),
  degrees_to_radians(),
  radians_to_degrees(),
  dist_to_line();

#define GEOMETRY_H 1
#endif
