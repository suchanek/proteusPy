/*  $Id: geometry.c,v 1.4 1995/06/23 00:09:56 suchanek Exp $  */
/*  $Log: geometry.c,v $
 * Revision 1.4  1995/06/23  00:09:56  suchanek
 * *** empty log message ***
 *
 * Revision 1.3  1995/06/20  03:00:29  suchanek
 * function renaming
 *
 * Revision 1.2  1995/06/11  05:13:10  suchanek
 * changed VECTOR_print to use pointers
 *
 * Revision 1.1  1993/12/10  09:23:21  egs
 * Initial revision
 * */

static char rcsid[] = "$Id: geometry.c,v 1.4 1995/06/23 00:09:56 suchanek Exp $";

#include "protutil.h"
#include "geometry.h"

double ROTMAT[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

/*            Here are the geometry function primitives       */


void VECTOR_init(v, x, y, z)
     VECTOR *v;
     double x, y, z;
{
  v->x = x;
  v->y = y;
  v->z = z;
}

void VECTOR_print(VECTOR *v)
{
  printf("\t%8.2lf %8.2lf %8.2lf\n",
	 v->x, v->y, v->z);
}

double dot_prod(u,v)  /* Calculate the inner product of two VECTORS */
  VECTOR *u,*v;
{
   return( u->x * v->x + u->y * v->y + u->z * v->z);
}




VECTOR cross_prod(u,v) /* Calculate the cross product of two VECTORS */
  VECTOR *u,*v;
{
   VECTOR temp;
   temp.x = u->y * v->z - u->z * v->y;
   temp.y = u->z * v->x - u->x * v->z;
   temp.z = u->x * v->y - u->y * v->x;
   return (temp);
}



VECTOR VECTOR_difference(u,v) /* Calc VECTOR_difference, i.e. U minus V */
  VECTOR *u,*v;
{
   VECTOR result;
   result.x = u->x - v->x;
   result.y = u->y - v->y;
   result.z = u->z - v->z;
   return(result);
}



double VECTOR_length (u) /* Calc. the length of a vector */
  VECTOR *u;

{
   return (sqrt(dot_prod(u,u)));
}



VECTOR VECTOR_sum(u,v) /* Return the sum of two vectors */
  VECTOR *u,*v;

{
  VECTOR result;

  result.x = u->x + v->x;
  result.y = u->y + v->y;
  result.z = u->z + v->z;
  return(result);
}



VECTOR scalar_times_vector(s,v) /* Return a vector for the resulting product */
  double s;
  VECTOR *v;

{
  VECTOR result;

  result.x = s * v->x;
  result.y = s * v->y;
  result.z = s * v->z;
  return(result);
}


VECTOR average_vectors(a,b)
  VECTOR *a,*b;

{
  VECTOR result;

  result = VECTOR_sum(a,b);
  result = scalar_times_vector(0.5,&result);
  return(result);
}



double dist(u,v) /* Return the distance between two VECTORS or points */
 VECTOR *u,*v;

{
   double x2 = (u->x - v->x);
   double y2 = (u->y - v->y);
   double z2 = (u->z - v->z);

   return(sqrt((x2 * x2) + (y2 * y2) + (z2 * z2)));
}



VECTOR unit (u)
  VECTOR *u;
{
   double length;
   VECTOR unit_vec;

   length=  (VECTOR_length(u));
  
   if (length == 0.0) length = 0.001; /* Can't divide by 0 */
  
   unit_vec = scalar_times_vector((1.0 / length),u);
   return(unit_vec);
}



double find_angle (v1,v2)
  VECTOR *v1,*v2;

{
   VECTOR vec1,vec2;
   double dotprod;
   double radians_to_degrees();


   vec1 = unit(v1);
   vec2 = unit(v2);

   dotprod = dot_prod(&vec1,&vec2);
   if (dotprod > 1.0)  dotprod = 1.0;
   if (dotprod < -1.0) dotprod = -1.0;
   return(radians_to_degrees(acos(dotprod)));
}



double find_bond_angle (p1,p2,p3)
  VECTOR *p1,*p2,*p3;
{
   VECTOR v1,v2;
   v1 = VECTOR_difference(p1,p2);
   v2 = VECTOR_difference(p3,p2);
   return(find_angle(&v1,&v2));
}



double find_dihedral(p1,p2,p3,p4)
  VECTOR *p1,*p2,*p3,*p4;

{
   VECTOR v1,v2,v3,v4,a,b;
   VECTOR empty; 
   double result;

   empty.x = 0.0; empty.y = 0.0; empty.z = 0.0;
   v1 = VECTOR_difference(p3,p2);
   v2 = VECTOR_difference(p1,p2);
   v3 = VECTOR_difference(p4,p3);
   v4 = VECTOR_difference(p2,p3);
   a = cross_prod(&v1,&v2);
   b = cross_prod(&v3,&v4);
   result = find_bond_angle(&a,&empty,&b);
   if (dot_prod(&a,&v3) < 0.0) return (-1.0 * result);
   return(result);
}


/************************ Unit Conversion Stuff ***********************/

double radians_to_degrees(rad)
 double rad;

{   
   return(rad * 180.0 / PI);
}




double degrees_to_radians(degrees)
 double degrees;
{
 return(degrees * PI / 180.0);
}



VECTOR cylindrical_to_cartesian (cylinder_coords)
   CYLINDRICAL_COORDS *cylinder_coords;

{
   double r = cylinder_coords->r;
   double theta = degrees_to_radians(cylinder_coords->theta);
   double z = cylinder_coords->z;
   VECTOR result;

   result.x = r * cos(theta);
   result.y = r * sin(theta);
   result.z = z;

   return(result);
}

   CYLINDRICAL_COORDS cartesian_to_cylindrical(cartesian_coords)
  VECTOR *cartesian_coords;

{
   CYLINDRICAL_COORDS result;
   double x = cartesian_coords->x;
   double y = cartesian_coords->y;
   double z = cartesian_coords->z;
   double th;

   th = radians_to_degrees(atan2(y,x));
   result.theta = (th < 0.0) ? (th + 360.0) : (th - 360.0);
   result.r = sqrt(x * x + y * y);
   result.z = z;
}


/**********************  Rotation Matrix Operators **********************/


void print_rotmat()
{
  extern double ROTMAT[3][3];
   printf("\nROTMAT is currently:\n     %lf  %lf  %lf\n",ROTMAT[0][0],ROTMAT[0][1],ROTMAT[0][2]);
   printf("     %lf  %lf  %lf\n",ROTMAT[1][0],ROTMAT[1][1],ROTMAT[1][2]);
   printf("     %lf  %lf  %lf\n",ROTMAT[2][0],ROTMAT[2][1],ROTMAT[2][2]);
}



void make_rotmat(direction_cosines, angle)
 VECTOR *direction_cosines;
 double angle;
{
  extern double ROTMAT[3][3];

   double sinang = sin(degrees_to_radians(angle));
   double cosang = cos(degrees_to_radians(angle));
   double l1 = direction_cosines->x;
   double l2 = direction_cosines->y;
   double l3 = direction_cosines->z;

   ROTMAT[0][0] = cosang + l1 * l1 * (1.0 - cosang);
   ROTMAT[0][1] = l1 * l2 * (1.0 - cosang) - (l3 * sinang);
   ROTMAT[0][2] = l3 * l1 * (1.0 - cosang) + (l2 * sinang);

   ROTMAT[1][0] = l1 * l2 * (1.0 - cosang) + (l3 * sinang);
   ROTMAT[1][1] = cosang + l2 * l2 * (1.0 - cosang);
   ROTMAT[1][2] = l2 * l3 * (1.0 - cosang) - (l1 * sinang);

   ROTMAT[2][0] = l3 * l1 * (1.0 - cosang) - (l2 * sinang);
   ROTMAT[2][1] = l2 * l3 * (1.0 - cosang) + (l1 * sinang);
   ROTMAT[2][2] = cosang + l3 * l3 * (1.0 - cosang);
}



VECTOR apply_rotmat (point, origin)
  VECTOR *point, *origin;

{
  extern double ROTMAT[3][3];

  double x = point->x - origin->x;
  double y = point->y - origin->y;
  double z = point->z - origin->z;
  VECTOR result;

  result.x = origin->x + ROTMAT[0][0] * x + ROTMAT[0][1] * y + ROTMAT[0][2] * z;
  result.y = origin->y + ROTMAT[1][0] * x + ROTMAT[1][1] * y + ROTMAT[1][2] * z;
  result.z = origin->z + ROTMAT[2][0] * x + ROTMAT[2][1] * y + ROTMAT[2][2] * z;

  return(result);
}


VECTOR matrix_times_vector(matrix,vector_name)
  double matrix[3][3];
  VECTOR *vector_name;

{

  double x = vector_name->x;
  double y = vector_name->y;
  double z = vector_name->z;
  VECTOR result;

  result.x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z;
  result.y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z;
  result.z = matrix[2][0] * x + matrix[2][1] * y + matrix [2][2] * z;

  return(result);
}


void matrix_times_matrix(a,b,result)
 double a[3][3],b[3][3],result[3][3];

{
  int i,j,k;

  for (i = 0, j = 0; i < 3; i++, j++) {
result[i][j] = 0.0;
for (k = 0; k < 3; k++) {
   result[i][j] = result[i][j] + a[i][k] * b[k][j];
    }
  }
}



double determinant(a)
  double a[3][3];
{
  double dt;
 
  dt = a[0][0] * a[1][1] * a[2][2];
  dt = dt + a[0][1] * a[1][2] * a[2][0];
  dt = dt + a[0][2] * a[2][1] * a[1][0];
  dt = dt - a[0][2] * a[1][1] * a[2][0];

  return(dt);
}



double dist_to_line (point, line_origin, line_direction)
  VECTOR *point, *line_origin, *line_direction;

{
  double length = dist(point, line_origin);
  VECTOR vector2,dir;
  double theta;

  vector2 = VECTOR_difference(point, line_origin);
  vector2 = unit(&vector2);
  dir = unit(line_direction);
  theta = acos(dot_prod(&dir,&vector2));
  return(length * sin(theta));
}
