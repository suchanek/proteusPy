/*  $Id: turtle2.c,v 1.4 1995/06/25 02:44:01 suchanek Exp $  */
/*  $Log: turtle2.c,v $
 * Revision 1.4  1995/06/25  02:44:01  suchanek
 * added qsin, qcos and support functions use -DOLD_TURTLE to use the
 * original routines
 *
 * Revision 1.3  1995/06/23  00:10:39  suchanek
 * small change to use DEGRAD macro rather than function call
 *
 * Revision 1.2  1995/06/20  03:00:20  suchanek
 * function renaming
 *
 * Revision 1.1  1993/12/10  09:23:15  egs
 * Initial revision
 * */

static char rcsid[] = "$Id: turtle2.c,v 1.4 1995/06/25 02:44:01 suchanek Exp $";


#include <math.h>
#include "protutil.h"
#include "turtle.h"
#include "geometry.h"

#define ERROR -1
#define MIN_DEG -180
#define MAX_DEG 179

#define DEG_INCREMENT .1
/* 360 / DEG_INCREMENT */
#define SINCOS_ENTRIES (3600)
#define get_index(x) (((SINCOS_ENTRIES / 360) * (x + 180)))

static double sintab[SINCOS_ENTRIES+1];
static double costab[SINCOS_ENTRIES+1];

double my_qsin(double deg)
{
    int index = get_sincos_index(deg);

    return(sintab[index]);
}

double my_qcos(double deg)
{
    int index = get_sincos_index(deg);

    return(costab[index]);
}

int get_sincos_index(double deg)
{
    double d2 = deg;
    double d3 = 0.0;

    int ind = 0;

    while (d2 < -180.0)
	 d2 += 360.0;

    while (d2 > 180.0)
         d2 -= 360.0;

    d3 = rint((d2 * 10.0)) / 10;

    ind = (int)rint(get_index(d3));

    if (ind < 0 || ind > SINCOS_ENTRIES)
	{
	    fprintf(stderr,"Bad sincos index: %d \n", ind);
	    ind = 0;
	}
    return(ind);
}


void init_sincostables(void)
{
    double deg;
    int index;

    for (index = 0, deg = MIN_DEG; index < SINCOS_ENTRIES; index++, deg+=DEG_INCREMENT)
	{
	    sintab[index] = sin(DEGRAD(deg));
	    costab[index] = cos(DEGRAD(deg));
#ifdef NOISY2
	    fprintf(stderr,"index: %d, degrees: %lf \n", index, deg);
#endif
	}

	sintab[SINCOS_ENTRIES] = sin(0);
	sintab[SINCOS_ENTRIES] = cos(0);
}

/* Turtle stuff */

void TURTLE_copy_coords(dest,src)
     TURTLE *dest, *src;
{
  dest->position = src->position;
  dest->heading = src->heading;
  dest->left = src->left;
  dest->up = src->up;

}

void TURTLE_init(turtle,name)
     TURTLE *turtle;
     char *name;
{
  TURTLE_reset_coords(turtle);
  strcpy(turtle->name,name);
  TURTLE_pen_up(turtle);
}
  
void TURTLE_reset_coords (turtle)
TURTLE *turtle;
{
  VECTOR p, h, l, u;

  VECTOR_init(&p, 0.0, 0.0, 0.0);
  VECTOR_init(&h, 1.0, 0.0, 0.0);
  VECTOR_init(&l, 0.0, 1.0, 0.0);
  VECTOR_init(&u, 0.0, 0.0, 1.0);

  turtle->position = p;
  turtle->heading = h;
  turtle->left = l;
  turtle->up = u;
}


void TURTLE_print(turtle)
TURTLE turtle;
{
  printf("\nTurtle: <%s> \tPen is: %s",turtle.name, 
	 turtle.pen == UP ? "UP" : "DOWN");
  printf("\n\tPosition: \t%8.3lf, %8.3lf, %8.3lf",turtle.position.x, 
       turtle.position.y, 
       turtle.position.z);
  printf("\n\tHeading: \t%8.3lf, %8.3lf, %8.3lf", turtle.heading.x, 
	 turtle.heading.y, 
	 turtle.heading.z);
  printf("\n\tLeft:    \t%8.3lf, %8.3lf, %8.3lf", turtle.left.x, 
	 turtle.left.y, 
	 turtle.left.z);
  printf("\n\tUp:      \t%8.3lf, %8.3lf, %8.3lf\n",turtle.up.x, 
	 turtle.up.y, 
	 turtle.up.z);
}



void TURTLE_move(turtle,dist)
     TURTLE *turtle;
     double dist;

{
  turtle->position.x = turtle->position.x + dist * turtle->heading.x;
  turtle->position.y = turtle->position.y + dist * turtle->heading.y;
  turtle->position.z = turtle->position.z + dist * turtle->heading.z;
}


#ifdef OLD_TURTLE

void TURTLE_roll(turtle,angle)
     TURTLE *turtle;
     double angle;

{
  VECTOR lold, uold;

  double cosang, sinang;
  double ang;

  ang = DEGRAD(angle);
  cosang = cos(ang);
  sinang = sin(ang);

  lold = turtle->left;
  uold = turtle->up;

  turtle->up.x = cosang * uold.x - sinang * lold.x;
  turtle->up.y = cosang * uold.y - sinang * lold.y;
  turtle->up.z = cosang * uold.z - sinang * lold.z;

  turtle->left.x = cosang * lold.x + sinang * uold.x;
  turtle->left.y = cosang * lold.y + sinang * uold.y;
  turtle->left.z = cosang * lold.z + sinang * uold.z;
}

void TURTLE_yaw(turtle,angle)
     TURTLE *turtle;
     double angle;

{

  double sinang = sin((180.0 - angle) / 57.2958);
  double cosang = cos((180.0 - angle) / 57.2958);

  VECTOR lold, hold;

  lold = turtle->left;
  hold = turtle->heading;

  turtle->heading.x = cosang * hold.x + sinang * lold.x;
  turtle->heading.y = cosang * hold.y + sinang * lold.y;
  turtle->heading.z = cosang * hold.z + sinang * lold.z;

  turtle->left.x = cosang * lold.x - sinang * hold.x;
  turtle->left.y = cosang * lold.y - sinang * hold.y;
  turtle->left.z = cosang * lold.z - sinang * hold.z;
}


void TURTLE_pitch(turtle,angle)
     TURTLE *turtle;
     double angle;

{
  double ang,sinang,cosang;
  VECTOR hold, uold;

  ang = DEGRAD(angle);

  cosang = cos(ang);
  sinang = sin(ang);

  hold = turtle->heading;
  uold = turtle->up;

  turtle->heading.x = hold.x * cosang - uold.x * sinang;
  turtle->heading.y = hold.y * cosang - uold.y * sinang;
  turtle->heading.z = hold.z * cosang - uold.z * sinang;

  turtle->up.x = uold.x * cosang + hold.x * sinang;
  turtle->up.y = uold.y * cosang + hold.y * sinang;
  turtle->up.z = uold.z * cosang + hold.z * sinang;
}


#else

/*
  this version assumes the sin and cosine tables have already
  been initialized!
*/


void TURTLE_roll(turtle,angle)
     TURTLE *turtle;
     double angle;

{
  VECTOR lold, uold;

  double cosang, sinang;
  double ang;

  ang = angle;
  while (ang < -180)
      ang += 360.0;

  cosang = my_qcos(ang);  /* converted to radians already */
  sinang = my_qsin(ang);  /* converted to radians already */

  lold = turtle->left;
  uold = turtle->up;

  turtle->up.x = cosang * uold.x - sinang * lold.x;
  turtle->up.y = cosang * uold.y - sinang * lold.y;
  turtle->up.z = cosang * uold.z - sinang * lold.z;

  turtle->left.x = cosang * lold.x + sinang * uold.x;
  turtle->left.y = cosang * lold.y + sinang * uold.y;
  turtle->left.z = cosang * lold.z + sinang * uold.z;
}

void TURTLE_yaw(turtle,angle)
     TURTLE *turtle;
     double angle;

{

  double sinang;
  double cosang;
  double ang;

  VECTOR lold, hold;

  ang = angle;
  while (ang < -180.0)
      ang += 360.0;

  cosang = my_qcos((180.0 - ang));
  sinang = my_qsin((180.0 - ang));

  lold = turtle->left;
  hold = turtle->heading;

  turtle->heading.x = cosang * hold.x + sinang * lold.x;
  turtle->heading.y = cosang * hold.y + sinang * lold.y;
  turtle->heading.z = cosang * hold.z + sinang * lold.z;

  turtle->left.x = cosang * lold.x - sinang * hold.x;
  turtle->left.y = cosang * lold.y - sinang * hold.y;
  turtle->left.z = cosang * lold.z - sinang * hold.z;
}


void TURTLE_pitch(turtle,angle)
     TURTLE *turtle;
     double angle;

{
  double ang,sinang,cosang;
  VECTOR hold, uold;

  ang = DEGRAD(angle);

  cosang = my_qcos(ang);
  sinang = my_qsin(ang);

  hold = turtle->heading;
  uold = turtle->up;

  turtle->heading.x = hold.x * cosang - uold.x * sinang;
  turtle->heading.y = hold.y * cosang - uold.y * sinang;
  turtle->heading.z = hold.z * cosang - uold.z * sinang;

  turtle->up.x = uold.x * cosang + hold.x * sinang;
  turtle->up.y = uold.y * cosang + hold.y * sinang;
  turtle->up.z = uold.z * cosang + hold.z * sinang;
}


#endif




void TURTLE_turn(turtle,angle)
     TURTLE *turtle;
     double angle;

{
  double ang;
  double sinang, cosang;

  VECTOR hold, lold;

  hold = turtle->heading;
  lold = turtle->left;

  ang = degrees_to_radians(angle);
  sinang = sin(ang);
  cosang = cos(ang);

  turtle->heading.x = cosang * hold.x + sinang * lold.x;
  turtle->heading.y = cosang * hold.y + sinang * lold.y;
  turtle->heading.z = cosang * hold.z + sinang * lold.z;

  turtle->left.x = cosang * lold.x - sinang * hold.x;
  turtle->left.y = cosang * lold.y - sinang * hold.y;
  turtle->left.z = cosang * lold.z - sinang * hold.z;

}

void TURTLE_orient(turtle,p1,p2,p3)
     TURTLE *turtle;
     VECTOR *p1,*p2,*p3;

{
  VECTOR temp;

  turtle->position.x = p1->x; 
  turtle->position.y = p1->y; 
  turtle->position.z = p1->z;

  temp = VECTOR_difference(p2,p1);
  turtle->heading = unit(&temp);
  turtle->left = VECTOR_difference(p3,p1); /* temporary left */

  temp = cross_prod(&turtle->heading,&turtle->left);
  turtle->up = unit(&temp);

  temp = cross_prod(&turtle->up,&turtle->heading);
  turtle->left = unit(&temp); /* now left is really correct */
}


void TURTLE_pen_up(turtle)
TURTLE *turtle;
{
  turtle->pen = UP;
}

void TURTLE_pen_down(turtle)
TURTLE *turtle;
{
  turtle->pen = DOWN;
}

VECTOR TURTLE_to_local(turtle,global_vec)
     TURTLE *turtle;
     VECTOR *global_vec;

{
  VECTOR result;
  VECTOR new;


  double dp1;
  double dp2;
  double dp3;

  new  = VECTOR_difference(global_vec,&turtle->position);

  dp1 = dot_prod(&turtle->heading, &new);
  dp2 = dot_prod(&turtle->left, &new);
  dp3 = dot_prod(&turtle->up, &new);

  result.x = dp1; result.y = dp2; result.z = dp3;
  return(result);
}

VECTOR TURTLE_to_global(turt,loc)
     TURTLE *turt;
     VECTOR *loc;

{
   VECTOR result;

   double p1 = turt->position.x + turt->heading.x * loc->x + turt->left.x * loc->y + turt->up.x * loc->z;
   double p2 = turt->position.y + turt->heading.y * loc->x + turt->left.y * loc->y + turt->up.y * loc->z;
   double p3 = turt->position.z + turt->heading.z * loc->x + turt->left.z * loc->y + turt->up.z * loc->z;

   result.x = p1; result.y = p2; result.z = p3;
   return(result);
}


       
