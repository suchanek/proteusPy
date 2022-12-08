/*           @(#)geotest.c	1.3        3/29/91           */
static char SCCS_ID[] = "@(#)geotest.c	1.3 \t3/29/91";

/*************************************************************************/

#include <stdio.h>
#include <math.h>
#include "geometry.h"



main ()

{



/*  Declare Local variables */
 double length1,dp,length2;
 double distance,x,y,z,bangle,dihed;

 VECTOR AA,BB,CC,GG,HH,VD,VS;
 VECTOR orig;


 orig.x = 0.0;orig.y= 0.0; orig.z =  0.0;



 printf("\nEnter first vector: ");
 scanf("%lf %lf %lf",&x,&y,&z);
 AA.x = x; AA.y = y; AA.z = z;

 printf("Enter second vector: ");
 scanf("%lf %lf %lf",&x,&y,&z);
 BB.x = x; BB.y = y; BB.z = z;

 printf("Enter third vector: ");
 scanf("%lf %lf %lf",&x,&y,&z);
 GG.x = x; GG.y = y; GG.z = z;

 printf("Enter fourth vector: ");
 scanf("%lf %lf %lf",&x,&y,&z);
 HH.x = x; HH.y = y; HH.z = z;

 printf("Vector A:");
 VECTOR_print(&AA);

 dp = dot_prod(&AA,&BB);
 printf("\nThe Dot Product is: %lf",dp);

 CC = cross_prod(&AA,&BB);
 printf("\nThe Cross Product is: %lf, %lf, %lf",CC.x,CC.y,CC.z);

 VD = VECTOR_difference(&AA,&BB);
 printf("\n Vector Diff: %lf, %lf, %lf",VD.x,VD.y,VD.z);

 VS = VECTOR_sum(&AA,&BB);
 printf("\n Vector Sum: %lf, %lf, %lf",VS.x,VS.y,VS.z);

 length1 = VECTOR_length(&AA);
 length2 = VECTOR_length(&BB);
 printf("\n Length A: %lf \n Length b: %lf",length1,length2);

 distance = dist(&AA,&BB);
 printf("\n Distance between A and B is: %lf",distance);

 CC = scalar_times_vector(5.0,&AA);
 printf("\n 5.0 *  A is: %f, %f, %f\n",CC.x,CC.y,CC.z);

 CC = unit(&AA);
 printf("\nUNIT A = %lf, %lf, %lf",CC.x,CC.y,CC.z);

 bangle = find_angle(&AA,&BB);
 printf("\nThe angle between A and B is %lf",bangle);

 bangle = find_bond_angle(&AA,&orig,&BB);
 printf("\nThe bond angle between A and B is %lf",bangle);

 dihed = find_dihedral(&AA,&BB,&GG,&HH);
 printf("\nDihedral was: %lf\n",dihed);

 
} /* end of main routine */
