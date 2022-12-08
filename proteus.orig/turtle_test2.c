/*           @(#)turtle_test2.c	1.2        11/27/90           */
static char SCCS_ID[] = "@(#)turtle_test2.c	1.2 \t11/27/90";

#include "geometry.h"
#include "turtle.h"
#include "disulfide.h"

double ROTMAT[3][3];

TURTLE THE_TURTLE;

void setup_turtle()
{
  VECTOR n, ca, c;

  TURTLE_init(&THE_TURTLE,"The turtle is here");
  VECTOR_init(&n, -0.486, 1.366, 0.00);
  VECTOR_init(&ca, 0.0, 0.0, 0.0);
  VECTOR_init(&c, 1.53, 0.0, 0.0);
  
  TURTLE_orient(&THE_TURTLE, &ca, &c, &n);
  bbone_to_schain(&THE_TURTLE);
}

main()
{
  int i;
  int j;
  TURTLE egs, mo;
  VECTOR test;
  VECTOR n, ca, c, cb;
  DISULFIDE SS;
  DISULFIDE SS_array[100];
  int entries = 0;
  double rms = 0.0;


  VECTOR_init(&test, 10.0, 5.0, 1.0);

  TURTLE_init(&egs,"Eric");
  TURTLE_init(&mo,"Little Mo");

  
  printf("Enter SS iters: ");
  scanf("%d",&j);

/*

  printf("Rolling...\n");
  TURTLE_print(egs);
  TURTLE_roll(&egs,45.0);
  TURTLE_print(egs);
  TURTLE_roll(&egs,-45.0);
  TURTLE_print(egs);

  TURTLE_init(&egs,"Eric");
  printf("Pitching...\n");
  TURTLE_pitch(&egs,45.0);
  TURTLE_print(egs);
  TURTLE_pitch(&egs,-45.0);
  TURTLE_print(egs);

  TURTLE_init(&egs,"Eric");
  printf("TURTLE_Yawing...\n");
  TURTLE_yaw(&egs,45.0);
  TURTLE_print(egs);
  TURTLE_yaw(&egs,-45.0);
  TURTLE_print(egs);

  printf("\n***************************\n");
  schain_to_bbone(&egs);
  printf("Turtle after schain_to_bbone:\n");
  TURTLE_print(egs);

  bbone_to_schain(&egs);
  printf("Turtle after bbone_to_schain:\n");
  TURTLE_print(egs);

  printf("Vector test before conversion: ");
  VECTOR_print(&test);
  test = TURTLE_to_local(&egs,&test);
  printf("Vector test after conversion: ");
  VECTOR_print(&test);
  test = TURTLE_to_global(&egs,&test);
  printf("Vector test after conversion: ");
  VECTOR_print(&test);
*/

  setup_turtle();
  TURTLE_copy_coords(&egs,&THE_TURTLE);
  TURTLE_print(egs);
  
  build_residue(&egs, &n, &ca, &cb, &c);
  printf("N : ");
  VECTOR_print(&n);
  printf("CA : ");
  VECTOR_print(&ca);
  printf("CB : ");
  VECTOR_print(&cb);
  printf("C : ");
  VECTOR_print(&c);


  for (i = 0; i < j; i++) {
    char tmp[128];
    sprintf(tmp,"SS %d",i);
    DISULFIDE_init(&SS,tmp);
    DISULFIDE_random_build(&SS,egs);
/*    DISULFIDE_print(&SS); */
  }


  DISULFIDE_init(&SS,"hi there");
  DISULFIDE_set_conformation(&SS, -55.0, -53.0, -80.0, -66.0, -61.0);
  DISULFIDE_build(&SS,egs);
  DISULFIDE_print(&SS);


  entries = DISULFIDE_read_disulfide_database(DISULFIDE_DB,
				    &SS_array[0],100);
  printf("\nRead %d entries\n",entries);


/*  for (i = 0; i < entries; i++)
    DISULFIDE_print(&SS_array[i]);

  DISULFIDE_sort_array(&SS_array[0], entries,1);
  DISULFIDE_print_dihedral_array(SS_array,entries);

  rms = DISULFIDE_compare_dihedrals(&SS_array[0], &SS_array[1]);

  DISULFIDE_print(SS_array[0]);
  DISULFIDE_print(SS_array[1]);

  printf("RMS dihedral difference is %lf\n", rms);

  rms = DISULFIDE_compare_positions(&SS_array[0], &SS_array[1]);
  printf("RMS backbone difference is %lf\n", rms);
*/

}
