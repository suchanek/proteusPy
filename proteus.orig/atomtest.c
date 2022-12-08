/* $Id: atomtest.c,v 1.3 1995/06/20 03:00:15 suchanek Exp $ */
/* $log$ */

#include "turtle.h"
#include "residue.h"
#include "atom.h"
#include "phipsi.h"
#include "distmat.h"

#include <stdio.h>

#define POPSIZE 2

void initializepopulation(), PHIPSI_cpy();

PHIPSI *newpop[POPSIZE];
PHIPSI *oldpop[POPSIZE];

main(argc, argv)
     int argc;
     char *argv[];
{
  DISTMAT *dm;
  DISTMAT *dm2;
  DISTMAT *dm3;
  double rms;

  ATOM *patom;
  ATOM *new_atoms;
  ATOM *sphere;
  ATOM *foundit;
  PHIPSI *phi_psi;
  TURTLE turt;



  ATOM atom, atom2;
  int len = 0;
  int lores, hires;
  int i;


  TURTLE_init(&turt, "Eric");
  patom = new_ATOM_list(100);
  len = ATOM_list_length(patom);
  printf("Length is %d\n", len);
  
  
  ATOM_make_radii(patom);
  ATOM_make_atom_types(patom);
  ATOM_make_colors(patom);
  
  ATOM_init(&atom,"ca","ALA", 0, 1, 1.0, 0.0, 0.0,10.0);
  
  ATOM_print(&atom);
  ATOM_find_collisions(&atom,patom);
  ATOM_print(&atom);
  
/*
  sphere = ATOM_sphere(&atom,patom, 10.0);
  len = ATOM_list_length(sphere);
  printf("Sphere Length is %d\n", len);
  ATOM_list_print(sphere);
  
  free_ATOM_list(sphere);
    
  ATOM_exclude_backbone(patom,3);
  ATOM_range_print(patom,0,6);
  printf("**********************************\n");
  
  ATOM_exclude_sidechain(patom,3);
  ATOM_range_print(patom,0,6);
  printf("**********************************\n");
  
  ATOM_include_backbone(patom,3);
  ATOM_range_print(patom,0,6);
  printf("**********************************\n");
  
  ATOM_include_sidechain(patom,3);
  ATOM_range_print(patom,0,6);
  printf("**********************************\n");
  

  printf("Length is %d\n", len);
*/  
  
  foundit = ATOM_list_scan(patom, 4, "SG");
  if (foundit) {
      fprintf(stderr,"Found it! <1>\n");
      ATOM_print(foundit);
  }
  else		
      fprintf(stderr,"Can't find Residue %d, Type %s\n",
	      4,"CYS","SG");
  free_ATOM_list(patom);
  

  if (argc > 1)
    {
      len = ATOM_read_pdb(&patom,argv[1]);
      if (len <= 0) return(-1);
      fprintf(stderr,"\nRead %d atoms\n",len);
    }
  else
    {
      len = ATOM_read_pdb(&patom,"tst.pdb");
      if (len <= 0) return(-1);
      fprintf(stderr,"\nRead %d atoms\n",len);
    }
  
  foundit = ATOM_list_scan(patom, 3, "SG");
  if (foundit) {
      fprintf(stderr,"Found it! <1>\n");
      ATOM_print(foundit);
  }
  else		
      fprintf(stderr,"Can't find Residue %d, Type %s\n",
	      3,"CYS","SG");
  
  
  sphere = ATOM_sphere(foundit,patom, 3.0);
  len = ATOM_list_length(sphere);
  printf("Sphere Length is %d\n", len);
  ATOM_list_print(sphere);

  printf("1**********************************\n");
  ATOM_exclude_backbone(patom,3);
  ATOM_range_print(patom,0,10);
  printf("2**********************************\n");
  
  ATOM_exclude_sidechain(patom,3);
  ATOM_range_print(patom,0,10);
  printf("3**********************************\n");
  
  ATOM_include_backbone(patom,3);
  ATOM_range_print(patom,0,10);
  printf("4**********************************\n");
  
  ATOM_include_sidechain(patom,3); 
  ATOM_range_print(patom,0,10);

  printf("5**********************************\n");
  
  ATOM_include_all(patom,3);

  RESIDUE_phi_psi_list_build(patom,&phi_psi);
  fprintf(stderr,"N phi psis: %d\n",len);
  len = PHIPSI_list_length(phi_psi);
  fprintf(stderr,"N phi psis: %d\n",len);
  
  RESIDUE_range(patom,&lores,&hires);
  PHIPSI_list_print(phi_psi);
  
  /* orient along backbone for build */

  len = orient_at_residue(&turt, patom, lores+1, ORIENT_BACKBONE);
  if (len)
      TURTLE_print(turt);
  else
      fprintf(stderr,"Can't orient 1??\n");
  
  new_atoms = (ATOM *)RESIDUE_build_backbone_from_phipsi(phi_psi,&turt);
  if (new_atoms)
      {
	  fprintf(stderr,"\nBuilt %d atoms.\n",ATOM_list_length(new_atoms));
	  ATOM_list_print(new_atoms); 
	  free_ATOM_list(new_atoms);
      }
  else
      fprintf(stderr,"\nCould not build atom array from Phi-Psis?\n");
  
  
  
  dm = DISTMAT_build_from_atom_list(patom,"test", MAX_CUTOFF, 1);
  DISTMAT_print(dm);
  
  dm2 = DISTMAT_build_from_phipsi_list(phi_psi, "test2", MAX_CUTOFF);
  DISTMAT_print(dm2); 
  
  rms = DISTMAT_rms(dm,dm);
  fprintf(stderr,"\nRMS 1: %lf\n",rms);
  
  rms = DISTMAT_rms(dm,dm2);
  fprintf(stderr,"RMS 2: %lf\n",rms);

  dm3 = DISTMAT_difference(dm, dm2, "my diff");
  DISTMAT_print(dm3);

  free_DISTMAT(dm);
  free_DISTMAT(dm2);
  free_DISTMAT(dm3);
  free_ATOM_list(patom);
  free_PHIPSI_list(phi_psi);
  free_ATOM_list(sphere);


/*
  initializepopulation(10);
  PHIPSI_list_print(oldpop[0]);
  printf("\n-----------------------\n");

  PHIPSI_print(oldpop[0]);
  phi_psi[0].resnumb = -1; phi_psi[0].phi = 1234.0;

  PHIPSI_print(phi_psi[0]);
*/
}

#define dbg2 1 

void PHIPSI_cpy(dst,src)
PHIPSI *dst, *src;
{
#ifdef dbg2
  printf("\nBefore cpy, dest is:\n");
  PHIPSI_print(dst);
#endif

  dst->resnumb = src->resnumb;
  dst->numelem = src->numelem;
  dst->phi = src->phi;
  dst->psi = src->psi;
  dst->omega = src->omega;

#ifdef dbg2
  printf("\nAfter cpy, dest is:\n");
  PHIPSI_print(dst);
#endif
}

void initializepopulation(len)
     int len;
{
  int i;

  for (i=0;i<POPSIZE;i++) 
    {

      oldpop[i] = new_PHIPSI_list(len);
      if (oldpop[i] == (PHIPSI *)NULL)
	{
	  fprintf(stderr,"\nCan't allocate oldpop PHIPSI %d?\n",i);
	  exit(-2);
	}
      PHIPSI_random_init(oldpop[i]);

      newpop[i] = (PHIPSI *)new_PHIPSI_list(len);
      if (newpop[i] == (PHIPSI *)NULL)
	{
	  fprintf(stderr,"\nCan't allocate newpop PHIPSI %d?\n",i);
	  exit(-2);
	}
      PHIPSI_random_init(newpop[i]);

    }
}
