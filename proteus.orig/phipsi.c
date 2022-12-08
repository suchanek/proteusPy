/*  $Id: phipsi.c,v 1.1 1993/12/10 09:22:03 egs Exp $  */
/*  $Log: phipsi.c,v $
 * Revision 1.1  1993/12/10  09:22:03  egs
 * Initial revision
 * */

static char rcsid[] = "$Id: phipsi.c,v 1.1 1993/12/10 09:22:03 egs Exp $";


#include "protutil.h"
#include "phipsi.h"
#include "phipsi_all.h"

#undef DBG

/* ---------------  -------------- */

/* --------------- Memory Allocation and Initialization -------------- */

/*
 * Function allocates space for an phipsi list of NH elements.
 * It also calls PHIPSI_list_init to fill in default information
 * in the structure
 */

PHIPSI *new_PHIPSI_list(nh)
int nh;
{
	PHIPSI *v;
	int nl = 0;

	v=(PHIPSI *)malloc((unsigned) (nh-nl+1)*sizeof(PHIPSI));
	if (!v) Proteus_error("allocation failure in PHIPSI_list()");
	PHIPSI_list_init(v,nh);
	return v;
}

/*
 * Function frees space allocated to an PHIPSI list.
 *
 */

void free_PHIPSI_list(v)
PHIPSI *v;
{
  if (v != (PHIPSI *)NULL)
    free((char*) (v));
}

/*
 * Function initializes a specific PHIPSI with its name, number, and
 * position. It calls the necessary routines to parse the atype, radius,
 * charge, and color.
 */

void PHIPSI_init(phipsi, phi, psi, omega, resnumb)
     PHIPSI *phipsi;
     double phi, psi, omega;
     int resnumb;
{
  phipsi->resnumb = resnumb;
  phipsi->phi = phi;
  phipsi->psi = psi;
  phipsi->omega = omega;
}

/*
 * Function initializes an entire list of phipsis to default values.
 *
 */

void PHIPSI_list_init(phipsi, nh)
PHIPSI *phipsi;
int nh;
{
  int nl = 0;
  int i, j;

  for(i = nl; i < nh; i++) {
    phipsi[i].numelem = nh;
    phipsi[i].resnumb = i;
    phipsi[i].phi = -999.9;
    phipsi[i].psi = -999.9;
    phipsi[i].omega = -999.9;
  }
}

/*
 * Function returns the current phipsi list length. The list must have been
 * initialized first!
 */

int PHIPSI_list_length(phipsis)
PHIPSI *phipsis;
{
  int len = phipsis->numelem;
  return(len);
}


/* --------------- Writing the Structure Contents  -------------- */

/*
 * Function prints out the properties of a specific phipsi.
 *
 */
  
void PHIPSI_pprint(v)
PHIPSI *v;
{

    printf("PhiPsi %d:\n\tPhi: %7.2lf\n\tPsi: %7.2lf\n\tOmega: %7.2lf\n\n",
	   v->resnumb, v->phi, v->psi, v->omega);
}

void PHIPSI_print(v)
PHIPSI *v;
{

    printf("%6d: %7.2lf %7.2lf %7.2lf\n",
	   v->resnumb, v->phi, v->psi, v->omega);
}

/*
 * Function prints out the stats for an entire list of phipsis.
 *
 */

void PHIPSI_list_print(v)
PHIPSI *v;
{
  int i, j;
  int nl = 0;
  int nh = 0;

  nh = PHIPSI_list_length(v);

#ifdef DBG
	fprintf(stderr,"\n PHIPSI length is %d \n",nh);
#endif
  for (i = 0; i < nh; i++) {
    PHIPSI_print(&v[i]);
  }

}

/*
 * Function prints out the stats for a specific range of PHIPSIs in an
 * PHIPSI list.
 */

void PHIPSI_range_print(v,nl,nh)
PHIPSI *v;
int nl, nh;
{
  int i, j;

  int atnum_max = PHIPSI_list_length(v);

  if (nl < 0 || nl >= nh) {
    fprintf(stderr,"PHIPSI_range_print - error, NL <%d> out of range...\n",nl);
    nl = 0;
  }

  if (nh > atnum_max) {
    fprintf(stderr,"PHIPSI_range_print - error, NH <%d> out of range...\n",nh);
    nh = atnum_max;
  }

  for (i = nl; i < nh; i++) {
    PHIPSI_print(&v[i]);
  }

}


/* 
 * Function returns a pointer to an phipsi within the input phipsi_list
 * specified by the residue number (resnum), phipsi name (atnam),
 * and amino acid type (amino).
 *
 */

PHIPSI *PHIPSI_list_scan(phipsi_list, resnum)
     PHIPSI *phipsi_list;
     int resnum;
{

  int len;
  int i;

  len = PHIPSI_list_length(phipsi_list);

  for (i = 0; i < len; i++ ) {
    if (resnum == phipsi_list[i].resnumb)
      return(&phipsi_list[i]);
  }
  return((PHIPSI *)NULL);

}


/* ----------- Choose random dihedral between MINVAL and MAXVAL ----------*/
/* -----------------------------------------------------------------------*/

void PHIPSI_choose_random_dihedrals(phi,psi,omega)
     double *phi, *psi, *omega;
{
  *phi = ((double)(rand() % _ps_range) + MINVAL);
  *psi = ((double)(rand() % _ps_range) + MINVAL);
  *omega = ((double)(rand() % _omega_range) + MINOMEGA);
}

void PHIPSI_choose_random_allowed_dihedrals(phi,psi,omega)
     double *phi, *psi, *omega;
{
  int index = rand() % N_PHIPSI;
  
  *phi = phipsi_all[index][0];
  *psi = phipsi_all[index][1];
  *omega = ((double)(rand() % _omega_range) + MINOMEGA);
}

int PHIPSI_random_init(ps)
     PHIPSI *ps;
{
  int len = 0;
  int res = 0;
  int i;

  double phi, psi, omega;

  len = PHIPSI_list_length(ps);
  if (len <= 0)
    {
      fprintf(stderr,"\nPHIPSI_random_init() - can't get phipsi length?\n");
      return(res);
    }

  for (i = 0; i < len; i++)
    {
      PHIPSI_choose_random_dihedrals(&phi,&psi,&omega);
      ps[i].phi = phi;
      ps[i].psi = psi;
      ps[i].omega = omega;
    }
  return(1);
}
