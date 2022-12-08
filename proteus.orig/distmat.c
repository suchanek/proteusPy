/*  $Id: distmat.c,v 1.3 1995/06/23 00:09:11 suchanek Exp $  */
/*  $Log: distmat.c,v $
 * Revision 1.3  1995/06/23  00:09:11  suchanek
 * added DISTMAT_Build_from_backbone, and optimized a bit
 *
 * Revision 1.2  1995/06/20  01:11:49  suchanek
 * added an access function to get a specific distance out
 *
 * Revision 1.1  1993/12/10  09:21:49  egs
 * Initial revision
 * */

static char rcsid[] = "$Id: distmat.c,v 1.3 1995/06/23 00:09:11 suchanek Exp $";



#include "protutil.h"
#include "distmat.h"

#undef DBG

/* ---------------  -------------- */

/* --------------- Memory Allocation and Initialization -------------- */


static double **dmatrix(nrl,nrh,ncl,nch)
int nrl,nrh,ncl,nch;
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	int i;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*))-nrl;
	if (!m) Proteus_error("allocation failure 1 in dmatrix()");

	/* allocate rows and set pointers to them */
	for(i=nrl;i<=nrh;i++) {
		m[i]=(double *) malloc((unsigned) (nch-ncl+1)*sizeof(double))-ncl;
		if (!m[i]) Proteus_error("allocation failure 2 in dmatrix()");
	}
	/* return pointer to array of pointers to rows */
	return m;
}

static void free_dmatrix(m,nrl,nrh,ncl,nch)
double **m;
int nrl,nrh,ncl,nch;
/* free a double matrix allocated by dmatrix() */
{
	int i;

	for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
	free((char*) (m+nrl));
}

/*
 * Function allocates space for a distance matrix of (NH - NL + 1)^2 elements.
 * It also fills in the slots with the matrix name, cutoff distance and other
 * structure elements.
 *
 * returns (DISTMAT *)NULL on failure, a properly initialized DISTMAT * 
 * structure otherwise.
 */

DISTMAT *new_DISTMAT(nl,nh,name,cutoff)
     int nl;
     int nh;
     char *name;
     double cutoff;
{
	DISTMAT *v;
	int len = 0;

	v=(DISTMAT *)malloc((unsigned) sizeof(DISTMAT));
	if (!v) 
	  Proteus_error("allocation failure in new_DISTMAT()");

	len = strlen(name);
	v->name = (char *)malloc((unsigned) len+1);
	if (v->name == (char *)NULL) 
	  Proteus_error("allocation failure 0 in new_DISTMAT()");
	strncpy(v->name,name,len);
	v->name[len] = '\0';

	v->matrix = dmatrix(nl,nh,nl,nh);
	if (!v->matrix) 
	  Proteus_error("allocation failure 2 in new_DISTMAT()");

	v->lores = nl; v->hires = nh;
	v->cutoff = cutoff;
	DISTMAT_init(v);  /* set all distances to 0.0 */

	return (DISTMAT *)v;
}

/*
 * Function frees space allocated to an DISTMAT list.
 *
 */

void free_DISTMAT(v)
DISTMAT *v;
{
  if (v != (DISTMAT *)NULL)
    {
      free_dmatrix(v->matrix,v->lores,v->hires,v->lores,v->hires);
      free((char *)v->name);
      free((char*) (v));
      v = (DISTMAT *)NULL;
    }
}

/*
 * Function initializes all distances in the specified distmat to 0.0.
 */

void DISTMAT_init(dm)
     DISTMAT *dm;
{
  if (dm != (DISTMAT *)NULL)
    {
      int lores = dm->lores;
      int hires = dm->hires;
      int row, col;
      
      for (row = lores; row <= hires; row++)
	for (col = lores; col <= hires; col++)
	  dm->matrix[row][col] = 0.0;
    }
  else
    fprintf(stderr,"\nDISTMAT_init() received a NULL distmat??\n");
}

/*
 * Set the Distance matrix cutoff slot to the value cutoff.
 * Return 1 if success, 0 otherwise.
 *
 */

int DISTMAT_cutoff(v,cutoff)
     DISTMAT *v;
     double cutoff;
{
  int res = 0;

  if (v != (DISTMAT *)NULL)
    {
      v->cutoff = cutoff;
      res = 1;
    }
  else
    fprintf(stderr,"DISTMAT_cutoff received a NULL structure?\n");

  return(res);
}

/* retrieve the distance at the specific row, column point) */
double DISTMAT_get(DISTMAT *v, int row, int column)
{
  if (v != (DISTMAT *)NULL)
    {
	return(v->matrix[row][column]);
    }
  else
      {
	  fprintf(stderr,">>>DISTMAT_get received a NULL DISTMAT * ? \n");
	  return (-1.0);
      }
	
}


/* --------------- Writing the Structure Contents  -------------- */

/*
 * Function prints out the properties of a specific distmat.
 * Note that all distances > than the cutoff defined in the structure
 * are set to -1.0. Use the function DISTMAT_cutoff() to change this
 * criterion.
 *
 */
  
void DISTMAT_print(v)
DISTMAT *v;
{
  if (v != (DISTMAT *)NULL)
    {
      int row, col;
      int lores = v->lores;
      int hires = v->hires;

      double cutoff = v->cutoff;

      printf("\nDist. Matrix: <%s> LowRes: %5d, HiRes: %5d, Cutoff: %6.2lf\n",
	    v->name, lores, hires, cutoff);

      for (row = lores; row <= hires; row++)
	{
	  for (col = lores; col <= hires; col++)
	    {
	      double dist = v->matrix[row][col];

	      dist = dist <= cutoff ? dist : -1.0;
	      printf("%6.1lf ", dist);
	    }  /* for col */

	  printf("\n");

	}  /* for row */
    } /* if */

  else
    fprintf(stderr,"DISTMAT_print() received a NULL DISTMAT??\n");
}

/* tab-delimited prints for import into MAC graphing packages */
void DISTMAT_tab_print(v)
DISTMAT *v;
{
  if (v != (DISTMAT *)NULL)
    {
      int row, col;
      int lores = v->lores;
      int hires = v->hires;

      double cutoff = v->cutoff;

      printf("\nDist. Matrix: <%s> LowRes: %5d, HiRes: %5d, Cutoff: %6.2lf\n",
	    v->name, lores, hires, cutoff);

      for (row = lores; row <= hires; row++)
	{
	  for (col = lores; col <= hires; col++)
	    {
	      double dist = v->matrix[row][col];

	      dist = dist <= cutoff ? dist : -1.0;
	      printf("%6.1lf\t", dist);
	    }  /* for col */

	  printf("\n");

	}  /* for row */
    } /* if */

  else
    fprintf(stderr,"DISTMAT_print() received a NULL DISTMAT??\n");
}

/*
 * Function prints out the stats for a specific range of DISTMATs in an
 * DISTMAT list.
 */

void DISTMAT_range_print(v,nl,nh)
DISTMAT *v;
int nl, nh;
{

}


/*
 * Given an atom list build a Ca distance matrix.
 * Return (DISTMAT *)NULL on failure, a valid distance matrix otherwise.
 *
 * NOTE:
 *  Remember that residues should be contiguous!!
 *  Parameter first_res represents an residue number offset to start from.
 *  This is useful if you're comparing distance matrices between structures
 *  and modelled structures built from phi-psis, since in the latter case
 *  one uses the SECOND residue (first_res = 1). To utilize ALL residues
 *  set first_res to 0.
 *
 */

DISTMAT *DISTMAT_build_from_atom_list(atoms,name,cutoff,first_res)
     ATOM *atoms;
     char *name;
     double cutoff;
     int first_res;
{
  int lores, hires, i, j;
  double distance;

  DISTMAT *dm = (DISTMAT *)NULL;
  ATOM *ca1, *ca2;

  if (atoms != (ATOM *)NULL)
    {
      RESIDUE_range(atoms, &lores, &hires);  /* get the residue range */
      lores += first_res;

      dm = new_DISTMAT(lores, hires, name, cutoff);

      if (dm != (DISTMAT *)NULL)
	{
	  for (i = lores; i <= hires; i++)
	      {
		  ca1 = ATOM_list_scan(atoms,i,"CA");
		  for (j = lores; j <= hires; j++)
		      {
			  ca2 = ATOM_list_scan(atoms,j,"CA");
			  
			  if (ca1 == (ATOM *)NULL || ca2 == (ATOM *)NULL)
			      {
				  fprintf(stderr,
					  "\nDISTMAT_build_from_atom_list() - can't find\
an Alpha Carbon? Aborting...\n");
				  free_DISTMAT(dm);
				  return((DISTMAT *)NULL);
			      }
			  
			  distance = ATOM_dist(ca1, ca2);
			  
			  if (distance < 0.0)
			      {
				  fprintf(stderr,
					  "\nDISTMAT_build_from_atom_list() - bad distance\
. I set it to 0.\n");
				  distance = 0.0;
			      }
			  dm->matrix[i][j] = distance;
			  
		      } /* for j */
	      } /* for i */

	} /* if dm */
      else
	fprintf(stderr,
		"\nDISTMAT_build_from_atom_list() - can't build DISTMAT.\n");
    } /* if atoms */
  else
    fprintf(stderr,
	    "\nDISTMAT_build_from_atom_list() got a NULL atom list??\n");

  return((DISTMAT *)dm);
}

DISTMAT *DISTMAT_build_from_backbone_list(BACKBONE *bb,
					  char *name,
					  double cutoff,
					  int lores,
					  int hires)
{
  int i, j;
  double distance;

  DISTMAT *dm = (DISTMAT *)NULL;
  ATOM *ca1, *ca2;
  ATOM *n1, *c1, *cb1, *o1;
  ATOM *n2, *c2, *cb2, *o2;

  if (bb != (BACKBONE *)NULL)
    {
/* 
   RESIDUE_range(atoms, &lores, &hires);
   lores += first_res;
 */

      dm = new_DISTMAT(lores, hires, name, cutoff);

      if (dm != (DISTMAT *)NULL)
	{
	  for (i = lores; i <= hires; i++)
	      {
		  BACKBONE_get_backbone(bb, i, lores, 
					&n1, &ca1, &c1, &o1, &cb1);
		  for (j = lores; j <= hires; j++)
		      {
			  BACKBONE_get_backbone(bb, j, lores, 
						&n2, &ca2, &c2, &o2, &cb2);

			  if (ca1 == (ATOM *)NULL || ca2 == (ATOM *)NULL)
			      {
				  fprintf(stderr,
					  "\nDISTMAT_build_from_atom_list() - can't find\
an Alpha Carbon? Aborting...\n");
				  free_DISTMAT(dm);
				  return((DISTMAT *)NULL);
			      }

			  distance = ATOM_dist(ca1, ca2);
			  
			  if (distance < 0.0)
			      {
				  fprintf(stderr,
					  "\nDISTMAT_build_from_atom_list() - bad distance\
. I set it to 0.\n");
				  distance = 0.0;
			      }
			  dm->matrix[i][j] = distance;

		      } /* for j */
	      }  /* for i */

	} /* if dm */
      else
	fprintf(stderr,
		"\nDISTMAT_build_from_atom_list() - can't build DISTMAT.\n");
    } /* if atoms */
  else
    fprintf(stderr,
	    "\nDISTMAT_build_from_atom_list() got a NULL atom list??\n");

  return((DISTMAT *)dm);
}

/*
 * Given a phipsi list, a name for the distance matrix, and a cutoff value
 * build a distance matrix.
 * Returns this matrix if successful, (DISTMAT *)NULL otherwise.
 *
 */

DISTMAT *DISTMAT_build_from_phipsi_list(ps, name, cutoff)
     PHIPSI *ps;
     char *name;
     double cutoff;
{

  ATOM *new_atoms;
  DISTMAT *dm;

  /* we don't care about the orientation here, so pass NULL for turtle */
  new_atoms = (ATOM *)RESIDUE_build_backbone_from_phipsi(ps,NULL);

  if (new_atoms != (ATOM *)NULL)
    {
      /* build the distmat from the first residue (not second) */
      dm = DISTMAT_build_from_atom_list(new_atoms, name, cutoff,0);
    }
  else
    {
      fprintf(stderr,
	      "\nDISTMAT_build_from_phipsi_list - Could not build atom array from Phi-Psis?\n");
      return ((DISTMAT *)NULL);
    }

  if (dm == (DISTMAT *)NULL)
    fprintf(stderr,
	    "\nDISTMAT_build_from_phipsi_list - Could not build distmat?\n");

  free_ATOM_list(new_atoms);
  return((DISTMAT *)dm);
}

/*
 * Function calculates the RMS difference between two distance matrices.
 * Returns this value, or -1.0 on error (either distance matrix NULL).
 * The distmats MUST be of the same size!
 *
 */

double DISTMAT_rms(dm1, dm2)
     DISTMAT *dm1, *dm2;
{
  int i, j, _range;
  int lores1, hires1, lores2;
  int pairs = 0;
  double sum = 0.0;

  if (dm1 == (DISTMAT *)NULL || dm2 == (DISTMAT *)NULL)
      {
	fprintf(stderr,"\nDISTMAT_rms() - got NULL distmat(s)?\n");
	return(-1.0);
      }

  i = dm1->hires - dm1->lores;
  j = dm2->hires - dm2->lores;

  if (i <= 0 || j <= 0)
    {
      fprintf(stderr,"\nDISTMAT_rms() - got negative distmat size(s)?\n");
      return(-1.0);
    }

  if (i != j)
    {
      fprintf(stderr,"\nDISTMAT_rms() - incompatible distmat size(s)?\n");
      return(-1.0);
    }

  lores1 = dm1->lores;
  hires1 = dm1->hires;
  lores2 = dm2->lores;

  _range = hires1 - lores1;

  for (i = 0; i <= _range; i++)
    for (j = 0; j <= _range; j++)
      {
	double d1 = dm1->matrix[i+lores1][j+lores1];
	double d2 = dm2->matrix[i+lores2][j+lores2];

	d1 = d1 < 0.0 ? 0.0 : d1;
	d2 = d2 < 0.0 ? 0.0 : d2;

	sum += (d2 - d1) * (d2 - d1);
	pairs++;

      } /* for */
  sum = sqrt((sum / pairs));
  return(sum);
}

/*
 * Function calculates the difference between two distance matrices.
 * Returns a new DISTMAT holding the difference (dm2 - dm1), (DISTMAT *)NULL
 * otherwise. 
 *
 * NOTE: This is a simple difference, therefore one could have NEGATIVE
 *       distances in the resulting matrix! Be warned!!!
 *
 */

DISTMAT *DISTMAT_difference(dm1, dm2, name)
     DISTMAT *dm1, *dm2;
     char *name;
{
  double sum = 0.0;

  int pairs = 0;
  int i, j, _range;
  int lores1, hires1, lores2, hires2;

  DISTMAT *dm_new;

  if (dm1 == (DISTMAT *)NULL || dm2 == (DISTMAT *)NULL)
      {
	fprintf(stderr,"\nDISTMAT_difference() - got NULL distmat(s)?\n");
	return((DISTMAT *)NULL);
      }

  i = dm1->hires - dm1->lores;
  j = dm2->hires - dm2->lores;

  if (i <= 0 || j <= 0)
    {
      fprintf(stderr,"\nDISTMAT_difference() - invalid distmat size(s)?\n");
      return((DISTMAT *)NULL);
    }

  if (i != j)
    {
      fprintf(stderr,"\nDISTMAT_difference() - incompat. distmat size(s)?\n");
      return((DISTMAT *)NULL);
    }

  lores1 = dm1->lores;
  hires1 = dm1->hires;
  lores2 = dm2->lores;

  _range = hires1 - lores1;

  dm_new = new_DISTMAT(0, _range, name, MAX_CUTOFF);
  if (dm_new == (DISTMAT *)NULL)
    {
      fprintf(stderr,"\nDISTMAT_difference() - can't build new DISTMAT?\n");
      return((DISTMAT *)NULL);
    }

  for (i = 0; i <= _range; i++)
    for (j = 0; j <= _range; j++)
      {
	double d1 = dm1->matrix[i+lores1][j+lores1];
	double d2 = dm2->matrix[i+lores2][j+lores2];

	dm_new->matrix[i][j] = d2 - d1;
      } /* for */

  return(dm_new);

}

