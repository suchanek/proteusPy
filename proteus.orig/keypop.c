/*  $Id: keypop.c,v 1.1 1993/12/10 09:21:27 egs Exp $  */
/*  $Log: keypop.c,v $
 * Revision 1.1  1993/12/10  09:21:27  egs
 * Initial revision
 * */

static char rcsid[] = "$Id: keypop.c,v 1.1 1993/12/10 09:21:27 egs Exp $";


#ifndef AMIGA
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#include "keypop.h"

/* --------------- Memory Allocation and Initialization -------------- */

/*
 * Function allocates space for an atom list of NH elements.
 * It also calls KEYPOP_list_init to fill in default information
 * in the structure
 */

KEYPOP *new_KEYPOP(nh)
int nh;
{
	KEYPOP *v = NULL;
	int nl = 0;
	v=(KEYPOP *)malloc((unsigned) (nh-nl+1)*sizeof(KEYPOP));
	if (!v) fprintf(stderr,"\nERROR: allocation failure in KEYPOP_list()");
	KEYPOP_init(v,nh);
	return v;
}

/*
 * Function frees space allocated to an atom list.
 *
 */

void free_KEYPOP(v)
KEYPOP *v;
{
  if (v != (KEYPOP *)NULL)
    free((char*) (v));
}

void KEYPOP_init(v,n)
KEYPOP *v;
int n;
{
  int i = 0;

  for (i = 0; i < n; i++)
   {
    v->nelem = n;
    v->index = 0;
    v->energy = 0.0;
   }
}

void KEYPOP_sort(key_pop,POPSIZE,updown)
KEYPOP *key_pop;
int POPSIZE;
int updown;
{

#ifdef AMIGA
  int cmp_keysa();
  int cmp_keysd();

#else
  int cmp_keysa2();
  int cmp_keysd2();
#endif
#ifdef AMIGA

  if (updown > 0)
    qsort((char *)key_pop, POPSIZE, sizeof(KEYPOP), cmp_keysa);
  else
    qsort((char *)key_pop, POPSIZE, sizeof(KEYPOP), cmp_keysd);

#else
  if (updown > 0)
    qsort((void *)key_pop, POPSIZE, sizeof(KEYPOP), &cmp_keysa2);
  else
    qsort((void *)key_pop, POPSIZE, sizeof(KEYPOP), &cmp_keysd2);
#endif
}


/* sort the keys structure into descending order */

#ifdef AMIGA
int cmp_keysd(key1, key2)
KEYPOP *key1, *key2;
{
  double diff = (key2->energy - key1->energy);
  int res = 0;

  res = diff > 0.0 ? 1 : 0;
  res = diff < 0.0 ? -1 : 0;
  return((int) (res));
}

/* sort the keys structure into ascending order */

int cmp_keysa(key1, key2)
KEYPOP *key1, *key2;
{
  double diff = key1->energy - key2->energy;
  int res = 0;

  res = diff > 0.0 ? 1 : 0;
  res = diff < 0.0 ? -1 : 0;
  return((int) res);
}

#else
int cmp_keysd2(key1, key2)
const KEYPOP *key1, *key2;
{
  int res = 0;
  double res2 = 0.0;

  res2 = key2->energy - key1->energy;
  res = res2 < 0.0 ? -1 : res2 > 0.0 ? 1 : 0;

  return (res);
}

/* sort the keys structure into ascending order */

int cmp_keysa2(key1, key2)
const KEYPOP *key1, *key2;
{
  int res = 0;
  double res2 = 0.0;

  res2 = key1->energy - key2->energy;
  res = res2 < 0.0 ? -1 : res2 > 0.0 ? 1 : 0;

#ifdef NOISY
  fprintf(stderr,"Compare is: %d \n",res);
#endif

  return (res);
}
#endif

void KEYPOP_print(key)
     KEYPOP *key;
{
  if (key != (KEYPOP *)NULL)
    fprintf(stderr,"\n Index: %5d Value: %8.4lf\n", key->index, key->energy);
  else 
    fprintf(stderr,"\nERROR: KEYPOP_print() got a NULL pointer?\n");
}

void KEYPOP_print_keys(keys)
     KEYPOP *keys;
{
  int i = 0;

  if (keys != (KEYPOP *)NULL)
    {
      fprintf(stderr,"\n***************************************************");
      for (i = 0; i < keys[0].nelem; i++)
	{
	  fprintf(stderr,"\n Index: %4d Value: %8.4lf ",
		  keys[i].index,
		  keys[i].energy);
	}
      fprintf(stderr,"\n");
    }
  else fprintf(stderr,"\nERROR: KEYPOP_print_keys() got a NULL pointer?\n");
}

KEYPOP *KEYPOP_find_key(key,index)
     KEYPOP *key;
     int index;
{
  int i = 0;

  KEYPOP *res = (KEYPOP *)NULL;

  if (key != (KEYPOP *)NULL)
    {
      for (i = 0; i < key[0].nelem; i++)
	if (key[i].index == index)
	  return(&key[i]);
    }
  else
    {
      fprintf(stderr,"\nKEYPOP_find_key() - got a NULL pointer?\n");
      return(res);
    }
  return(res);
}

KEYPOP *merge_keys(keys1,keys2)
     KEYPOP *keys1, *keys2;
{
  KEYPOP *res = (KEYPOP *)NULL;
  KEYPOP *tmp = (KEYPOP *)NULL;

  int len1, len2, len;
  int i = 0;

  if ((keys1 == (KEYPOP *)NULL) || (keys2 == (KEYPOP *)NULL))
    {
      fprintf(stderr,"\nERROR: merge_keys() got NULL pointer(s)?\n");
      return((KEYPOP *)NULL);
    }

  len1 = keys1[0].nelem; len2 = keys2[0].nelem;
  if (len1 < len2)
    {
      fprintf(stderr,"\nERROR: merge_keys() incompatible sizes.");
      fprintf(stderr,"\nTry switching your command line arguments.\n");
      return(res);
    }

  if ((res = new_KEYPOP(len1)) == (KEYPOP *)NULL)
    {
      fprintf(stderr,"\nERROR: merge_keys() - can't allocate array?\n");
      return(res);
    }

  for (i = 0; i < len1; i++)
    {
      int index;
      double value, value2;

      index = keys1[i].index;
      value = keys1[i].energy;

      res[i].index = index;

      tmp = KEYPOP_find_key(keys2,index);
      if (tmp != (KEYPOP *)NULL)
	{
	  value2 = tmp->energy;
	  if (value == value2 && value == 1)     /* we have a match */
	    res[i].energy = CORRECT;
	  else
	    if (value == value2 && value == 0)   /* consistent but wrong */
	      res[i].energy = WRONG;
	  else
	    res[i].energy = INCORRECT;  /* doesn't match */
	}
      else
	res[i].energy = MISSING;    /* couldn't find index */
    }

  return(res);
}

void KEYPOP_print_subgroup(keys,group)
     KEYPOP *keys;
     double group;
{
  double value;
  int index, i;

  if (keys != (KEYPOP *)NULL)
    {
      if (group != CORRECT && group != INCORRECT && 
	  group != MISSING && group != WRONG)
	{
	  fprintf(stderr,"\nERROR: Invalid group type <%lf>. Aborting.\n",
		  group);
	  return;
	}
	
      printf("\n--------------------------------------------------------\n");
      printf("Classification SubGroup analysis for group <%s>\n",
	     group == CORRECT ? "Correct" : 
	     group == INCORRECT ? "Incorrect" :
	     group == WRONG ? "Consistently Wrong" : "MISSING");
      printf("Index \n");
      for (i = 0; i < keys[0].nelem; i++)
	if (keys[i].energy == group)
	  printf("%4d\n",keys[i].index);
    }
  else
    {
      fprintf(stderr,"\nERROR: KEYPOP_print_subgroup() got a NULL pointer?\n");
    }
}
