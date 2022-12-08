/*  $Id: residue.c,v 1.8 1995/06/25 03:18:10 suchanek Exp $  */
/*  $Log: residue.c,v $
 * Revision 1.8  1995/06/25  03:18:10  suchanek
 * fixed problem with BACKBONE_create - still work on RESIDUE_range
 * it's not returning the right number with the -O2 flag set
 *
 * Revision 1.7  1995/06/23  14:16:40  suchanek
 * found a memory leak with backbone_init
 *
 * Revision 1.6  1995/06/22  20:38:06  suchanek
 * added the backbone functions
 *
 * Revision 1.5  1995/06/22  18:55:06  suchanek
 * added backbone stuff - still some problems with it
 *
 * Revision 1.4  1995/06/20  03:00:01  suchanek
 * function renaming
 *
 * Revision 1.3  1995/06/12  03:56:48  suchanek
 * some cleanup
 *
 * Revision 1.2  1995/06/11  05:13:31  suchanek
 * added ansi like arg lists
 *
 * Revision 1.1  1993/12/10  09:22:19  egs
 * Initial revision
 * */

static char rcsid[] = "$Id: residue.c,v 1.8 1995/06/25 03:18:10 suchanek Exp $";




#include "protutil.h"
#include "residue.h"


/*
  bbone_to_schain assumes turtle is in orientation #2 (at alpha carbon,
  headed towards carbonyl, with nitrogen on left) to orientation #1
  (at alpha c, headed to beta carbon, with nitrogen on left 
*/

void bbone_to_schain(TURTLE *turtle)
{
  TURTLE_roll(turtle, 240.0);
  TURTLE_pitch(turtle, 180.0);
  TURTLE_yaw(turtle, 110.0);
  TURTLE_roll(turtle, 240.0);
}

/*
  schain_to_bbone moves turtle from orientation #1 to orientation #2
*/

void schain_to_bbone(TURTLE *turtle)
{
  TURTLE_pitch(turtle, 180.0);
  TURTLE_roll(turtle, 240.0);
  TURTLE_yaw(turtle, 110.0);
  TURTLE_roll(turtle, 120.0);
}


/*
  build residue assumes the turtle is in orientation #2
  (at Ca, headed to C, with N on left), and returns coordinates
  for n, ca, cb, and c.

  NOTE: Position of Oxygen depends on psi, which may not be known
*/

void build_residue(TURTLE *turtle, VECTOR *n, VECTOR *ca, 
		   VECTOR *cb, 
		   VECTOR *c)
{

  VECTOR _n, _ca, _cb, _c;

  _n.x = -0.486; _n.y = 1.366; _n.z = 0.0;
  _ca.x = 0.0; _ca.y = 0.0; _ca.z = 0.0;
  _cb.x = -0.523; _cb.y = -0.719; _cb.z = -1.245;
  _c.x = 1.53; _c.y = 0.0; _c.z = 0.0;

  *n = TURTLE_to_global(turtle,&_n);
  *ca = TURTLE_to_global(turtle,&_ca);
  *cb = TURTLE_to_global(turtle,&_cb);
  *c = TURTLE_to_global(turtle,&_c);
}

/*
 * function will orient a turtle at a residue specified by resnum,
 * resnam, within the list pointed to by atoms.
 * Zero is returned if the orientation is not possible (usually means
 * the atom was not found). Orientation specifies either 1 or 2.
 *
 *
 */

int orient_at_residue(TURTLE *turt, ATOM *atoms, int resnum, int orientation)
{

  ATOM *n_atom, *ca_atom, *c_atom, *cb_atom;
  int res = 0;

  if (orientation != 1 && orientation != 2)
    return(0);  /* invalid orientation */

  n_atom = ATOM_list_scan(atoms, resnum, "N");
  ca_atom = ATOM_list_scan(atoms, resnum, "CA");
  c_atom = ATOM_list_scan(atoms, resnum, "C");
  cb_atom = ATOM_list_scan(atoms, resnum, "CB");

  switch(orientation)
    {
    case ORIENT_SIDECHAIN:
      if (n_atom == (ATOM *)NULL || ca_atom == (ATOM *)NULL ||
	  cb_atom == (ATOM *)NULL) {
	fprintf(stderr,
		"ORIENT_AT_RESIDUE - error <SIDECHAIN> - missing atoms?\n");
	res = 0;
	break;
      }
      TURTLE_orient(turt, &ca_atom->position, &cb_atom->position, &n_atom->position);
      res = 1;
      break;
    case ORIENT_BACKBONE:
      if (n_atom == (ATOM *)NULL || ca_atom == (ATOM *)NULL ||
	  c_atom == (ATOM *)NULL) {
	fprintf(stderr,
		"ORIENT_AT_RESIDUE - error <BACKBONE> - missing atoms?\n");
	res = 0;
	break;
      }
      TURTLE_orient(turt, &ca_atom->position, &c_atom->position, &n_atom->position);
      res = 1;
      break;
    default:
      fprintf(stderr,"ORIENT_AT_RESIDUE - error - <INVALID ORIENTATION>?\n");
      break;
    } /* switch */

  return(res);
}

/************************* Residue Building Subroutines **********************/

/*
	Abstract:	TO_ALPHA moves turtle from backbone nitrogen to
			alpha carbon.  Turtle begins at nitrogen, headed 
			towards alpha carbon, with carbonyl carbon of 
			previous residue on left side and ends in 
			orientation #2 (at alpha carbon, headed towards 
			carbonyl carbon, with nitrogen on left side).


*/

VECTOR to_alpha(TURTLE *turtle, double phi)
{
  TURTLE_move(turtle, 1.45);
  TURTLE_roll(turtle, phi);
  TURTLE_yaw(turtle,110.0);
  return(turtle->position);
}


/*
   Abstract: TO_CARBONYL moves turtle from alpha carbon to carbonyl carbon.
		Turtle begins in orientation #2 (at alpha carbon,
		headed towards carbonyl carbon, with nitrogen on
		left) and ends at carbonyl carbon, headed towards 
		nitrogen of next residue, with alpha carbon of current 
		residue on left side.


*/

VECTOR to_carbonyl(TURTLE *turtle, double psi)
{

  TURTLE_move(turtle, 1.53);
  TURTLE_roll(turtle, psi);
  TURTLE_yaw(turtle, 114.0);
  return(turtle->position);
}

/*
	Abstract:	TO_NITROGEN shifts turtle from carbonyl carbon of 
			one residue to nitrogen of next residue.  
			Turtle begins at carbonyl carbon, headed towards 
			nitrogen of second residue, with alpha carbon of 
			first residue on left side.  Turtle ends at 
			nitrogen of second residue, headed towards alpha 
			carbon of second residue, with carbonyl carbon of 
			first residue on left side.

*/

VECTOR to_nitrogen(TURTLE *turtle, double omega)
{
  TURTLE_move(turtle, 1.32);
  TURTLE_roll(turtle, omega);
  TURTLE_yaw(turtle, 123.0);
  return(turtle->position);
}

/*
	Abstract:	ADD_OXYGEN calculates coordinates of carbonyl oxygen
			(in global coords. system) Subroutine assumes that 
			turtle is at carbonyl carbon, headed towards nitrogen 
			of next	residue, with previous alpha carbon on its left 
			(and therefore with carbonyl oxygen on its right).


	Environment:	Appropriate to call this immediately after calling
			subroutine TO_CARBONYL.


*/

void add_oxygen(TURTLE *turtle, ATOM *O)
{

  VECTOR loc, res;

  VECTOR_init(&loc, -0.673, -1.029, 0.0);
  res = TURTLE_to_global(turtle,&loc);
  O->position = res;
}


/* as above, except only updates the turtle */
VECTOR to_oxygen(TURTLE *turtle)
{

  VECTOR loc, res;

  VECTOR_init(&loc, -0.673, -1.029, 0.0);
  res = TURTLE_to_global(turtle,&loc);
  return(res);
}



/* --------------- Memory Allocation and Initialization -------------- */

/*
 * Function allocates space for an BACKBONE list of NH elements.
 * It also calls BACKBONE_list_init to fill in default information
 * in the structure
 */

BACKBONE *new_BACKBONE_list(int nl, int nh)
{
	BACKBONE *v = NULL;

	v=(BACKBONE *)malloc((unsigned) (nh-nl+1)*sizeof(BACKBONE));
	if (!v) Proteus_error("allocation failure in BACKBONE_list()");

	BACKBONE_list_init(v, nl, nh);

	return v;
}

/*
 * Function frees space allocated to an BACKBONE list.
 *
 */

void free_BACKBONE_list(BACKBONE *v)
{
    int i = 0;
    int lowres;
    int elem = sizeof(v)/sizeof(v[0]);

    if (v != (BACKBONE *)NULL)
	{
	    for (i = 0; i < elem; i++)
		{
		    if (v[i].free_cb)
			free_ATOM_list(v[i].cb_atom);
		}
	}

    free((char*) (v));
}

/*
 * Function initializes a specific BACKBONE with its name, number, and
 * position. It calls the necessary routines to parse the atype, radius,
 * charge, and color.
 */

void BACKBONE_init(BACKBONE *backbone, int resnumb, ATOM *n, 
		   ATOM *ca, ATOM *c, ATOM *o, ATOM *cb, int free_cb)
{

  backbone->resnumb = resnumb;
  backbone->n_atom = n;
  backbone->ca_atom = ca;
  backbone->c_atom = c;
  backbone->o_atom = o;
  backbone->cb_atom = cb;
}

/*
 * Function initializes an entire list of BACKBONEs to default values.
 * assumes residues are numbered CONSECUTIVELY!!!
 */

void BACKBONE_list_init(BACKBONE *backbone, int nl, int nh)
{
  int i = 0;

  for(i = nl; i <= nh; i++) {
    backbone[i-nl].resnumb = i;
    backbone[i-nl].nres = nh - nl + 1;
    backbone[i-nl].lowres = nl;
    backbone[i-nl].hires = nh;
  }
}

/*
 * Function returns the current BACKBONE list length. The list must have been
 * initialized first!
 */

int BACKBONE_list_length(BACKBONE *backbone)
{
  return(backbone->nres);
}


void BACKBONE_print(BACKBONE *backbone, int lowres, int hires)
{

    int i = 0;

    printf("Backbone info.  Low: %d, High: %d, NRES: %d \n", backbone[lowres].lowres, backbone[lowres].hires,
	   backbone[lowres].nres);

    for (i = lowres; i <= hires; i++)
	{
	    ATOM_print(backbone[i-lowres].n_atom);
	    ATOM_print(backbone[i-lowres].ca_atom);
	    ATOM_print(backbone[i-lowres].c_atom);
	    ATOM_print(backbone[i-lowres].o_atom);
	    ATOM_print(backbone[i-lowres].cb_atom);

	}
}

/* 
   Subroutine creates a backbone list given a passed atom list,
   constructing CB as needed. Assumes residues are numbered CONSECUTIVELY!!!
*/

BACKBONE *BACKBONE_create(ATOM *atoms)
{
    BACKBONE *bb = NULL;
    ATOM *n, *ca, *c, *o, *cb;

    int i;
    int lowres, hires;

    int do_cleanup = 0;

    RESIDUE_range(atoms, &lowres, &hires);
    if (lowres <= 0 || hires <= 0)
	{
	    fprintf(stderr,">>>>>>BACKBONE_create(), ERROR! invalid residue range %d, %d \n", lowres, hires);
	    return((BACKBONE *)NULL);
	}


    if ((bb = new_BACKBONE_list(lowres, hires)) == (BACKBONE *)NULL)
	{
	    fprintf(stderr,">>>>>>BACKBONE_create(), ERROR! can't allocate BACKBONE\n");
	    return((BACKBONE *)NULL);
	}

    for (i = lowres; i <= hires; i++)
	{
	    RESIDUE_get_backbone(atoms, i, &n, &ca, &c, &o, &cb, 1, &do_cleanup);
	    BACKBONE_init(&bb[i-lowres], i, n, ca, c, o, cb, do_cleanup);
	}

    return(bb);
}


int BACKBONE_get_backbone(BACKBONE *backbone, int resnum, int lowres, 
			  ATOM **n,
			  ATOM **ca,
			  ATOM **c,
			  ATOM **o,
			  ATOM **cb)
{
    int res = 0;
    int rnum = 0;

    if ((backbone[resnum-lowres].resnumb != resnum))
	{
	    for (rnum = lowres; rnum <= backbone[rnum-lowres].hires; rnum++)
		{
		    if (backbone[rnum-lowres].resnumb == resnum)
			{
			    *n = backbone[rnum-lowres].n_atom;
			    *ca = backbone[rnum-lowres].ca_atom;
			    *c = backbone[rnum-lowres].c_atom;
			    *o = backbone[rnum-lowres].o_atom;
			    *cb = backbone[rnum-lowres].cb_atom;
			    res = 1;
			}
		}
	}

    else
	{
	    if (backbone[resnum-lowres].resnumb == resnum)
		{
		    *n = backbone[resnum-lowres].n_atom;
		    *ca = backbone[resnum-lowres].ca_atom;
		    *c = backbone[resnum-lowres].c_atom;
		    *o = backbone[resnum-lowres].o_atom;
		    *cb = backbone[resnum-lowres].cb_atom;
		    res = 1;
		}
	}

    return(res);
}


/*
  Subroutine returns backbone atoms and CB if it exists for a given
  atom list and residue number
*/

void RESIDUE_get_backbone(ATOM *atoms, int resnum, 
			  ATOM **n,
			  ATOM **ca,
			  ATOM **c,
			  ATOM **o,
			  ATOM **cb,
			  int build_cb,
			  int *do_cleanup)
{

    VECTOR n2, ca2, c2, cb2;
    TURTLE turtle;
    ATOM *cb_tmp;
    int res = 0;

    *c = ATOM_list_scan(atoms, resnum, "C");
    *n = ATOM_list_scan(atoms, resnum, "N");
    *ca = ATOM_list_scan(atoms, resnum, "CA");
    *o = ATOM_list_scan(atoms, resnum, "O");
    *cb = ATOM_list_scan(atoms, resnum, "CB");

    if ((*cb == (ATOM *)NULL) && build_cb)
	{
	    /* orient at residue to get CB. if error, return NULL */
	    TURTLE_init(&turtle,"tmp");
#ifdef NOISY2
	    fprintf(stderr,"Created a CB in get residues\n");
#endif
	    cb_tmp = new_ATOM(); /* fatal if error */

	    if ((res = orient_at_residue(&turtle, 
					 atoms, 
					 resnum, ORIENT_BACKBONE)) == 0)
		{
		    fprintf(stderr,"\nget backbone - can't orient\n");
		    return;
		}


	    build_residue(&turtle, &n2, &ca2, &cb2, &c2);
	    ATOM_init(cb_tmp, "  CB", "XXX", resnum, 999, cb2.x, 
		      cb2.y, cb2.z, 0.0);
	    *cb = cb_tmp;
	    *do_cleanup = TRUE;
	}
    *do_cleanup = FALSE;

}


/* 
   Function returns dihedral angles phi, psi, and omega given a residue
   number and list of atoms. Avoid the use of duplicate residue numbers!!!
   0 is returned if failure, 1 otherwise
*/

int RESIDUE_phi_psi_omega(ATOM *atoms, int resnum, double *phi, double *psi, double *omega)
{
  int res = 1;
  int phi_ok, psi_ok, omega_ok;
  double ang;

  ATOM *n, *ca, *c, *n2, *c2, *ca2;
  VECTOR p1, p2, p3, p4;

  c2 = ATOM_list_scan(atoms, resnum-1, "C");
  n = ATOM_list_scan(atoms, resnum, "N");
  ca = ATOM_list_scan(atoms, resnum, "CA");
  c = ATOM_list_scan(atoms, resnum, "C");
  n2 = ATOM_list_scan(atoms, resnum+1, "N");
  ca2 = ATOM_list_scan(atoms, resnum+1, "CA");

  phi_ok = psi_ok = omega_ok = 1;

  if (n == (ATOM *)NULL || c == (ATOM *)NULL || ca == (ATOM *)NULL)
    {
      fprintf(stderr,"RESIDUE_phi_psi_omega: Can't find backbone atom(s)?\n");
      return(0);
    }

  if (c2 == (ATOM *)NULL) 
    {
      *phi = PHIPSI_UNDEFINED; /* can't compute phi */
      phi_ok = 0;
    }

  if (n2 == (ATOM *)NULL) 
    {
      *psi = PHIPSI_UNDEFINED; /* can't compute psi */
      psi_ok = 0;
    }

  if (ca2 == (ATOM *)NULL) 
    {
      *omega = PHIPSI_UNDEFINED; /* can't compute omega */
      omega_ok = 0;
    }


  if (phi_ok)
    {
      p1 = c2->position;
      p2 = n->position;
      p3 = ca->position;
      p4 = c->position;
      ang = find_dihedral(&p1, &p2, &p3, &p4);
      *phi = ang;
    }

  if (psi_ok)
    {
      p1 = n->position;
      p2 = ca->position;
      p3 = c->position;
      p4 = n2->position;
      ang = find_dihedral(&p1, &p2, &p3, &p4);
      *psi = ang;
    }

  if (omega_ok)
    {
      p1 = ca->position;
      p2 = c->position;
      p3 = n2->position;
      p4 = ca2->position;
      ang = find_dihedral(&p1, &p2, &p3, &p4);
      *omega = ang;
    }

  return(res);
}

/* Function computes the range of residue numbers for a given atom list.
   These values are returned in *lores and *hires. This function will
   work properly as long as contiguous residues have contiguous residue
   numbers!! 
*/

int RESIDUE_range(ATOM *atoms, int *lores, int *hires)
{

  int len = 0;
  int i;
  int low, hi, numb;
  int oldnumb = 0;
  int tot_residues= -1;

  low = 9999;
  hi = -9999;

  len = ATOM_list_length(atoms);
  oldnumb = atoms[0].resnumb;
  if (len)
    tot_residues = 1;

  for (i = 0; i < len; i++ ) 
    {
      numb = atoms[i].resnumb;
      if (oldnumb != numb) {
	tot_residues++;
	oldnumb = numb;
      }
      low = low < numb ? low : numb;
      hi = hi > numb ? hi : numb;
    }
  
  *lores = low;
  *hires = hi;
  return(tot_residues);
}

/* 
  Build a PHIPSI list given a list of atoms. Values which cannot be
  computed are assigned PHIPSI_UNDEFINED.

  USEAGE:
     ATOM *atom_list;
     PHIPSI *phipsi_list;

     result = RESIDUE_phi_psi_list_build(atom_list, &phipsi_list);
*/

  
int RESIDUE_phi_psi_list_build(ATOM *atoms, PHIPSI **phipsi_out)
{
  int low, hi, i, j, res;
  PHIPSI *ps;
  double phi, psi, omega;

  RESIDUE_range(atoms, &low, &hi);
  i = hi - low + 1;
  if (i <= 0) 
    {
      fprintf(stderr,
	      "\nRESIDUE_phi_psi_list_build: Invalid number of residues?\n");
      return(0);
    }

  ps = new_PHIPSI_list(i);
  if (ps == (PHIPSI *)NULL) 
    {
      fprintf(stderr,
	      "\nRESIDUE_phi_psi_list_build: Can't allocate phipsi list?\n");
      return(0);
    }

  for (i = low, j = 0; i <= hi; i++, j++) 
    {
      res = RESIDUE_phi_psi_omega(atoms, i, &phi, &psi, &omega);
      if (res == 0)
	{
	  fprintf(stderr,
		  "\nRESIDUE_phi_psi_list_build: can't compute phi, psi?\n");
	  free_PHIPSI_list(ps);
	  return(0);
	}
      PHIPSI_init(&ps[j], phi, psi, omega, i);
    }  /* for */

  *phipsi_out = ps;
  return(1);

}

/******************************************************************************
  Given a list of PHIPSI's and a defining turtle orientation, build 
  the protein backbone (N, Ca, C and O).

  Return (ATOM *)NULL if Failure, a valid ATOM list otherwise. 

  To model a specific protein in a specific orientation call:
  orient_at_residue(turt, atoms, first_res, ORIENT_BACKBONE) first.

  This will set up the turtle at Ca in the proper orientation to build the
  rest of the structure. If you don't care about the orientation, 
  pass (TURTLE *)NULL to this function. 

  NOTE: This function will start its build from the second element of the
        phipsi array, since in general the first element will have an
	undefined phi value. You should make sure you're oriented on the
	second residue to use this correctly!!

******************************************************************************/

ATOM *RESIDUE_build_backbone_from_phipsi(PHIPSI *ps_list, TURTLE *turtle)
{
  TURTLE tmpturtle;
  ATOM *tmpatoms = (ATOM *)NULL;
  VECTOR pos;

  int lores, hires, i;
  int phipsi_len = 0;
  int num_newatoms = 0;
  int turtle_passed = 0;
  int atom_cnt = 0;

  TURTLE_init(&tmpturtle,"tmp");

  if (turtle != (TURTLE *)NULL)
    {
      TURTLE_copy_coords(&tmpturtle,turtle);  /* now has coords */
      turtle_passed = TRUE;
    }

  phipsi_len = PHIPSI_list_length(ps_list);
  if (phipsi_len <= 0)
    {
      fprintf(stderr,
	      "\nRESIDUE_build_backbone_from_phipsi: No phipsi length ?\n");
      return((ATOM *)NULL);
    }

  num_newatoms = 4 * (phipsi_len - 1);
  tmpatoms = new_ATOM_list(num_newatoms);
  if (tmpatoms == (ATOM *)NULL)
    {
      fprintf(stderr,
	      "\nRESIDUE_build_backbone_from_phipsi: Can't allocate Atoms?\n");
      return((ATOM *)NULL);
    }

  /* start building at the second element of the phipsi array, since
     in general, the first element will have phi undefined!
   */

#define FIRST 1

  for (i = FIRST; i < phipsi_len; i++)
    {
      char *resnam = "XXX";
      double _phi;
      double _psi;
      double _omega;

      ATOM _o;

      ATOM_init(&_o, "O", resnam, 0, 0, 0.0, 0.0, 0.0, 0.0);

      _phi = ps_list[i].phi;
      _psi = ps_list[i].psi;
      _omega = ps_list[i].omega;

      if (turtle_passed == FALSE && i == FIRST) /* first time at N */
	{
	  pos = tmpturtle.position;
	  ATOM_init(&tmpatoms[atom_cnt], "N", resnam, ps_list[i].resnumb, 
		    atom_cnt, pos.x, pos.y, pos.z, 0.0);
		    
	  atom_cnt++;

	  pos = to_alpha(&tmpturtle, _phi);
	  ATOM_init(&tmpatoms[atom_cnt], "CA", resnam, ps_list[i].resnumb, 
		    atom_cnt, pos.x, pos.y, pos.z, 0.0);
	  atom_cnt++;
	}

      if (turtle_passed == TRUE && i == FIRST) /* first time at Ca */
	{
	  /* fake the nitrogen position */
	  VECTOR _n, _ca, _cb, _c;

	  build_residue(&tmpturtle, &_n, &_ca, &_cb, &_c);
	  ATOM_init(&tmpatoms[atom_cnt], "N", resnam, ps_list[i].resnumb, 
		    atom_cnt, _n.x, _n.y, _n.z, 0.0);
	  atom_cnt++;
	  
	  pos = tmpturtle.position;
	  ATOM_init(&tmpatoms[atom_cnt], "CA", resnam, ps_list[i].resnumb, 
		    atom_cnt, pos.x, pos.y, pos.z, 0.0);
	  atom_cnt++;
	}

      if (i != FIRST)
	{
	  pos = to_alpha(&tmpturtle, _phi);
	  ATOM_init(&tmpatoms[atom_cnt], "CA", resnam, ps_list[i].resnumb, 
		    atom_cnt, pos.x, pos.y, pos.z, 0.0);
	  atom_cnt++;

	}

      pos = to_carbonyl(&tmpturtle, _psi);
      ATOM_init(&tmpatoms[atom_cnt], "C", resnam, 
		ps_list[i].resnumb, atom_cnt, pos.x, pos.y, pos.z, 0.0);
      atom_cnt++;
	  
      add_oxygen(&tmpturtle, &_o);
      ATOM_init(&tmpatoms[atom_cnt], "O", resnam, 
		ps_list[i].resnumb, atom_cnt, _o.position.x, 
		_o.position.y, _o.position.z, 0.0);
      atom_cnt++;

      if (_psi == PHIPSI_UNDEFINED && _omega == PHIPSI_UNDEFINED) break; /* at C term. */
      pos = to_nitrogen(&tmpturtle, _omega);
      ATOM_init(&tmpatoms[atom_cnt], "N", resnam, ps_list[i+1].resnumb, 
		atom_cnt, pos.x, pos.y, pos.z, 0.0);
      atom_cnt++;

    }  /* for */

  return(tmpatoms);
}
