/*  $Id: atom.c,v 2.3 1995/06/22 20:37:41 suchanek Exp $  */
/*  $Log: atom.c,v $
 * Revision 2.3  1995/06/22  20:37:41  suchanek
 * *** empty log message ***
 *
 * Revision 2.2  1995/06/20  03:00:34  suchanek
 * function renaming
 *
 * Revision 2.1  1995/06/20  01:12:09  suchanek
 * added a formatting change for an error message
 *
 * Revision 2.0  1995/06/07  20:19:16  suchanek
 * cleanup, changed to use pointers rather than copying structures to
 * speed things up a bit.  needs more debugging, so be careful.
 *
 * Revision 1.3  1993/12/10  13:50:58  egs
 * *** empty log message ***
 *
 * Revision 1.2  93/12/10  13:47:56  egs
 * changed strcpy to strncpy
 * 
 * Revision 1.1  93/12/10  09:23:10  egs
 * Initial revision
 *  */

static char rcsid[] = "$Id: atom.c,v 2.3 1995/06/22 20:37:41 suchanek Exp $";




#include "protutil.h"
#include "atom.h"

#undef DBG

/* ---------------  -------------- */

/* --------------- Memory Allocation and Initialization -------------- */

/*
 * Function allocates space for an atom list of NH elements.
 * It also calls ATOM_list_init to fill in default information
 * in the structure
 */

ATOM *new_ATOM_list(nh)
int nh;
{
	ATOM *v = NULL;
	int nl = 0;
	v=(ATOM *)malloc((unsigned) (nh-nl+1)*sizeof(ATOM));
	if (!v) Proteus_error("allocation failure in ATOM_list()");
	ATOM_list_init(v,nh);
	return v;
}

ATOM *new_ATOM(void)
{
	ATOM *v = NULL;

	v=(ATOM *)malloc((unsigned)sizeof(ATOM));
	if (!v) Proteus_error("allocation failure in new_ATOM()");

	return v;
}

/*
 * Function frees space allocated to an atom list.
 *
 */

void free_ATOM_list(ATOM *v)
{
  if (v != (ATOM *)NULL)
    free((char*) (v));
}

/*
 * Function initializes a specific atom with its name, number, and
 * position. It calls the necessary routines to parse the atype, radius,
 * charge, and color.
 */

void ATOM_init(atom, atnam, resnam, resnumb, atnumb, x, y, z, charge)
     ATOM *atom;
     char *atnam;
     char *resnam;
     int resnumb;
     int atnumb;
     double x, y, z, charge;
{

  strncpy(atom->atnam, atnam, 4);
  atom->atnam[4] = '\0';
  strncpy(atom->resnam, resnam, 3);
  atom->resnam[3] = '\0';
  atom->atnumb = atnumb;
  atom->resnumb = resnumb;
  VECTOR_init(&atom->position,x, y, z);
  atom->collisions = 0;
  atom->charge = charge;
  atom->excluded = 0;
  ATOM_make_atom_types(atom);
  ATOM_make_radii(atom, VDW_BUMP);  /* scale the radii for bump checking */
  ATOM_make_colors(atom);
  ATOM_make_charges(atom);
}

/*
 * Function initializes an entire list of atoms to default values.
 *
 */

void ATOM_list_init(atom, nh)
ATOM *atom;
int nh;
{
  int nl = 0;
  int i, j;
  VECTOR orig;

  VECTOR_init(&orig, 0.0, 0.0, 0.0);

  for(i = nl; i < nh; i++) {
    strcpy(atom[i].atnam, "XXXX");
    strcpy(atom[i].resnam, "XXX");
    atom[i].position = orig;
    atom[i].atnumb = i;
    atom[i].resnumb = 0;
    atom[i].collisions = 0;
    atom[i].charge = 0.0;
    atom[i].natoms = nh;
    atom[i].excluded = 0;
    ATOM_make_colors(&atom[i]);
    ATOM_make_radii(&atom[i],VDW_BUMP); /* scale the radii for bump checking */
    ATOM_make_atom_types(&atom[i]);
  }
}

/*
 * Function returns the current atom list length. The list must have been
 * initialized first!
 */

int ATOM_list_length(atoms)
ATOM *atoms;
{
  return(atoms->natoms);
}


/* --------------- Writing the Structure Contents  -------------- */

/*
 * Function prints out the properties of a specific atom.
 *
 */
  
void ATOM_pprint(v)
ATOM *v;
{

    printf("\tAtom: <%s> \tNumber: %d\n\tResidue: %s\tNumber: %d\n",
	   v->atnam,v->atnumb,v->resnam, v->resnumb);
    printf("\tPosition: ");
    VECTOR_print(v->position);
    printf("\tType:\t          %2d\n\tRadius: \t%8.2lf\n\tCharge: \t%8.2lf\n",
	   v->atype,v->radius,v->charge);
    printf("\tCollsions: \t%4d\n\tColor:\t          %2d\n\tExcluded:\t  %2d\n",
	   v->collisions, v->color, v->excluded);
    printf("\n");

}

void ATOM_print(v)
ATOM *v;
{
    printf("%4s %5d %5s %5d", v->atnam,v->atnumb,v->resnam, v->resnumb);
    printf("%8.2lf %8.2lf%8.2lf",v->position.x,v->position.y,v->position.z);
    printf(" %2d %8.2lf %8.2lf ", v->atype,v->radius,v->charge);
    printf("%4d %2d %2d", v->collisions, v->color, v->excluded);
    printf("\n");
}

/*
 * Function prints out the stats for an entire list of atoms.
 *
 */

void ATOM_list_print(v)
ATOM *v;
{
  int i, j;
  int nl = 0;
  int nh = 0;

  nh = ATOM_list_length(v);

  for (i = 0; i < nh; i++) {
    ATOM_print(&v[i]);
  }

}
void ATOM_list_pprint(v)
ATOM *v;
{
  int i, j;
  int nl = 0;
  int nh = 0;

  nh = ATOM_list_length(v);

  for (i = 0; i < nh; i++) {
    ATOM_pprint(&v[i]);
  }

}

/*
 * Function prints out the stats for a specific range of atoms in an
 * atom list.
 */

void ATOM_range_print(v,nl,nh)
ATOM *v;
int nl, nh;
{
  int i, j;

  int atnum_max = ATOM_list_length(v);

  if (nl < 0 || nl >= nh) {
    fprintf(stderr,"ATOM_range_print - error, NL <%d> out of range...\n",nl);
    nl = 0;
  }

  if (nh > atnum_max) {
    fprintf(stderr,"ATOM_range_print - error, NH <%d> out of range...\n",nh);
    nh = atnum_max;
  }

  for (i = nl; i < nh; i++) {
    ATOM_print(&v[i]);
  }

}

/* --------------- Parsing the String param slots  -------------- */

/*
 * Function parses atomtype from string and fills the appropriate
 * slot.
 */

void ATOM_make_atom_types(atoms)
ATOM *atoms;
{
    /*
     * Generate into atomtype the integer 0..7 atom type.
     * 
     *
     */

    register int i;
    char      a;
    int       anum;

    a = atoms->atnam[0];
    if (islower(a))
      a = toupper(a);
    switch (a) 
      {
      case 'H':
	atoms->atype = H_TYPE;
	break;
      case 'C':
	atoms->atype = C_TYPE;
	break;
      case 'O':
	atoms->atype = O_TYPE;
	break;
      case 'N':
	atoms->atype = N_TYPE;
	break;
      case 'S':
	atoms->atype = S_TYPE;
	break;
      case 'P':
	atoms->atype = P_TYPE;
	break;
      default:
	atoms->atype = NO_TYPE;
	break;
      }
}

/*
 * Function parses atom radius and fills the appropriate
 * slot.
 */

void ATOM_make_radii(atoms, scale)
ATOM *atoms;
double scale;
{
  register int i;
  double rad;
  double radius;

  switch(atoms->atype) 
    {
    case N_TYPE:
      radius = N_RAD;
      break;
    case O_TYPE:
      radius = O_RAD;
      break;
    case C_TYPE:
      radius = C_RAD;
      break;
    case H_TYPE:
      radius = H_RAD;
      break;
    case P_TYPE:
      radius = P_RAD;
      break;
    default:
      radius = C_RAD;
      break;
    }
  rad = (scale * radius);
  atoms->radius = rad;
}


/*
 * Function parses atom color and fills the appropriate
 * slot.
 */

void ATOM_make_colors(atoms)
ATOM *atoms;
{
  register int i;
  int color;

  switch(atoms->atype) 
    {
    case N_TYPE:
      color = N_COLOR;
      break;
    case O_TYPE:
      color = O_COLOR;
      break;
    case C_TYPE:
      color = C_COLOR;
      break;
    case H_TYPE:
      color = H_COLOR;
      break;
    case P_TYPE:
      color = P_COLOR;
      break;
    default:
      color = C_COLOR;
      break;
    }
  atoms->color = color;
}

/* 
 * Function returns a pointer to an atom within the input atom_list
 * specified by the residue number (resnum), atom name (atnam),
 * and amino acid type (amino).
 *
 */

ATOM *ATOM_list_scan(atom_list, resnum, atnam)
     ATOM *atom_list;
     int resnum;
     char *atnam;
{

  int len;

  char name[5];
  char tmpname[5];

  int i;

  len = ATOM_list_length(atom_list);
  strcpy(name,atnam);

  for (i = 0; i < len; i++ ) {
      strcpy(tmpname,atom_list[i].atnam);
      if ((strcmp(name,tmpname) == 0 && resnum == atom_list[i].resnumb)) 
      return(&atom_list[i]);
  }
  return((ATOM *)NULL);

}
/* --------------- Physical and Geometric Relationships  -------------- */

/*
 * Function computes approx. atomic charge and fills the appropriate
 * slot.
 */


void ATOM_make_charges(atom)
ATOM *atom;
{

}

/*
 * Function returns TRUE if two atoms bump, FALSE otherwise. Assumes
 * radii are already set up!
 *
 */

int ATOM_pair_bumpsP(atom1, atom2)
     ATOM *atom1, *atom2;
{
  double distance = 0.0;

/*
  VECTOR pos1, pos2;
  pos1 = atom1->position;
  pos2 = atom2->position;
  distance = dist(&pos1, &pos2);
*/

  distance = dist(&atom1->position, &atom2->position);

  if (distance < (atom1->radius + atom2->radius))
    return(TRUE);
  else
    return(FALSE);
}
      
/*
 * Function computes collisions between the single atom input, and the
 * atom list. It updates the number of collistions on the input atom.
 * NOTE: it ADDS the number of collisions found to the current number
 * found in the slot. It does NOT reset it to 0 first.
 */

void ATOM_find_collisions(an_atom, atom_list)
ATOM *an_atom;
ATOM *atom_list;
{
  int len;
  int cols = an_atom->collisions;
  int abump = 0;

  register int i = 0;
  register int nbumps = 0;

  len = ATOM_list_length(atom_list);

  for (i = 0; i < len; i++) {
    if (!atom_list[i].excluded) {
      abump = ATOM_pair_bumpsP(an_atom, atom_list[i]);
      nbumps += abump;
    }
#if DBG
    fprintf(stderr,"\nChecking atom %d - %d",i,abump);
#endif
  }

  an_atom->collisions = nbumps + cols;
}

/*
 * Simply return the number of collisions
 */

int ATOM_collisions(atom)
     ATOM *atom;
{
  return(atom->collisions);
}

void ATOM_collisions_reset(atom,colls)
     ATOM *atom;
     int colls;
{
  atom->collisions = colls;
}

/*
 * Function takes an atom, and and atom_list, and returns a new
 * atom list with all atoms in atom_list which are within the radius
 * specified by rad.
 */

ATOM *ATOM_sphere(atom,atom_list,rad)
     ATOM *atom;
     ATOM *atom_list;
     double rad;
{
  ATOM *atom_sphere;    /* the new list */
  int len = 0;
  register int i = 0;
  int newlen = 0;
  int atmcnt = 0;
  int j = 0;

  VECTOR *src;
  VECTOR *dest;

  len = ATOM_list_length(atom_list);
  src = &atom->position;

/*  VECTOR_init(&dest, 0.0, 0.0, 0.0); */

  /* first find out how many atoms are within the cutoff to build the list */

  for (i = 0; i < len; i++) {
    dest = &(atom_list[i].position);
    if (dist(src, dest) <= rad)
      atmcnt++;
  }

  atom_sphere = new_ATOM_list(atmcnt);

  j = 0;
  for (i = 0; i < len; i++) {
    dest = &(atom_list[i].position);
    if (dist(src, dest) <= rad) {
      atom_sphere[j] = atom_list[i];

      /* the previous assign is destructive over ALL fields (including the
	 list length. We correct it here */

      atom_sphere[j].natoms = atmcnt; 
      j++;
    }
  }

  return(atom_sphere);
}

void free_ATOM_sphere(ATOM *sphere)
{
    free_ATOM_list(sphere);
}


/* function takes a list of atoms and a residue number, and sets the
   exclusion flag to TRUE for all backbone atoms of the indicated residue.
*/

int ATOM_exclude_backbone(atoms, resnumb)
     ATOM *atoms;
     int resnumb;
{
  ATOM *tmpatm;
  int res = 0;

  tmpatm = (ATOM *)ATOM_list_scan(atoms, resnumb, "N");
  if (tmpatm) {
    tmpatm->excluded = TRUE;
    res = 1;
  }
  else
    res = 0;
    
  tmpatm = (ATOM *)ATOM_list_scan(atoms, resnumb, "CA");
  if (tmpatm) {
    tmpatm->excluded = TRUE;
    res = 1;
  }
  else
    res = 0;
    
  tmpatm = (ATOM *)ATOM_list_scan(atoms, resnumb, "C");
  if (tmpatm) {
    tmpatm->excluded = TRUE;
    res = 1;
  }
  else
    res = 0;

  tmpatm = (ATOM *)ATOM_list_scan(atoms, resnumb, "O");
  if (tmpatm) {
    tmpatm->excluded = TRUE;
    res = 1;
  }
  else
    res = 0;

  return(res);
}

int ATOM_include_backbone(atoms, resnumb)
     ATOM *atoms;
     int resnumb;
{
  ATOM *tmpatm;
  int res = FALSE;
  
  tmpatm = (ATOM *)ATOM_list_scan(atoms, resnumb, "N");
  if (tmpatm) {
    tmpatm->excluded = FALSE;
    res = TRUE;
  }
  else
    res = FALSE;

  tmpatm = (ATOM *)ATOM_list_scan(atoms, resnumb, "CA");
  if (tmpatm) {
    tmpatm->excluded = FALSE;
    res = TRUE;
  }
  else
    res = FALSE;

  tmpatm = (ATOM *)ATOM_list_scan(atoms, resnumb, "C");
  if (tmpatm) {
    tmpatm->excluded = FALSE;
    res = TRUE;
  }
  else
    res = FALSE;

  tmpatm = (ATOM *)ATOM_list_scan(atoms, resnumb, "O");
  if (tmpatm) {
    tmpatm->excluded = FALSE;
    res = TRUE;
  }
  else
    res = FALSE;

  return(res);
}

/* Predicate returns TRUE if the atom input is a backbone atom, FALSE
   otherwise 
*/

int ATOM_is_backboneP(atom)
     ATOM *atom;
{
  static char *N = "N";
  static char *CA = "CA";
  static char *C = "C";
  static char *O = "O";

  char tmpname[5];

  strncpy(tmpname,atom->atnam,4);
  tmpname[4] = '\0';

/*
  if ((strcmp(N,tmpname) == 0) || (strcmp(CA,tmpname) == 0)  ||
      (strcmp(C,tmpname) == 0) || (strcmp(O,tmpname) == 0))
*/

  if ((strncmp(N,atom->atnam,4) == 0) || 
      (strncmp(CA,atom->atnam,4) == 0)  ||
      (strncmp(C,atom->atnam,4) == 0) || 
      (strncmp(O,atom->atnam,4) == 0))

    return(TRUE);
  else
    return(FALSE);
}

int ATOM_exclude_sidechain(atoms, resnumb)
     ATOM *atoms;
     int resnumb;
{
  int len = ATOM_list_length(atoms);
  register int i = 0;

  for (i = 0; i < len; i++) {
    if (atoms[i].resnumb == resnumb && (ATOM_is_backboneP(&atoms[i]) == 0))
      atoms[i].excluded = TRUE;
  }
  return(TRUE);
}

int ATOM_include_sidechain(atoms, resnumb)
     ATOM *atoms;
     int resnumb;
{
  int len = ATOM_list_length(atoms);
  register int i = 0;

  for (i = 0; i < len; i++) {
    if (atoms[i].resnumb == resnumb && (ATOM_is_backboneP(&atoms[i]) == 0))
      atoms[i].excluded = FALSE;
  }
  return(TRUE);
}


/* function resets the exclusion flag to FALSE for all atoms in the residue */

void ATOM_include_all(atoms, resnumb)
     ATOM *atoms;
     int resnumb;
{
  int len = ATOM_list_length(atoms);
  register int i = 0;

  for (i = 0; i < len; i++) 
    if (atoms[i].resnumb == resnumb) atoms[i].excluded = FALSE;
}

/* function resets the exclusion flag to TRUE for all atoms in the residue */

void ATOM_exclude_all(atoms, resnumb)
     ATOM *atoms;
     int resnumb;
{
  int len = ATOM_list_length(atoms);
  register int i = 0;

  for (i = 0; i < len; i++) 
    if (atoms[i].resnumb == resnumb) atoms[i].excluded = TRUE;
}

/* --------------- File I/O to & from ATOM structures -------------- */

/* function reads a PDB file and builds the ATOM structures, putting
 * the results into the atom_list input array. The number of atoms read
 * is returned. Zero is returned if there is a problem...
 */

int count_pdb_file_atoms(fname)
     char *fname;
{
}

/*
 * Read a Brookhaven file putting the coords into apos and the char
 *  string names into atype. Return number of atoms.  M Pique
 * Stop reading on End-of-file, or 'END' line, or line with just a ".".
 * Here is some sample input: 
0123456789012345678901234567890
ATOM     19  HN  THR     2      15.386  10.901   4.600
ATOM     20  HOG THR     2      14.161   9.481   4.420
ATOM     21  N   CYS     3      13.507  11.238   8.398
ATOM     22  CA  CYS     3      13.659  10.715   9.763
ATOM     23  C   CYS     3      12.265  10.443  10.307
HETATM   99  FE  XXX     4      33.265  10.443  10.307
ATOM         OT  OME    52     -16.009  30.871 -13.037                -0.543
HETATM       CU+2CU    152     -20.577   2.601 -14.587                 2.000
HETATM       ZN+2ZN    152     -17.911   1.372 -19.974                 2.000
END
                                                         These charge values
							 are non-standard
							 and are not genuine
							 PDB. They are optional
*/

int ATOM_read_pdb(all_atoms,fname)
     ATOM **all_atoms;
     char *fname;
{

  ATOM *atoms;
  FILE *ifp;
  int atmcnt = 0;

  char a_line[MAX_LINE];
  char atype[4];
  char amino[4];

  int tmp;

  ATOM tmp_atom;

  double x, y, z, charge, tcharge;
  register int n; /* number of atoms */
  int resnumb;

  if ((ifp = fopen(fname,"r")) == NULL) {
    fprintf(stderr,"ATOM_read_pdb - error. Can't open <%s> for reading \n",
	    fname);
    return(-1);
  }

   /* now count fields */

  do 
    {
      if (fgets(&a_line[0],MAX_LINE,ifp) != (char *) NULL)
	if ((tmp = strncmp("ATOM",&a_line[0],4)) == 0) atmcnt++;
    } while (!feof(ifp));

  rewind(ifp);

  if(atmcnt <= 0) {
    fprintf(stderr,"ATOM_read_pdb - error - can't get atoms from <%s>\n",
	    fname);
    return(-1);
  }

  /* set up defaults, and stores the length of the list */
  atoms = new_ATOM_list(atmcnt); 


  n=0;
  while(fgets(a_line, sizeof a_line, ifp) != NULL && n < atmcnt &&
	0!=strcmp(".", a_line) &&
	0!=strncmp("END", a_line,3) &&
	0!=strncmp("end", a_line,3))
    {
      if( (0==strncmp("ATOM",a_line,4)||
	   0==strncmp("atom",a_line,4) ||
	   0==strncmp("HETATM",a_line,6) ||
	   0==strncmp("hetatm",a_line,6) )
	 && 1==sscanf(&a_line[12]," %3s", atype)
	 && 1==sscanf(&a_line[17],"%3s", amino)
	 && 1==sscanf(&a_line[23],"%d", &resnumb)
	 && 3==sscanf(&a_line[30],"%lf %lf %lf", &x, &y, &z))

	{

#ifdef DBG
	  printf("atom %d residue %d, type %s %s at %lf %lf %lf\n",
		 n, resnumb, atype, amino, x, y, z);
#endif
	  if(1==sscanf(&a_line[70],"%lf", &tcharge))
	    charge = tcharge;
	  else charge = 0.00;
	  
	  ATOM_init(&atoms[n], atype, amino, resnumb, n, x, y, z, charge);
	  n++;
	} /* if    */
    }     /* while */

  fclose(ifp);
  *all_atoms = atoms;
  return(atmcnt);
}


void ATOM_write_pdb()
{

}


/*
  Return the distance between two atoms. Return -1.0 if either one
  is undefined, the distance otherwise.
*/

double ATOM_dist(atom1, atom2)
     ATOM *atom1;
     ATOM *atom2;
{
  double res = -1.0;

  VECTOR *v1, *v2;

  if (atom1 == (ATOM *)NULL || atom2 == (ATOM *)NULL)
    {
      fprintf(stderr,
	      "\nATOM_DIST() - got NULL atom(s)? No distance returned!\n");
      return(res);
    }

  v1 = &atom1->position;
  v2 = &atom2->position;

  res = dist(v1, v2);

  return(res);
}
