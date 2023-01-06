def _render(self, pvplot: pv.Plotter(), style='bs', 
            bondcolor=BOND_COLOR, bs_scale=BS_SCALE, spec=SPECULARITY, 
            specpow=SPEC_POWER) -> pv.Plotter:
    ''' 
    Update the passed pyVista plotter() object with the mesh data for the input Disulfide Bond
    Arguments:
        pvpplot: pyvista.Plotter() object
        style: 'bs', 'st', 'cpk', 'plain', 'cov': Whether to render as CPK, ball-and-stick or stick.
        Bonds are colored by atom color, unless 'plain' is specified.
    Returns:
        Updated pv.Plotter() object.
    '''
    
    radius = BOND_RADIUS
    coords = self.internal_coords()
    
    atoms = ('N', 'C', 'C', 'O', 'C', 'SG', 'N', 'C', 'C', 'O', 'C', 'SG', 'Z', 'Z', 'Z', 'Z')
    pvp = pvplot
    
    # bond connection table with atoms in the specific order shown above: 
    # returned by ss.get_internal_coords()
    
    bond_conn = numpy.array(
        [
            [0, 1], # n-ca
            [1, 2], # ca-c
            [2, 3], # c-o
            [1, 4], # ca-cb
            [4, 5], # cb-sg
            [6, 7], # n-ca
            [7, 8], # ca-c
            [8, 9], # c-o
            [7, 10], # ca-cb
            [10, 11], #cb-sg
            [5, 11],   #sg -sg
            [12, 0],  # cprev_prox-n
            [2, 13],  # c-nnext_prox
            [14,6],   # cprev_dist-n_dist
            [8,15]    # c-nnext_dist
        ])
    
    # colors for the bonds. Index into ATOM_COLORS array
    bond_split_colors = numpy.array(
        [
            ('N', 'C'),
            ('C', 'C'),
            ('C', 'O'),
            ('C', 'C'),
            ('C', 'SG'),
            ('N', 'C'),
            ('C', 'C'),
            ('C', 'O'),
            ('C', 'C'),
            ('C', 'SG'),
            ('SG', 'SG'),
            # prev and next C-N bonds - color by atom Z
            ('Z', 'Z'),
            ('Z', 'Z'),
            ('Z', 'Z'),
            ('Z', 'Z')
        ]
    )

    def draw_bonds(pvp, radius=BOND_RADIUS, plain=False, pd=False):
        # work through connectivity and colors
        for i in range(len(bond_conn)):
            bond = bond_conn[i]

            # get the indices for the origin and destination atoms
            orig = bond[0]
            dest = bond[1]

            col = bond_split_colors[i]
            if not plain:
                orig_col = ATOM_COLORS[col[0]]
                dest_col = ATOM_COLORS[col[1]]
            else:
                orig_col = dest_col = BOND_COLOR
            
            # get the coords
            prox_pos = coords[orig]
            distal_pos = coords[dest]
            
            # compute a direction vector
            direction = distal_pos - prox_pos

            # and vector length. divide by 2 since split bond
            height = math.dist(prox_pos, distal_pos) / 2.0

            origin1 = prox_pos + 0.25 * direction # the cylinder origin is actually in the middle so we translate
            origin2 = prox_pos + 0.75 * direction # the cylinder origin is actually in the middle so we translate
            
            cap1 = pv.Sphere(center=prox_pos, radius=radius)
            cap2 = pv.Sphere(center=distal_pos, radius=radius)

            cyl1 = pv.Cylinder(origin1, direction, radius=radius, height=height)
            cyl2 = pv.Cylinder(origin2, direction, radius=radius, height=height)
            
            # proximal-distal red/green coloring
            if pd == True:
                if i <= 4 or i == 11 or i == 12:
                    orig_color = dest_color = 'red'
                else:
                    orig_color = dest_color = 'green'
                if i == 10:
                    orig_color = dest_color = 'yellow'
            
            pvp.add_mesh(cyl1, color=orig_col)
            pvp.add_mesh(cyl2, color=dest_col)
            pvp.add_mesh(cap1, color=orig_col)
            pvp.add_mesh(cap2, color=dest_col)

        return pvp
        
    if style=='cpk':
        i = 0
        for atom in atoms:
            rad = ATOM_RADII_CPK[atom]
            pvp.add_mesh(pv.Sphere(center=coords[i], radius=rad), color=ATOM_COLORS[atom], smooth_shading=True, specular=spec, specular_power=specpow)
            i += 1
    elif style=='cov':
        i = 0
        for atom in atoms:
            rad = ATOM_RADII_COVALENT[atom]
            pvp.add_mesh(pv.Sphere(center=coords[i], radius=rad), color=ATOM_COLORS[atom], smooth_shading=True, specular=spec, specular_power=specpow)
            i += 1

    elif style == 'bs': # ball and stick
        i = 0
        for atom in atoms:
            rad = ATOM_RADII_CPK[atom] * bs_scale
            pvp.add_mesh(pv.Sphere(center=coords[i], radius=rad), color=ATOM_COLORS[atom], 
                            smooth_shading=True, specular=spec, specular_power=specpow)
            i += 1
        pvp = draw_bonds(pvp)

    elif style == 'sb': # splitbonds
        pvp = draw_bonds(pvp),
    
    elif style == 'pd': # proximal-distal
        pvp = draw_bonds(pvp, pd=True)

    else: # plain
        pvp = draw_bonds(pvp, plain=True)
        
    return pvp
