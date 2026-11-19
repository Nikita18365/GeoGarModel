"""Full-resolution opaque voxel shell. Interior voxels stay on the original grid."""
import numpy as np



def terrain_nodes(active,shape,depth):
    """Share neighbouring column tops without collapsing a thin body or a cave."""
    nx,ny,nz=shape
    columns=active.reshape(shape)
    top=columns.argmax(axis=2)
    occupied=columns.any(axis=2)
    if not occupied.any() or top[occupied].min()==top[occupied].max():
        return None
    bottom=np.full((nx,ny),nz)
    levels=np.arange(nz)
    for x in range(nx):
        void=(~columns[x]) & (levels>=top[x,:,None])
        found=void.any(axis=1)
        bottom[x,found]=void[found].argmax(axis=1)
    size=(nx+1,ny+1)
    first=np.full(size,nz)
    last=np.full(size,-1)
    floor=np.full(size,nz)
    total=np.zeros(size)
    count=np.zeros(size,np.uint8)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        node=np.s_[dx:dx+nx,dy:dy+ny]
        np.minimum(first[node],top,out=first[node],where=occupied)
        np.maximum(last[node],top,out=last[node],where=occupied)
        np.minimum(floor[node],bottom,out=floor[node],where=occupied)
        total[node]+=np.where(occupied,depth[top],0.)
        count[node]+=occupied
    floor[last>=floor]=0
    last[(last>=floor)|(first==last)]=-1
    return first,last,np.divide(total,count,out=total,where=count>0),floor,depth[top[occupied].max()]


def upper_prism_limits(active, shape, percentage):
    """Exclusive depth index of each column's last detailed active cell."""
    if not np.isfinite(percentage) or not 0 <= percentage <= 100:
        raise ValueError('Доля верхних призм должна быть от 0 до 100 %.')
    columns = active.reshape(-1, shape[2])
    limits = np.zeros(len(columns), dtype=np.intp)
    if percentage == 0:
        return limits
    if percentage == 100:
        limits.fill(shape[2])
        return limits
    # Bound temporary rank arrays even for very large grids.
    batch = max(1, 131072 // shape[2])
    for start in range(0, len(columns), batch):
        block = columns[start:start + batch]
        count = block.sum(axis=1)
        wanted = np.ceil(count * percentage / 100).astype(np.intp)
        ranks = np.cumsum(block, axis=1, dtype=np.intp)
        last = np.argmax(ranks >= wanted[:, None], axis=1)
        limits[start:start + len(block)] = np.where(count, last + 1, 0)
    return limits


def prism_parts(grid,band_limits,nz,smooth,section,partial,edges):
    """Share the interpolated scalar values along the detail boundary."""
    import pyvista as pv
    columns,depths=np.divmod(grid.cell_data['voxel_id'],nz)
    detailed=depths<band_limits[columns]
    if smooth and not section:
        # Separate perpendicular faces before averaging: the top colours
        # come from cells on the roof, independently of the vertical walls.
        if partial:
            grid=grid.clean(tolerance=0)
        grid=grid.compute_normals(split_vertices=True,feature_angle=45,cell_normals=False,
                                  consistent_normals=False)
    if smooth:
        nodal=grid.cell_data_to_point_data().point_data['property']
    parts=[]
    for upper,mask in ((False,~detailed),(True,detailed)):
        if not mask.any():
            continue
        faces=grid.regular_faces[mask]
        cell_values=grid.cell_data['property'][mask]

        if upper and smooth:
            # Linear triangles retain the measured cell value at the centre
            # and share the smooth boundary colours of the lower shell.
            nodes,local=np.unique(faces,return_inverse=True)
            local=local.reshape(faces.shape)
            coordinates=grid.points[nodes]
            centers=coordinates[local].mean(axis=1)
            center_ids=np.broadcast_to((len(nodes)+np.arange(len(faces)))[:,None],local.shape)
            triangles=np.stack((local,np.roll(local,-1,axis=1),center_ids),axis=-1).reshape(-1,3)
            part=pv.PolyData.from_regular_faces(np.vstack((coordinates,centers)),triangles)
            part.point_data['property']=np.concatenate((nodal[nodes],cell_values))
        else:
            part=pv.PolyData.from_regular_faces(grid.points,faces)
            if smooth:
                part.point_data['property']=nodal
            else:
                part.cell_data['property']=cell_values
        wire=None
        if upper and smooth and edges:
            wire=pv.PolyData.from_regular_faces(grid.points,faces).extract_all_edges()
        parts.append((upper,part,wire))
    return parts


def section_faces(result, component, axis, coordinate, limits, cutoff=0.):
    """Native cell footprints in a section, with Down reflected into display Z."""
    import pyvista as pv
    mesh = result.mesh
    axes = (mesh.east, mesh.north, mesh.depth)
    if not limits[axis][0] <= coordinate <= limits[axis][1]:
        return pv.PolyData()
    at_end = coordinate == limits[axis][1]
    layer = np.clip(np.searchsorted(axes[axis], coordinate, side='left' if at_end else 'right')-1, 0, mesh.shape[axis]-1)
    section = [slice(None)]*3
    section[axis] = layer
    free = [i for i in range(3) if i != axis]
    indices = [None]*3
    indices[free[0]], indices[free[1]] = np.nonzero(mesh.active.reshape(mesh.shape)[tuple(section)])
    indices[axis] = np.full_like(indices[free[0]], layer)
    ids = np.ravel_multi_index(tuple(indices), mesh.shape)
    if mesh.bounds is None:
        low = np.column_stack([axes[i][indices[i]] for i in free])
        high = np.column_stack([axes[i][indices[i]+1] for i in free])
    else:
        bounds = mesh.bounds[ids]
        hits = (bounds[:,axis] <= coordinate) & ((coordinate < bounds[:,axis+3]) |
                (at_end & (coordinate == bounds[:,axis+3])))
        ids, bounds = ids[hits], bounds[hits]
        low, high = bounds[:,free], bounds[:,[i+3 for i in free]]
    low = np.maximum(low, [limits[i][0] for i in free])
    high = np.minimum(high, [limits[i][1] for i in free])
    values = result.model[ids]
    if values.ndim == 2:
        values = np.linalg.norm(values,axis=1) if component=='amplitude' else values[:,{'East':0,'North':1,'Down':2}[component]]
    keep = (high > low).all(axis=1) & (np.abs(values) >= cutoff)
    ids, low, high = ids[keep], low[keep], high[keep]
    if not ids.size:
        return pv.PolyData()
    if mesh.bounds is None:
        # Adjacent rectangles share vertices directly. Moving a section then
        # requires no filter to merge millions of duplicated corner points.
        u,v=free
        ui,vi=indices[u][keep],indices[v][keep]
        u0,u1=int(ui.min()),int(ui.max())+1
        v0,v1=int(vi.min()),int(vi.max())+1
        us=np.clip(axes[u][u0:u1+1],*limits[u])
        vs=np.clip(axes[v][v0:v1+1],*limits[v])
        nodes=np.empty((us.size,vs.size,3))
        nodes[:,:,axis]=coordinate
        nodes[:,:,u]=us[:,None]
        nodes[:,:,v]=vs
        nodes[:,:,2]*=-1
        first=(ui-u0)*vs.size+vi-v0
        faces=first[:,None]+[0,vs.size,vs.size+1,1]
        grid=pv.PolyData.from_regular_faces(nodes.reshape(-1,3),faces)
        grid.cell_data['voxel_id']=ids
        grid.cell_data['property']=values[keep]
        return grid
    points = np.empty((len(ids),4,3))
    points[:,:,axis] = coordinate
    points[:,:,free[0]] = np.column_stack((low[:,0],high[:,0],high[:,0],low[:,0]))
    points[:,:,free[1]] = np.column_stack((low[:,1],low[:,1],high[:,1],high[:,1]))
    points[:,:,2] *= -1
    grid = pv.PolyData.from_regular_faces(points.reshape(-1,3),np.arange(len(ids)*4).reshape(-1,4))
    grid.cell_data['voxel_id'] = ids
    grid.cell_data['property'] = values[keep]
    return grid.clean(tolerance=0)


def exterior(result,component,percent,smooth_top=False,relief_gain=1.,*,prism_shell=False,cut_axis=2,first_layer=0,limits=None):
    import pyvista as pv
    mesh=result.mesh
    values=result.model
    if values.ndim==2:
        if component=='amplitude':
            magnitude=np.empty(mesh.size)
            for begin in range(0,mesh.size,131072):
                s=slice(begin,begin+131072)
                magnitude[s]=np.linalg.norm(values[s],axis=1)
            values=magnitude
        else:
            values=values[:,{'East':0,'North':1,'Down':2}[component]]
    active=mesh.active
    if percent:
        lo=np.min(values,where=active,initial=np.inf)
        hi=np.max(values,where=active,initial=-np.inf)
        limit=max(abs(lo),abs(hi))*percent/100
        active=active.copy()
        for begin in range(0,mesh.size,131072):
            s=slice(begin,begin+131072)
            active[s]&=np.abs(values[s])>=limit
    shape=mesh.shape
    if prism_shell and first_layer:
        if not percent:
            active=active.copy()
        removed=[slice(None)]*3
        removed[cut_axis]=slice(0,first_layer)
        active.reshape(shape)[tuple(removed)]=False
    if limits is not None:
        if active is mesh.active:
            active=active.copy()
        mask=active.reshape(shape)
        for i,edges in enumerate((mesh.east,mesh.north,mesh.depth)):
            excluded=(edges[1:]<=limits[i][0]) | (edges[:-1]>=limits[i][1])
            key=[slice(None)]*3
            key[i]=excluded
            mask[tuple(key)]=False
        if mesh.bounds is not None:
            for start in range(0,mesh.size,131072):
                s=slice(start,start+131072)
                b=mesh.bounds[s]
                active[s]&=((b[:,3:]>limits[:,0]) & (b[:,:3]<limits[:,1])).all(axis=1)
    strides=(shape[1]*shape[2],shape[2],1)
    axes=(mesh.east,mesh.north,-mesh.depth)
    nodes=tuple(len(a) for a in axes)
    terrain=terrain_nodes(active,shape,mesh.depth) if smooth_top and not prism_shell and mesh.bounds is None else None
    source_cells=[]
    indices=[]; properties=[]; vertices=[]
    triangles=[]; triangle_values=[]
    for start in range(0,mesh.size,131072):
        ids=start+np.flatnonzero(active[start:start+131072])
        ijk=np.unravel_index(ids,shape)
        for axis in range(3):
            for side in (0,1):
                neighbour=ids+strides[axis]*(2*side-1)
                inside=(ijk[axis]<shape[axis]-1) if side else (ijk[axis]>0)
                exposed=~inside
                exposed[inside]=~active[neighbour[inside]]
                if mesh.bounds is not None:
                    paired=~exposed
                    a=mesh.bounds[ids[paired]]
                    b=mesh.bounds[neighbour[paired]]
                    # A truncated neighbour can expose part of this face. Keep its
                    # true rectangle; the opaque neighbour occludes the covered part.
                    covered=a[:,axis+3*side]==b[:,axis+3*(1-side)]
                    for j in ((axis+1)%3,(axis+2)%3):
                        covered&=(b[:,j]<=a[:,j])&(b[:,j+3]>=a[:,j+3])
                    exposed[paired]=~covered
                if not exposed.any(): continue
                offsets=np.zeros((4,3),int)
                offsets[:,axis]=side
                offsets[:,(axis+1)%3]=[0,1,1,0]
                offsets[:,(axis+2)%3]=[0,0,1,1]
                if side: offsets=offsets[::-1]  # Depth is reflected to Up.
                cells=ids[exposed]
                if mesh.bounds is None:
                    corners=tuple(idx[exposed,None]+offsets[:,j] for j,idx in enumerate(ijk))
                    if terrain is not None:
                        x,y,z=corners
                        first,last=terrain[:2]
                        low=first[x,y]
                        z=np.where((z>=low)&(z<=last[x,y]),low,z)
                        corners=(x,y,z)
                    face_ids=np.ravel_multi_index(corners,nodes)
                    if terrain is not None and axis!=2:
                        distinct=face_ids!=np.roll(face_ids,1,axis=1)
                        counts=distinct.sum(axis=1)
                        tri=counts==3
                        if tri.any():
                            triangles.append(face_ids[tri][distinct[tri]].reshape(-1,3))
                            triangle_values.append(values[cells[tri]])
                        cells=cells[counts==4]
                        face_ids=face_ids[counts==4]
                    indices.append(face_ids)
                else:
                    bounds=mesh.bounds[cells]
                    points=np.where(offsets,bounds[:,None,3:],bounds[:,None,:3])
                    points[:,:,2]*=-1
                    vertices.append(points.reshape(-1,3))
                properties.append(values[cells])
                if prism_shell:
                    source_cells.append(cells.astype(np.int32 if mesh.size <= np.iinfo(np.int32).max else np.int64, copy=False))
    if not properties:
        return pv.PolyData()
    if vertices:
        points=np.concatenate(vertices)
        faces=np.arange(len(points)).reshape(-1,4)
    else:
        packed=np.concatenate(indices).ravel()
        quad_size=packed.size
        indices.clear()
        if triangles:
            packed=np.concatenate((packed,np.concatenate(triangles).ravel()))
        node_ids,inverse=np.unique(packed,return_inverse=True)
        del packed
        x,y,z=np.unravel_index(node_ids,nodes)
        points=np.column_stack((axes[0][x],axes[1][y],axes[2][z]))
        if terrain is not None:
            first,last,elevation,floor,deepest=terrain
            adjusted=(z==first[x,y])&(last[x,y]>=z)
            points[adjusted,2]=-elevation[x[adjusted],y[adjusted]]
            if relief_gain!=1.:
                # Amplify relative to the deepest top; taper to zero at the first
                # cavity or base. The positive stretch preserves vertical ordering.
                for start in range(0,len(points),65536):
                    block=slice(start,start+65536)
                    u,v,k=x[block],y[block],z[block]
                    move=(k>=first[u,v]) & (k<floor[u,v])
                    u,v=u[move],v[move]
                    base=mesh.depth[floor[u,v]]
                    cap=elevation[u,v]
                    heights=points[block,2]
                    fade=np.clip((base+heights[move])/(base-cap),0.,1.)
                    heights[move]+=(relief_gain-1.)*(deepest-cap)*fade
        faces=inverse[:quad_size].reshape(-1,4)
    if limits is not None:
        np.clip(points, [limits[0,0],limits[1,0],-limits[2,1]],
                [limits[0,1],limits[1,1],-limits[2,0]], out=points)
    if triangles:
        tris=inverse[quad_size:].reshape(-1,3)
        quads=np.column_stack((np.full(len(faces),4),faces)).ravel()
        tri_cells=np.column_stack((np.full(len(tris),3),tris)).ravel()
        grid=pv.PolyData(points,np.concatenate((quads,tri_cells)))
    else:
        grid=pv.PolyData.from_regular_faces(points,faces)
    if prism_shell:
        grid.cell_data['voxel_id']=np.concatenate(source_cells)
    grid.cell_data['property']=np.concatenate(properties+triangle_values)
    return grid
