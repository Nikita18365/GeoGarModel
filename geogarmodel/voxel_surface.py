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


def exterior(result,component,percent,smooth_top=False,relief_gain=1.,*,prism_shell=False,cut_axis=2,first_layer=0):
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
