"""Full-resolution opaque voxel shell. Interior voxels stay on the original grid."""
import numpy as np



def exterior(result,component,percent):
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
    strides=(shape[1]*shape[2],shape[2],1)
    axes=(mesh.east,mesh.north,-mesh.depth)
    nodes=tuple(len(a) for a in axes)
    indices=[]; properties=[]; vertices=[]
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
                    indices.append(np.ravel_multi_index(corners,nodes))
                else:
                    bounds=mesh.bounds[cells]
                    points=np.where(offsets,bounds[:,None,3:],bounds[:,None,:3])
                    points[:,:,2]*=-1
                    vertices.append(points.reshape(-1,3))
                properties.append(values[cells])
    if vertices:
        points=np.concatenate(vertices)
        faces=np.arange(len(points)).reshape(-1,4)
    else:
        node_ids,inverse=np.unique(np.concatenate(indices).ravel(),return_inverse=True)
        points=np.column_stack([a[i] for a,i in zip(axes,np.unravel_index(node_ids,nodes))])
        faces=inverse.reshape(-1,4)
    grid=pv.PolyData.from_regular_faces(points,faces)
    grid.cell_data['property']=np.concatenate(properties)
    return grid
