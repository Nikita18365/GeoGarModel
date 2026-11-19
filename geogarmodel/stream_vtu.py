"""Uncompressed VTK XML appended data, generated in bounded batches."""
from .work_progress import report
import numpy as np
from .model_archive import atomic_output,regular_cells


def write_vtu(result,path):
    mesh=result.mesh
    n=int(mesh.active.sum())
    vector=result.model.ndim==2
    arrays=[('property','Float64',1,n*8)]
    if vector:
        arrays.append(('M_END','Float64',3,n*24))
    arrays.extend([('Points','Float64',3,n*192),('connectivity','Int64',1,n*64),
                   ('offsets','Int64',1,n*8),('types','UInt8',1,n)])
    offset=0
    tags={}
    for name,dtype,components,size in arrays:
        tags[name]=f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{components}" format="appended" offset="{offset}"/>'
        offset+=8+size
    header=(f'<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">'
            f'<UnstructuredGrid><Piece NumberOfPoints="{8*n}" NumberOfCells="{n}">'
            f'<CellData Scalars="property">{tags["property"]}{tags.get("M_END","")}</CellData>'
            f'<Points>{tags["Points"]}</Points><Cells>{tags["connectivity"]}{tags["offsets"]}{tags["types"]}'
            '</Cells></Piece></UnstructuredGrid><AppendedData encoding="raw">_')
    corners=np.array([[0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,0,0],[1,0,0],[1,1,0],[0,1,0]],bool)
    with atomic_output(path) as tmp,tmp.open('wb') as out:
        out.write(header.encode('ascii'))
        payload_start=out.tell()
        for name,_,_,size in arrays:
            out.write(size.to_bytes(8,'little'))
            if name in ('connectivity','offsets','types'):
                for start in range(0,n,16384):
                    end=min(start+16384,n)
                    if name=='connectivity': data=np.arange(8*start,8*end,dtype='<i8')
                    elif name=='offsets': data=np.arange(start+1,end+1,dtype='<i8')*8
                    else: data=np.full(end-start,12,dtype=np.uint8)
                    out.write(data.tobytes())
                    report(out.tell()-payload_start,offset,'Байты VTU')
            else:
                for start in range(0,mesh.size,16384):
                    ids=start+np.flatnonzero(mesh.active[start:start+16384])
                    if name=='Points':
                        b=regular_cells(mesh,ids) if mesh.bounds is None else mesh.bounds[ids]
                        data=np.where(corners,b[:,None,3:],b[:,None,:3])
                        data[:,:,2]*=-1
                    else:
                        data=result.model[ids]
                        if name=='property' and vector: data=np.linalg.norm(data,axis=1)
                    out.write(data.astype('<f8',copy=False).tobytes())
                    report(out.tell()-payload_start,offset,'Байты VTU')
        out.write(b'</AppendedData></VTKFile>')
