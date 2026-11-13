"""Isolated official GX reader; invoked by voxel.read_geosoft."""
import sys
from .voxel import _read_geosoft_native

if __name__=='__main__':
    try:
        model=_read_geosoft_native(sys.argv[1],sys.argv[3],sys.argv[4])
        model.save(sys.argv[2])
    except Exception as exc:
        print(str(exc),file=sys.stderr)
        sys.exit(1)
