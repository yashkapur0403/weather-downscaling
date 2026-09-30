"""
Minimal NetCDF-3 (classic, CDF-1 / CDF-2) reader.

Why this exists: CHIRPS monthly NetCDF files on data.chc.ucsb.edu are large
(~100 MB) and the server supports HTTP Range requests. Classic NetCDF stores a
fully self-describing header at byte 0, so we can parse the header from a small
range request, compute the exact byte offsets of the data we need, and download
only those bytes. No GDAL/rasterio/netCDF4 required at download time.

Supports:
- CDF-1 (32-bit offsets) and CDF-2 (64-bit offsets)
- fixed-size and record (unlimited-dimension) variables
- reading a (time, lat, lon) rectangular subset over HTTP ranges or local files

Not supported: NetCDF-4/HDF5 files (magic CDF\x04/\x05 -> HDF5 signature).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

NC_TYPES = {1: ("int8", 1), 2: ("char", 1), 3: ("int16", 2),
            4: ("int32", 4), 5: ("float32", 4), 6: ("float64", 8)}

_NC_DIMENSION = 10
_NC_VARIABLE = 11
_NC_ATTRIBUTE = 12


@dataclass
class Nc3Dim:
    name: str
    size: int  # 0 means unlimited (record dimension)

    @property
    def unlimited(self) -> bool:
        return self.size == 0


@dataclass
class Nc3Var:
    name: str
    dims: list            # dimension names in order
    dtype: str            # numpy-style name
    itemsize: int
    attrs: dict = field(default_factory=dict)
    vsize: int = 0        # padded size in bytes (per record if record var)
    begin: int = 0        # absolute byte offset of data
    is_record: bool = False

    @property
    def shape(self):
        return tuple(self._sizes) if hasattr(self, "_sizes") else None


class Nc3File:
    """Header-level view of a classic NetCDF file (remote via fileobj or local path)."""

    def __init__(self, fileobj):
        self._f = fileobj
        magic = fileobj.read(4)
        if magic[:3] != b"CDF":
            raise ValueError(
                f"Not a classic NetCDF-3 file (magic={magic!r}). "
                "HDF5/NetCDF-4 files are not supported by this reader."
            )
        self.format = {1: "CDF-1", 2: "CDF-2", 5: "CDF-5"}.get(magic[3])
        if self.format is None:
            raise ValueError(f"Unsupported NetCDF-3 variant {magic!r}")
        self.off_size = 8 if self.format in ("CDF-2", "CDF-5") else 4
        self.numrecs = self._u32()
        self.dims = {}
        self.dim_order = []
        self.attrs = {}
        self.variables = {}
        self._parse_dims()
        self._parse_attrs(self.attrs)
        self._parse_vars()

    # -- primitive readers -------------------------------------------------
    def _u32(self) -> int:
        return struct.unpack(">I", self._f.read(4))[0]

    def _i32(self) -> int:
        return struct.unpack(">i", self._f.read(4))[0]

    def _off(self) -> int:
        return struct.unpack(">Q" if self.off_size == 8 else ">I",
                             self._f.read(self.off_size))[0]

    def _name(self) -> str:
        n = self._u32()
        raw = self._f.read(n)
        pad = (4 - n % 4) % 4
        self._f.read(pad)
        return raw.decode("utf-8", "replace")

    def _values(self, nc_type: int, nelems: int):
        dtype, itemsize = NC_TYPES[nc_type]
        fmt = {1: "b", 2: "c", 3: "h", 4: "i", 5: "f", 6: "d"}[nc_type]
        n = nelems * itemsize
        raw = self._f.read(n)
        pad = (4 - n % 4) % 4
        self._f.read(pad)
        vals = struct.unpack(f">{nelems}{fmt}", raw)
        if nc_type == 2:
            return raw.decode("utf-8", "replace").rstrip("\x00")
        return vals[0] if nelems == 1 else list(vals)

    # -- header sections ----------------------------------------------------
    def _parse_dims(self):
        tag = self._u32()
        if tag == 0:
            return
        if tag != _NC_DIMENSION:
            raise ValueError("Corrupt header: expected NC_DIMENSION")
        for _ in range(self._u32()):
            name = self._name()
            size = self._i32()
            self.dims[name] = Nc3Dim(name, size)
            self.dim_order.append(name)

    def _parse_attrs(self, out: dict):
        tag = self._u32()
        if tag == 0:
            return
        if tag != _NC_ATTRIBUTE:
            raise ValueError("Corrupt header: expected NC_ATTRIBUTE")
        for _ in range(self._u32()):
            name = self._name()
            nc_type = self._u32()
            nelems = self._u32()
            out[name] = self._values(nc_type, nelems)

    def _parse_vars(self):
        tag = self._u32()
        if tag == 0:
            return
        if tag != _NC_VARIABLE:
            raise ValueError("Corrupt header: expected NC_VARIABLE")
        for _ in range(self._u32()):
            name = self._name()
            ndims = self._u32()
            dimids = [self._u32() for _ in range(ndims)]
            dims = [self.dim_order[i] for i in dimids]
            attrs = {}
            self._parse_attrs(attrs)
            nc_type = self._u32()
            vsize = self._u32()
            begin = self._off()
            dtype, itemsize = NC_TYPES[nc_type]
            is_record = bool(dims) and self.dims[dims[0]].unlimited
            var = Nc3Var(name=name, dims=dims, dtype=dtype, itemsize=itemsize,
                         attrs=attrs, vsize=vsize, begin=begin, is_record=is_record)
            var._sizes = tuple(0 if self.dims[d].unlimited else self.dims[d].size
                               for d in dims)
            var._sizes = (self.numrecs,) + var._sizes[1:] if is_record else var._sizes
            self.variables[name] = var
        # record layout: record vars are interleaved, one slab per record
        self._record_vars = [v for v in self.variables.values() if v.is_record]
        self.record_size = sum((v.vsize + 3) // 4 * 4 for v in self._record_vars)


def _apply_scale(raw, attrs):
    """Apply scale_factor/add_offset if present (CHIRPS packs data as int16)."""
    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is None and offset is None:
        return raw
    out = raw.astype("float64")
    if scale is not None:
        out = out * scale
    if offset is not None:
        out = out + offset
    return out


def read_record_subset(nc: Nc3File, var_name: str, recs, rows, cols, fetch):
    """
    Read a rectangular subset of a record variable.

    nc     : parsed Nc3File
    var    : variable with dims (record_dim, dim_a, dim_b)  e.g. (time, lat, lon)
    recs   : (start, end_exclusive) along the record dimension
    rows   : (start, end_exclusive) along dim_a
    cols   : (start, end_exclusive) along dim_b
    fetch  : callable(start_byte, end_byte_exclusive) -> bytes
    """
    var = nc.variables[var_name]
    if not (var.is_record and len(var.dims) == 3):
        raise ValueError(f"{var_name} must be a record var with 3 dims, got {var.dims}")
    # position of this variable inside each record
    cum = 0
    for v in nc._record_vars:
        if v.name == var_name:
            break
        cum += (v.vsize + 3) // 4 * 4
    r0, r1 = recs
    y0, y1 = rows
    x0, x1 = cols
    ny, nx = var._sizes[1], var._sizes[2]
    row_bytes = nx * var.itemsize
    slab_row_bytes = (x1 - x0) * var.itemsize
    out_dtype = {"int8": "i1", "int16": "i2", "int32": "i4",
                 "float32": "f4", "float64": "f8"}[var.dtype]
    import numpy as np
    out = np.empty((r1 - r0, y1 - y0, x1 - x0), dtype=out_dtype)
    # contiguous rows within one record are separate requests; fetch whole
    # record span per row so a single range can cover several days? No: records
    # interleave vars, so fetch per (record, row).
    for ri in range(r0, r1):
        rec_base = var.begin + ri * nc.record_size
        row_base = rec_base + cum + y0 * row_bytes
        # rows are contiguous within the record -> one request for all rows
        data = fetch(row_base, row_base + (y1 - y0) * row_bytes)
        buf = np.frombuffer(data, dtype=f">{'i2' if var.dtype == 'int16' else out_dtype}")
        buf = buf.reshape(y1 - y0, nx)
        out[ri - r0] = buf[:, x0:x1]
    return _apply_scale(out.astype("float64") if var.dtype in ("int8", "int16", "int32")
                        else out, var.attrs)


def read_record_var_1d(nc: Nc3File, var_name: str, start: int, end: int, fetch):
    """Read a slice of a 1-D record variable (e.g. the unlimited TIME coordinate)."""
    var = nc.variables[var_name]
    if not (var.is_record and len(var.dims) == 1):
        raise ValueError(f"{var_name} is not a 1-D record variable (dims={var.dims})")
    import numpy as np
    fmt = {"int8": "b", "int16": "h", "int32": "i",
           "float32": "f", "float64": "d"}[var.dtype]
    vals = []
    for i in range(start, end):
        base = var.begin + i * nc.record_size
        vals.append(struct.unpack(f">{fmt}", fetch(base, base + var.itemsize))[0])
    return _apply_scale(np.array(vals, dtype="float64"), var.attrs)


def read_fixed_subset(nc: Nc3File, var_name: str, start, count, fetch):
    """Read a 1-D slice of a fixed (non-record) variable, e.g. latitude."""
    var = nc.variables[var_name]
    if var.is_record:
        raise ValueError(f"{var_name} is a record variable")
    base = var.begin + start * var.itemsize
    data = fetch(base, base + count * var.itemsize)
    import numpy as np
    fmt = {"int8": "b", "int16": "h", "int32": "i",
           "float32": "f", "float64": "d"}[var.dtype]
    vals = struct.unpack(f">{count}{fmt}", data)
    out = np.array(vals, dtype="float64")
    return _apply_scale(out, var.attrs)
