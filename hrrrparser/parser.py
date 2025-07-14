from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
)

import numpy as np
from gribberish import parse_grib_dataset
from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.store import ObjectStoreRegistry, get_store_prefix
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.types import ChunkKey
from virtualizarr.utils import ObstoreReader
from zarr.registry import register_codec

from hrrrparser.codecs import CODEC_ID, HRRRGribberishCodec

if TYPE_CHECKING:
    from obstore.store import ObjectStore


@dataclass
class VarInfo:
    chunk_entries: Dict[str, ChunkEntry]
    dims: List[str]
    shape: List[int]
    attrs: Dict[str, Any]


# Vendored function from Gribberish
def _split_file(f):
    if hasattr(f, "size"):
        size = f.size
    else:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)

    while f.tell() < size:
        start = f.tell()
        head = f.read(16)
        marker = head[:4]
        if not marker:
            break  # EOF
        assert head[:4] == b"GRIB", "Bad grib message start marker"
        part_size = int.from_bytes(head[12:], "big")
        f.seek(start)
        yield start, part_size, f.read(part_size)


def _scan_messages(filepath: str, reader: ObstoreReader) -> dict[str, dict]:
    levels: dict[str, dict] = {}
    for offset, size, data in _split_file(reader):
        chunk_entry: ChunkEntry = ChunkEntry.with_validation(
            path=filepath,
            offset=offset,
            length=size,
        )
        try:
            dataset = parse_grib_dataset(data, encode_coords=False)
            var_name, var_data = next(iter(dataset["data_vars"].items()))
            level_coord = f"{var_data['attrs']['first_fixed_surface_type_coordinate']}"
            coord_value = var_data["attrs"]["fixed_surface_value"]

            var_name = f"{var_name}_{level_coord}"
            if level_coord in levels and var_name in levels[level_coord]["variables"]:
                levels[level_coord]["variables"][var_name].chunk_entries[
                    coord_value
                ] = chunk_entry
            else:
                dims = var_data["dims"][:]
                dims.insert(1, level_coord)
                var_info = VarInfo(
                    chunk_entries={coord_value: chunk_entry},
                    dims=dims,
                    shape=var_data["values"]["shape"],
                    attrs=var_data["attrs"],
                )
                levels.setdefault(level_coord, {}).setdefault("variables", {})[
                    var_name
                ] = var_info
            levels.setdefault(level_coord, {}).setdefault(
                "coord_values", {}
            ).setdefault(coord_value, chunk_entry)

        except Exception:
            continue

    return levels


def _create_file_coordinate_array(
    varname: str, chunk_entry: ChunkEntry, shape: list[int], dims: list[str]
) -> ManifestArray:
    data_type = np.dtype("float64")
    codec = HRRRGribberishCodec(var=varname).to_dict()
    metadata = create_v3_array_metadata(
        shape=shape,
        chunk_shape=shape,
        data_type=data_type,
        codecs=[codec],
        dimension_names=dims,
    )
    entries: dict[ChunkKey, ChunkEntry] = {}
    key = ".".join(["0"] * len(shape))
    chunk_key = ChunkKey(key)
    entries[chunk_key] = chunk_entry
    chunk_manifest = ChunkManifest(entries=entries)
    array = ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
    return array


def _create_level_coordinate_array(
    coord: str, coord_values: dict[str, ChunkEntry]
) -> ManifestArray:
    data_type = np.dtype("float64")
    codec = HRRRGribberishCodec(var=coord).to_dict()
    sorted_coord_values = dict(
        sorted(coord_values.items(), key=lambda item: float(item[0]))
    )
    shape = [len(sorted_coord_values)]
    metadata = create_v3_array_metadata(
        shape=shape,
        chunk_shape=[1],
        data_type=data_type,
        codecs=[codec],
        dimension_names=[coord],
    )

    entries: dict[ChunkKey, ChunkEntry] = {}
    for idx, coord_value in enumerate(sorted_coord_values):
        key = f"{idx}"
        chunk_key = ChunkKey(key)
        entries[chunk_key] = sorted_coord_values[coord_value]
    chunk_manifest = ChunkManifest(entries=entries)
    array = ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
    return array


def _create_variable_array(
    varname: str, varinfo: VarInfo, coord_values: list[str]
) -> ManifestArray:
    data_type = np.dtype("float64")
    codec_config = HRRRGribberishCodec(varname).to_dict()
    chunk_shape = varinfo.shape[:]
    chunk_shape.insert(1, 1)
    shape = chunk_shape[:]
    shape[1] = len(coord_values)

    entries: dict[ChunkKey, ChunkEntry] = {}
    for idx, coord_value in enumerate(coord_values):
        key = f"0.{idx}.0.0"
        chunk_key = ChunkKey(key)

        if coord_value in varinfo.chunk_entries.keys():
            entries[chunk_key] = varinfo.chunk_entries[coord_value]
        else:
            # print(varname, coord_value, idx)
            entry = ChunkEntry.with_validation(
                path="",
                offset=0,
                length=1,
            )
            entries[chunk_key] = entry

    metadata = create_v3_array_metadata(
        shape=shape,
        data_type=data_type,
        chunk_shape=chunk_shape,
        codecs=[codec_config],
        dimension_names=varinfo.dims,
        attributes=varinfo.attrs,
    )
    chunk_manifest = ChunkManifest(entries=entries)
    array = ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
    return array


class HRRRParser:
    def __init__(
        self,
    ):
        register_codec(CODEC_ID, HRRRGribberishCodec)

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        reader = ObstoreReader(store=object_store, path=file_url)
        levels = _scan_messages(filepath=file_url, reader=reader)

        variable_arrays: dict[str, ManifestArray] = {}
        for level in levels.keys():
            for var in levels[level]["variables"].keys():
                coord_values = sorted(levels[level]["coord_values"].keys(), key=float)
                variable_array = _create_variable_array(
                    varname=var,
                    varinfo=levels[level]["variables"][var],
                    coord_values=coord_values,
                )
                variable_arrays[var] = variable_array

        level_coordinate_arrays: dict[str, ManifestArray] = {}
        for level in levels:
            coord_values = levels[level]["coord_values"]
            level_coordinate_array = _create_level_coordinate_array(
                coord=level, coord_values=coord_values
            )
            level_coordinate_arrays[level] = level_coordinate_array

        varinfo = next(iter(levels[level]["variables"].values()))
        chunk_entry = next(iter(varinfo.chunk_entries.values()))
        time_array = _create_file_coordinate_array(
            varname="time", chunk_entry=chunk_entry, shape=[1], dims=["time"]
        )
        latitude_array = _create_file_coordinate_array(
            varname="latitude",
            chunk_entry=chunk_entry,
            shape=[varinfo.shape[1], varinfo.shape[2]],
            dims=["y", "x"],
        )
        longitude_array = _create_file_coordinate_array(
            varname="longitude",
            chunk_entry=chunk_entry,
            shape=[varinfo.shape[1], varinfo.shape[2]],
            dims=["y", "x"],
        )

        arrays = (
            {"time": time_array}
            | level_coordinate_arrays
            | variable_arrays
            | {"latitude": latitude_array}
            | {"longitude": longitude_array}
        )
        group = ManifestGroup(arrays=arrays)
        registry = ObjectStoreRegistry({get_store_prefix(file_url): object_store})
        store = ManifestStore(store_registry=registry, group=group)
        return store
