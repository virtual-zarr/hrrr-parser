from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
)

import numpy as np
from gribberish import parse_grib_dataset, parse_grib_message_metadata
from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.store import ObjectStoreRegistry
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.types import ChunkKey
from virtualizarr.utils import ObstoreReader
from zarr.registry import register_codec

from hrrrparser.codecs import CODEC_ID, HRRRGribberishCodec

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from virtualizarr.registry import ObjectStoreRegistry


@dataclass
class VarInfo:
    chunk_entries: Dict[str, ChunkEntry]
    dims: List[str]
    shape: List[int]
    attrs: Dict[str, Any]
    step: np.timedelta64


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


def parse_step(data):
    message = parse_grib_message_metadata(data, 0)
    forecast_date = message.forecast_date
    reference_date = message.reference_date
    step = forecast_date - reference_date
    return np.timedelta64(step, "s")


def _scan_messages(filepath: str, reader: ObstoreReader) -> dict[str, dict]:
    levels: dict[str, dict] = {}
    step = None
    for offset, size, data in _split_file(reader):
        if offset == 0:
            step = parse_step(data)
        chunk_entry: ChunkEntry = ChunkEntry.with_validation(  # type: ignore[attr-defined]
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
                    step=step,
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
    varname: str,
    data_type: DTypeLike,
    chunk_entry: ChunkEntry,
    shape: list[int],
    dims: list[str],
    steps: int = 1,
) -> ManifestArray:
    codec = HRRRGribberishCodec(var=varname, steps=steps).to_dict()
    metadata = create_v3_array_metadata(
        shape=tuple(shape),
        chunk_shape=tuple(shape),
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
        shape=tuple(shape),
        chunk_shape=(1,),
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
    varname: str,
    varinfo: VarInfo,
    coord_values: list[str],
    steps: int = 1,
) -> ManifestArray:
    if steps > 1:
        step_array = np.array(
            [np.timedelta64(i, "h") for i in range(steps)], dtype="timedelta64[s]"
        )
    else:
        step_array = np.array([varinfo.step])

    data_type = np.dtype("float64")
    codec_config = HRRRGribberishCodec(varname).to_dict()
    chunk_shape = varinfo.shape[:]
    chunk_shape.insert(1, 1)
    chunk_shape.insert(1, 1)
    shape = chunk_shape[:]
    shape[1] = steps
    shape[2] = len(coord_values)

    entries: dict[ChunkKey, ChunkEntry] = {}

    for idx, coord_value in enumerate(coord_values):
        for step_idx, step_value in enumerate(step_array):
            key = f"0.{step_idx}.{idx}.0.0"
            chunk_key = ChunkKey(key)

            if (
                coord_value in varinfo.chunk_entries.keys()
                and step_value == varinfo.step
            ):
                entries[chunk_key] = varinfo.chunk_entries[coord_value]
            else:
                entry = ChunkEntry.with_validation(  # type: ignore[attr-defined]
                    path="",
                    offset=0,
                    length=1,
                )
                entries[chunk_key] = entry
    dims = varinfo.dims
    dims.insert(1, "step")
    metadata = create_v3_array_metadata(
        shape=tuple(shape),
        data_type=data_type,
        chunk_shape=tuple(chunk_shape),
        codecs=[codec_config],
        dimension_names=varinfo.dims,
        attributes=varinfo.attrs,
    )
    chunk_manifest = ChunkManifest(entries=entries)
    array = ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
    return array


class HRRRParser:
    def __init__(self, steps: int = 1):
        self.steps = steps
        register_codec(CODEC_ID, HRRRGribberishCodec)

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given HRRR Hourly GRIB file to produce a VirtualiZarr
        [ManifestStore][virtualizarr.manifests.ManifestStore].

        Parameters
        ----------
        url
            The URL of the input HRRR GRIB file (e.g., `"s3://bucket/hrrr.t22z.wrfsfcf16.grib2"`).
        registry
            An [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A [ManifestStore][virtualizarr.manifests.ManifestStore] which provides a Zarr representation of the parsed file.
        """
        store, path_in_store = registry.resolve(url)
        reader = ObstoreReader(store=store, path=path_in_store)
        levels = _scan_messages(filepath=url, reader=reader)

        variable_arrays: dict[str, ManifestArray] = {}
        for level in levels.keys():
            for var in levels[level]["variables"].keys():
                coord_values = sorted(levels[level]["coord_values"].keys(), key=float)
                variable_array = _create_variable_array(
                    varname=var,
                    varinfo=levels[level]["variables"][var],
                    coord_values=coord_values,
                    steps=self.steps,
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
            varname="time",
            data_type=np.dtype("datetime64[s]"),
            chunk_entry=chunk_entry,
            shape=[1],
            dims=["time"],
        )
        latitude_array = _create_file_coordinate_array(
            varname="latitude",
            data_type=np.dtype("float64"),
            chunk_entry=chunk_entry,
            shape=[varinfo.shape[1], varinfo.shape[2]],
            dims=["y", "x"],
        )
        longitude_array = _create_file_coordinate_array(
            varname="longitude",
            data_type=np.dtype("float64"),
            chunk_entry=chunk_entry,
            shape=[varinfo.shape[1], varinfo.shape[2]],
            dims=["y", "x"],
        )
        step_array = _create_file_coordinate_array(
            varname="step",
            data_type=np.dtype("timedelta64[s]"),
            chunk_entry=chunk_entry,
            shape=[self.steps],
            dims=["step"],
            steps=self.steps,
        )

        arrays = (
            {"time": time_array}
            | {"step": step_array}
            | level_coordinate_arrays
            | variable_arrays
            | {"latitude": latitude_array}
            | {"longitude": longitude_array}
        )
        group = ManifestGroup(arrays=arrays)
        store = ManifestStore(registry=registry, group=group)
        return store
