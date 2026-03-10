import os

import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from hrrrparser import HRRRParser


def test_parser():
    scheme = "file://"
    prefix = os.getcwd()
    url = f"{scheme}{prefix}/examples/hrrr.t22z.wrfsfcf16.grib2"
    object_store = LocalStore()
    registry = ObjectStoreRegistry({scheme: object_store})
    parser = HRRRParser()
    manifest_store = parser(url=url, registry=registry)
    ds = xr.open_dataset(
        manifest_store, engine="zarr", consolidated=False, zarr_format=3
    )
    assert ds["x"][0] != 0


def test_parser_multi_steps():
    scheme = "file://"
    prefix = os.getcwd()
    url = f"{scheme}{prefix}/examples/hrrr.t22z.wrfsfcf16.grib2"
    object_store = LocalStore()
    registry = ObjectStoreRegistry({scheme: object_store})
    parser = HRRRParser(steps=18)
    manifest_store = parser(url=url, registry=registry)
    xr.open_dataset(manifest_store, engine="zarr", consolidated=False, zarr_format=3)
