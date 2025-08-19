import os

import xarray as xr
from obstore.store import LocalStore
from virtualizarr.manifests.store import ObjectStoreRegistry

from hrrrparser import HRRRParser


def test_parser():
    scheme = "file://"
    prefix = os.getcwd()
    url = f"{scheme}{prefix}/examples/hrrr.t22z.wrfsfcf16.grib2"
    object_store = LocalStore()
    registry = ObjectStoreRegistry({scheme: object_store})
    parser = HRRRParser()
    parser(url=url, registry=registry)


def test_parser_multi_steps():
    scheme = "file://"
    prefix = os.getcwd()
    url = f"{scheme}{prefix}/examples/hrrr.t22z.wrfsfcf16.grib2"
    object_store = LocalStore()
    registry = ObjectStoreRegistry({scheme: object_store})
    parser = HRRRParser(steps=18)
    manifest_store = parser(url=url, registry=registry)
    xr.open_dataset(manifest_store, engine="zarr", consolidated=False, zarr_format=3)
