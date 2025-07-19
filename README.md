## HRRRParser
A VirtualiZarr parser for NOAA HRRR GRIB files.

### Status
Experimental, proof-of-concept.

### Usage
See [VirtualiZarr](https://github.com/zarr-developers/VirtualiZarr) for
documentation on how parsers work to generate virtual representations of
datasets.

- Currently the HRRRParser only supports hourly files.  Sub-hourly support should
be added soon.

- Using the generated VirtualiZarr `ManifstStore` or virtual dataset requires use
of the included `HRRRGribberishCodec` for decoding.

    This codec can be registered with `zarr-python` via
    ```
    from zarr.registry import register_codec
    from hrrrparser.codecs import CODEC_ID, HRRRGribberishCodec
    register_codec(CODEC_ID, HRRRGribberishCodec)
    ```
- The resulting virtual Zarr stores contain a "reference time" and "step"
  dimension (`time` and `step` respectively) that can be utilized for different “Forecast Model Run Collections” (FMRC) approaches.
  See https://xarray-indexes.readthedocs.io/earth/forecast.html for use with
  valid time approaches.

- See `/examples/surface_local.ipynb` for more detailed usage and file
  concatenation options.
