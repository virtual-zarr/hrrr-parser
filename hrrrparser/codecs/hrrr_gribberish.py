from dataclasses import dataclass
from typing import Self

import numpy as np
from gribberish import parse_grib_array, parse_grib_message_metadata
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration

LEVEL_COORDINATES = [
    "atm",
    "clt",
    "hag",
    "sfc",
    "lfc",
    "isotherm",
    "isobar",
    "sigma",
    "entire_atm",
    "bndry_cloud",
    "lcl",
    "mcl",
    "hcl",
    "cld_ceiling",
    "clb",
    "zero_deg_isotherm",
    "htfl",
    "adiabatic_condensation_lifted",
    "eqm",
    "depth_bls",
    "hybid",
]

CODEC_ID = "hrrr_gribberish"


# Vendored from Gribberish with some modifications


@dataclass(frozen=True)
class HRRRGribberishCodec(ArrayBytesCodec):
    """Transform GRIB2 bytes into zarr arrays using gribberish library"""

    var: str | None

    def __init__(self, var: str | None) -> None:
        object.__setattr__(self, "var", var)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, CODEC_ID, require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        if not self.var:
            return {"name": CODEC_ID}
        else:
            return {"name": CODEC_ID, "configuration": {"var": self.var}}

    async def _decode_single(
        self,
        chunk_data: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        assert isinstance(chunk_data, Buffer)
        chunk_bytes = chunk_data.to_bytes()
        if self.var == "latitude" or self.var == "longitude":
            message = parse_grib_message_metadata(chunk_bytes, 0)
            lat, lng = message.latlng()
            data = lat if self.var == "latitude" else lng  # type: ignore[no-redef]
        elif self.var == "time":
            message = parse_grib_message_metadata(chunk_bytes, 0)
            reference_date = message.reference_date
            data = np.datetime64(reference_date, "s")  # type: ignore[no-redef]

            # timestamp = reference_date.timestamp()
            # float64_ts = np.float64(timestamp)
            # data = float64_ts  # type: ignore[no-redef]
        elif self.var == "step":
            message = parse_grib_message_metadata(chunk_bytes, 0)
            forecast_date = message.forecast_date
            reference_date = message.reference_date
            step = forecast_date - reference_date
            data = np.timedelta64(step, "s")
        elif self.var in LEVEL_COORDINATES:
            message = parse_grib_message_metadata(chunk_bytes, 0)
            level_value = message.level_value
            data = np.float64(level_value)  # type: ignore[no-redef]
        else:
            data = parse_grib_array(chunk_bytes, 0)  # type: ignore[no-redef]

        if chunk_spec.dtype != data.dtype:
            data = data.astype(chunk_spec.dtype.to_native_dtype())  # type: ignore[no-redef]
        if data.shape != chunk_spec.shape:
            data = data.reshape(chunk_spec.shape)  # type: ignore[no-redef]

        return data

    async def _encode_single(
        self,
        chunk_data: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        # This is a read-only codec
        raise NotImplementedError

    def compute_encoded_size(
        self, input_byte_length: int, _chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError
