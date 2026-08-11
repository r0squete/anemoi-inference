# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import threading
from pathlib import Path

import numpy as np

from anemoi.inference.context import Context
from anemoi.inference.metadata import Metadata
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..decorators import ensure_path
from ..decorators import format_dataset_name
from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


# In case HDF5 was not compiled with thread safety on
LOCK = threading.RLock()


@output_registry.register("netcdf")
@main_argument("path")
@format_dataset_name("path")
@ensure_path("path")
class NetCDFOutput(Output):
    """NetCDF output class."""

    def __init__(
        self,
        context: Context,
        metadata: Metadata,
        *,
        path: Path,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        float_size: str = "f4",
        missing_value: float | None = np.nan,
        zlib: bool = False,
        complevel: int = 4,
        shuffle: bool = True,
    ) -> None:
        """Initialise the NetCDF output object.

        Parameters
        ----------
        context : dict
            The context dictionary.
        path : Path
            The path to save the NetCDF file to.
            If the parent directory does not exist, it will be created.
        variables : list, optional
            The list of variables to write, by default None.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        float_size : str, optional
            The size of the float, by default "f4".
        missing_value : float, optional
            The missing value, by default np.nan.
        zlib : bool, optional
            Whether to zlib-compress the variables, by default False.
            When False, `complevel` and `shuffle` are ignored.
        complevel : int, optional
            zlib compression level, 1 (fastest) to 9 (smallest), by default 4.
        shuffle : bool, optional
            Whether to apply the HDF5 shuffle filter, by default True.
            It reorders bytes before deflating, which usually improves the
            ratio on float data at negligible cost.
        """

        super().__init__(
            context,
            metadata,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )

        from netCDF4 import Dataset

        self.path = path
        self.ncfile: Dataset | None = None
        self.float_size = float_size
        self.missing_value = missing_value
        self.compression = dict(zlib=zlib, complevel=complevel, shuffle=shuffle) if zlib else {}

    def __repr__(self) -> str:
        """Return a string representation of the NetCDFOutput object."""
        return f"NetCDFOutput({self.path})"

    def open(self, state: State) -> None:
        """Open the NetCDF file and initialize dimensions and variables.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        from netCDF4 import Dataset

        with LOCK:
            if self.ncfile is not None:
                return

        # If the file exists, we may get a 'Permission denied' error
        if os.path.exists(self.path):
            os.remove(self.path)

        with LOCK:
            self.ncfile = Dataset(self.path, "w", format="NETCDF4")

        state = self.post_process(state)

        compression = self.compression

        values = len(state["latitudes"])
        self.reference_date = state["date"]

        if reference_date := getattr(self.context, "reference_date", None):
            self.reference_date = reference_date

        # Ensemble size comes from the runner (e.g. DiffusionDownscalingRunner).
        # Non-ensemble runners have no `num_members`, so this falls back to 1 and
        # the output keeps its original (time, values) layout.
        self.num_members = getattr(self.context, "num_members", 1)
        self.ensemble = self.num_members > 1

        with LOCK:
            self.values_dim = self.ncfile.createDimension("values", values)
            # unlimited time dimension avoids pre-allocation mismatches
            self.time_dim = self.ncfile.createDimension("time", None)
            self.time_var = self.ncfile.createVariable("time", "i4", ("time",), **compression)

            self.time_var.units = f"seconds since {self.reference_date}"
            self.time_var.long_name = "time"
            self.time_var.calendar = "gregorian"

            # Ensemble members become their own fixed dimension + coordinate.
            if self.ensemble:
                self.ncfile.createDimension("member", self.num_members)
                member_var = self.ncfile.createVariable("member", "i4", ("member",), **compression)
                member_var.long_name = "ensemble member"
                member_var[:] = np.arange(self.num_members)

        with LOCK:
            latitudes = state["latitudes"]

            self.latitude_var = self.ncfile.createVariable("latitude", self.float_size, ("values",), **compression)
            self.latitude_var.units = "degrees_north"
            self.latitude_var.long_name = "latitude"

            longitudes = state["longitudes"]

            self.longitude_var = self.ncfile.createVariable("longitude", self.float_size, ("values",), **compression)
            self.longitude_var.units = "degrees_east"
            self.longitude_var.long_name = "longitude"

            self.latitude_var[:] = latitudes
            self.longitude_var[:] = longitudes

        # Map "seconds since reference_date" -> time index, so members sharing a
        # date land on the same time slot instead of being smeared along time.
        self._time_indices = {}
        self.vars = {}

    def ensure_variables(self, state: State) -> None:
        """Ensure that all variables are created in the NetCDF file.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        values = len(state["latitudes"])

        compression = self.compression

        for name in state["fields"].keys():
            if self.skip_variable(name):
                continue

            if name in self.vars:
                continue

            # Ensemble runs add a `member` axis between time and values.
            if self.ensemble:
                dims = ("time", "member", "values")
                chunksizes = (1, 1, values)
            else:
                dims = ("time", "values")
                chunksizes = (1, values)

            while np.prod(chunksizes) > 1000000:
                chunksizes = tuple(int(np.ceil(x / 2)) for x in chunksizes)

            with LOCK:
                missing_value = self.missing_value

                # A NaN _FillValue/missing_value makes CF readers (e.g. xarray with
                # the default mask_and_scale=True) blank the ENTIRE variable on read.
                # Use a real, out-of-range sentinel instead, so valid data reads back
                # unmasked while genuine gaps stay detectable. Only _FillValue and
                # missing_value are CF-standard; drop the non-standard `fill_value`.
                if missing_value is None or (
                    isinstance(missing_value, float) and np.isnan(missing_value)
                ):
                    from netCDF4 import default_fillvals

                    missing_value = default_fillvals[self.float_size]

                self.vars[name] = self.ncfile.createVariable(
                    name,
                    self.float_size,
                    dims,
                    chunksizes=chunksizes,
                    fill_value=missing_value,
                    **compression,
                )

                self.vars[name].missing_value = missing_value

    def write_step(self, state: State) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        self.ensure_variables(state)

        # Time index is keyed by date, not a raw counter, so all members of the
        # same valid date write into one time slot.
        step = state["date"] - self.reference_date
        secs = step.total_seconds()
        if secs not in self._time_indices:
            self._time_indices[secs] = len(self._time_indices)
            self.time_var[self._time_indices[secs]] = secs
        t = self._time_indices[secs]

        # States without a member key (non-ensemble runs, or the initial state)
        # map to member 0.
        member = state.get("member", 0)

        for name, value in state["fields"].items():
            if self.skip_variable(name):
                continue

            with LOCK:
                if self.ensemble:
                    self.vars[name][t, member] = value
                else:
                    self.vars[name][t] = value

    def close(self) -> None:
        """Close the NetCDF file."""
        if self.ncfile is not None:
            with LOCK:
                self.ncfile.close()
            self.ncfile = None
