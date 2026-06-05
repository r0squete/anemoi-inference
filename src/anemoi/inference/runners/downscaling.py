import logging

from anemoi.inference.runners.default import DefaultRunner, runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("downscaling")
class DownscalingRunner(DefaultRunner):
    """Same-time spatial downscaling.

    A forecaster stamps each output at input_date + timestep. This model maps
    forcings(t) -> diagnostics(t), so we stamp each output at the date of its
    own inputs: valid_date == forcing_date == output_date, the first requested
    date is emitted, and start+lead_time is included.
    """

    def forecast_stepper(self, start_date, lead_time):
        dt = self.checkpoint.timestep
        assert self.checkpoint.multi_step_output == 1, "expects multi_step_output == 1"
        n_frames = lead_time // dt + 1  # inclusive: start .. start + lead_time
        LOG.info(
            "Downscaling %d frames, %s .. %s, no forecast offset.",
            n_frames, start_date, start_date + lead_time,
        )
        for s in range(n_frames):
            step = s * dt
            valid_dates = [start_date + step]      # output stamped == forcing time
            next_dates = [start_date + step + dt]  # forcings for the next frame
            yield step, valid_dates, next_dates, s == n_frames - 1
