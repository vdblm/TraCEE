from src.configs import CurriculumConfig


class Curriculum:
    def __init__(self, conf: CurriculumConfig):
        self.n_dims_truncated = conf.dims.start
        self.n_points = conf.points.start
        self.n_dims_schedule = conf.dims
        self.n_points_schedule = conf.points
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self._update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self._update_var(self.n_points, self.n_points_schedule)

    def _update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)
