from src.configs import CurriculumConfig, CurriculumBaseConfig


class Curriculum:
    def __init__(self, conf: CurriculumConfig):
        self.conf = conf

    def get_max_dims(self):
        return self.conf.dims.end

    def get_max_points(self):
        return self.conf.points.end

    def get_n_dims(self, steps: int):
        return self._get_current_value(
            self.conf.dims.start,
            self.conf.dims.end,
            self.conf.dims.interval,
            self.conf.dims.inc,
            steps,
        )

    def get_n_points(self, steps: int):
        return self._get_current_value(
            self.conf.points.start,
            self.conf.points.end,
            self.conf.points.interval,
            self.conf.points.inc,
            steps,
        )

    @staticmethod
    def _get_current_value(start: int, end: int, interval: int, inc: int, steps: int):
        if interval == 0:
            return end
        return min(start + (steps // interval) * inc, end)

    @staticmethod
    def get_fixed_curriculum(n_points: int, n_dims: int) -> CurriculumConfig:
        dim_conf = CurriculumBaseConfig(start=n_dims, end=n_dims, inc=0, interval=1)
        point_conf = CurriculumBaseConfig(
            start=n_points, end=n_points, inc=0, interval=1
        )
        return CurriculumConfig(dims=dim_conf, points=point_conf)
