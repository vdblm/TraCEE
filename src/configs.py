from typing import Optional
from pydantic import BaseModel, validator

NOISE_TYPES = ['gaussian', 'laplace', 'exponential', 'log-normal', 'uniform']
SCM_TYPES = ['linear']
MODELS = ['gpt2']


class CurriculumBaseConfig(BaseModel):
    start: int
    end: Optional[int]
    inc: int
    interval: int


class CurriculumConfig(BaseModel):
    dims: CurriculumBaseConfig
    points: CurriculumBaseConfig


class TransformerConfig(BaseModel):
    n_dims: Optional[int]
    n_embd: int
    n_layer: int
    n_head: int
    n_positions: int


class SCMConfig(BaseModel):
    scm_type: str
    noise_types: list or str
    x_dim: int
    t_dim: int
    y_dim: int

    @validator('noise_types')
    def noise_types_validation(cls, v):
        if isinstance(v, list):
            assert all([noise_type in NOISE_TYPES for noise_type in v])
        else:
            assert v in NOISE_TYPES, f"Noise type {v} not supported"
        return v

    @validator('scm_type')
    def scm_type_validation(cls, v):
        assert v in SCM_TYPES, f"SCM type {v} not supported"
        return v


class OptimizerConfig(BaseModel):
    learning_rate: float


class TrainConfig(BaseModel):
    train_steps: int
    batch_size: int
    save_every_steps: int
    keep_every_steps: int
    curriculum: CurriculumConfig
    scm: SCMConfig
    optimizer: OptimizerConfig
    model: TransformerConfig
    out_dir: str
    seed: int
    resume_id: Optional[str]
    test_run: bool
    

def populate_config(config: TrainConfig) -> TrainConfig:
    if config.curriculum.dims.end is None:
        config.curriculum.dims.end = config.scm.x_dim
    else:
        assert config.curriculum.dims.end <= config.scm.x_dim

    if config.curriculum.points.end is None:
        config.curriculum.points.end = config.model.n_positions
    else:
        assert config.curriculum.points.end <= config.model.n_positions

    scm_dim = config.scm.x_dim + config.scm.t_dim + config.scm.y_dim
    if config.model.n_dims is None:
        config.model.n_dims = scm_dim
    else:
        assert config.model.n_dims >= scm_dim

    return config
