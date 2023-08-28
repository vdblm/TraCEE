from typing import Optional
from pydantic import BaseModel, validator

NOISE_TYPES = ["gaussian", "laplace", "exponential", "log-normal", "uniform"]
SCM_TYPES = ["linear"]
MODELS = ["gpt2"]


class CurriculumBaseConfig(BaseModel):
    start: int
    end: int
    inc: int
    interval: int


# TODO fill out the inc and interval based on training steps
class CurriculumConfig(BaseModel):
    dims: CurriculumBaseConfig
    points: CurriculumBaseConfig


class TransformerConfig(BaseModel):
    n_dims: int
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
    conf_factor: float = None

    @validator("noise_types")
    def noise_types_validation(cls, v):
        if isinstance(v, list):
            assert all([noise_type in NOISE_TYPES for noise_type in v])
        else:
            assert v in NOISE_TYPES, f"Noise type {v} not supported"
        return v

    @validator("scm_type")
    def scm_type_validation(cls, v):
        assert v in SCM_TYPES, f"SCM type {v} not supported"
        return v


class TrainerConfig(BaseModel):
    train_steps: int
    batch_size: int
    learning_rate: float


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str]
    log_every_steps: int
    run_name: Optional[str]
    run_id: Optional[str]
    resume: bool = False


class TraCEEConfig(BaseModel):
    save_every_steps: int
    keep_every_steps: int
    curriculum: CurriculumConfig
    scm: SCMConfig
    validate_scm: SCMConfig
    train: TrainerConfig
    model: TransformerConfig
    out_dir: str
    seed: int
    test_run: bool
    wandb: WandbConfig

    @validator("model")
    def model_validation(cls, model_value, values, field, config):
        curriculum_dims_end = values["curriculum"].dims.end
        curriculum_points_end = values["curriculum"].points.end
        scm_dims = values["scm"].x_dim + values["scm"].t_dim + values["scm"].y_dim

        assert (
            model_value.n_dims >= scm_dims
        ), f"Model dimension {model_value.n_dims} is less than SCM dimension {scm_dims}"

        assert (
            model_value.n_positions >= curriculum_points_end
        ), f"Model seq length {model_value.n_positions} is less than curriculum end points {curriculum_points_end}"

        assert (
            curriculum_dims_end <= values["scm"].x_dim
        ), f"Curriculum dimension {curriculum_dims_end} is greater than SCM dimension {values['scm'].x_dim}"

        return model_value
