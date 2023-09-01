from typing import Optional
from pydantic import BaseModel, validator
from torch.distributions import Normal, Uniform, Laplace, Exponential, LogNormal

NOISE_TYPES = [Normal, Uniform, Laplace, Exponential, LogNormal]

SCM_TYPES = ["linear"]
MODELS = ["gpt2"]


class CurriculumBaseConfig(BaseModel):
    start: int
    end: int
    inc: int
    interval: int


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
    noise_types: list
    x_dim: int
    t_dim: int
    y_dim: int
    conf_factor: float = None

    class Config:
        arbitrary_types_allowed = True

    @validator("noise_types")
    def noise_types_validation(cls, v):
        assert all(
            [
                any([isinstance(noise, noise_type) for noise_type in NOISE_TYPES])
                for noise in v
            ]
        )
        return v

    @validator("scm_type")
    def scm_type_validation(cls, v):
        assert v in SCM_TYPES, f"SCM type {v} not supported"
        return v


class DataConfig(BaseModel):
    scm: SCMConfig
    curriculum: CurriculumConfig
    num_samples: int = -1
    normalize: bool = False


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
    train_data: DataConfig
    val_data: DataConfig
    trainer: TrainerConfig
    model: TransformerConfig
    wandb: WandbConfig
    out_dir: str
    seed: int
    save_every_steps: int
    keep_every_steps: int
    test_run: bool
    partial_id: bool = False

    @validator("model")
    def model_validation(cls, model_value, values, field, config):
        for data in [values["train_data"], values["val_data"]]:
            curriculum_dims_end = data.curriculum.dims.end
            curriculum_points_end = data.curriculum.points.end
            scm_dims = data.scm.x_dim + data.scm.t_dim + data.scm.y_dim

            assert (
                model_value.n_dims >= scm_dims
            ), f"Model dimension {model_value.n_dims} is less than SCM dimension {scm_dims}"

            assert (
                model_value.n_positions >= curriculum_points_end
            ), f"Model seq length {model_value.n_positions} is less than curriculum end points {curriculum_points_end}"

            assert (
                curriculum_dims_end <= data.scm.x_dim
            ), f"Curriculum dimension {curriculum_dims_end} is greater than SCM dimension {data.scm.x_dim}"

        return model_value
