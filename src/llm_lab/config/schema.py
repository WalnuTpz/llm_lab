from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal, get_args


ArchitectureName = Literal["original_transformer", "modern_decoder", "qwen36", "deepseek_v4"]
NormType = Literal["layernorm", "rmsnorm"]
ActivationType = Literal["relu", "silu", "swiglu"]
PositionEncoding = Literal["sinusoidal", "rope", "none"]
AttentionType = Literal[
    "mha",
    "gqa",
    "linear",
    "hybrid",
    "sliding",
    "compressed_sparse",
    "heavily_compressed",
]
FFNType = Literal["relu", "swiglu", "moe"]


@dataclass(slots=True)
class ModelConfig:
    architecture: ArchitectureName
    vocab_size: int = 16_384
    context_length: int = 1_024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int | None = None
    d_ff: int = 3_072
    tie_embeddings: bool = True

    norm_type: NormType = "rmsnorm"
    activation: ActivationType = "swiglu"
    ffn_type: FFNType = "swiglu"
    attention_type: AttentionType = "gqa"
    position_encoding: PositionEncoding = "rope"

    dropout: float = 0.0
    eps: float = 1e-5
    rope_theta: float = 10_000.0
    partial_rotary_factor: float = 1.0
    qk_norm: bool = False
    local_window: int | None = None

    full_attention_interval: int | None = None
    layer_types: list[str] = field(default_factory=list)

    num_experts: int | None = None
    active_experts: int | None = None
    shared_experts: int = 0
    expert_d_ff: int | None = None
    router_score: str = "softmax"
    moe_layer_types: list[str] = field(default_factory=list)

    mtp_layers: int = 0
    mtp_loss_weight: float = 0.0
    residual_streams: int = 1
    compressed_topk: int | None = None
    compression_ratio: int | None = None

    def validate(self) -> None:
        _ensure_choice("architecture", self.architecture, ArchitectureName)
        _ensure_choice("norm_type", self.norm_type, NormType)
        _ensure_choice("activation", self.activation, ActivationType)
        _ensure_choice("ffn_type", self.ffn_type, FFNType)
        _ensure_choice("attention_type", self.attention_type, AttentionType)
        _ensure_choice("position_encoding", self.position_encoding, PositionEncoding)

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_kv_heads is not None:
            if self.num_kv_heads <= 0:
                raise ValueError("num_kv_heads must be positive when set")
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.partial_rotary_factor <= 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        if self.layer_types and len(self.layer_types) != self.num_layers:
            raise ValueError("layer_types length must equal num_layers")
        if self.full_attention_interval is not None and self.full_attention_interval <= 0:
            raise ValueError("full_attention_interval must be positive when set")
        if self.local_window is not None and self.local_window <= 0:
            raise ValueError("local_window must be positive when set")
        if self.ffn_type == "moe":
            if not self.num_experts or self.num_experts <= 0:
                raise ValueError("moe ffn_type requires positive num_experts")
            if not self.active_experts or self.active_experts <= 0:
                raise ValueError("moe ffn_type requires positive active_experts")
            if self.active_experts > self.num_experts:
                raise ValueError("active_experts cannot exceed num_experts")
        if self.moe_layer_types and len(self.moe_layer_types) != self.num_layers:
            raise ValueError("moe_layer_types length must equal num_layers")
        if self.mtp_layers < 0:
            raise ValueError("mtp_layers cannot be negative")
        if self.residual_streams <= 0:
            raise ValueError("residual_streams must be positive")


@dataclass(slots=True)
class DataConfig:
    train_data: str | None = None
    val_data: str | None = None
    tokenizer_path: str | None = None
    dataset_dtype: str = "int32"
    add_eot: bool = True
    eot_token: str = "<|endoftext|>"

    def validate(self) -> None:
        if self.dataset_dtype not in {"uint16", "int32", "int64"}:
            raise ValueError("dataset_dtype must be one of uint16, int32, int64")


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    max_iters: int = 1_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 100
    cosine_cycle_iters: int = 1_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    seed: int = 0
    dtype: str = "float32"
    device: str = "cpu"

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate cannot be negative")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")


@dataclass(slots=True)
class RuntimeConfig:
    log_interval: int = 10
    eval_interval: int = 200
    eval_batches: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    run_dir: str = "runs"
    use_wandb: bool = False

    def validate(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        self.model.validate()
        self.data.validate()
        self.training.validate()
        self.runtime.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def experiment_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise TypeError("experiment config must be a mapping")
    if "model" not in raw:
        raise ValueError("experiment config requires a model section")

    cfg = ExperimentConfig(
        name=str(raw.get("name", raw["model"].get("architecture", "experiment"))),
        model=_dataclass_from_dict(ModelConfig, raw["model"]),
        data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
        training=_dataclass_from_dict(TrainingConfig, raw.get("training", {})),
        runtime=_dataclass_from_dict(RuntimeConfig, raw.get("runtime", {})),
    )
    cfg.validate()
    return cfg


def _dataclass_from_dict(cls: type[Any], raw: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{cls.__name__} config must be a mapping")

    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {cls.__name__} field(s): {joined}")
    return cls(**raw)


def _ensure_choice(name: str, value: str, literal: Any) -> None:
    allowed = set(get_args(literal))
    if value not in allowed:
        joined = ", ".join(sorted(str(v) for v in allowed))
        raise ValueError(f"{name} must be one of: {joined}")
