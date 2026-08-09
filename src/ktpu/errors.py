class KtpuError(RuntimeError):
    """Base exception for actionable ktpu failures."""


class SafetyError(KtpuError):
    """Raised when a safety gate blocks a risky operation."""


class CheckpointError(SafetyError):
    """Raised when local code is not safely checkpointed remotely."""


class SizingError(SafetyError):
    """Raised when the model/request cannot fit safely."""


class EngineError(KtpuError):
    """Raised when engine setup or startup fails."""


class StreamingError(KtpuError):
    """Raised when the inference stream fails."""

