import warnings

# Suppress torchcodec incompatibility warning emitted by pyannote on import.
# torchcodec does not support the installed PyTorch version; audio is loaded
# via ffmpeg/numpy anyway so this has no functional impact.
warnings.filterwarnings(
    "ignore",
    message="torchcodec is not installed correctly",
    category=UserWarning,
    module=r"pyannote\.audio\.core\.io",
)

