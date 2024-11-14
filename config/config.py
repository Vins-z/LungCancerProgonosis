import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Application configuration."""
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    MAX_DEPTH: int = 5
    OUTPUT_DIR: Path = Path("output")
    LOGGING_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)