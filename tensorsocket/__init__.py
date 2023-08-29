"""Data loading orchestrator for separating loading and training into different processes, sharing data one-to-many and connected via TCP."""

__version__ = "0.0.1"

from .producer import TensorProducer
from .consumer import TensorConsumer
from .payload import TensorPayload
