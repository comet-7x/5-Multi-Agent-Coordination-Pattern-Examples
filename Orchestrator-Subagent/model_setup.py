import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelPlatformType

load_dotenv()

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class ModelConfig:
    url: str
    name: str
    api_key: str
    config: dict[str, Any]


def get_model_config() -> ModelConfig:
    with open(_CONFIG_PATH) as f:
        yaml_config = yaml.safe_load(f)

    return ModelConfig(
        url=os.environ["MODEL_URL"],
        name=os.environ["MODEL_NAME"],
        api_key=os.environ["MODEL_API_KEY"],
        config=yaml_config["model"],
    )


def get_model_backend() -> BaseModelBackend:
    model_config = get_model_config()
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_config.name,
        api_key=model_config.api_key,
        url=model_config.url,
        model_config_dict=model_config.config,
    )
