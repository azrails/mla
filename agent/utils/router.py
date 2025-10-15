from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from .config import Config
import os
from loguru import logger
from typing import Literal


def get_model(cfg: Config, model_type: Literal["code", "feedback", "expert"] = "code"):
    model_base_url = os.getenv("OPENAI_BASE_URL")
    model = cfg.agent[model_type].model
    if model_base_url is None:
        logger.info(f"Selected OpenAI {model_type} model: {model}")
        return ChatOpenAI(
            model=model,
            temperature=cfg.agent.code.temp,
        )
    if model_base_url is not None and model_base_url.split("/")[-1] == "v1":
        logger.info(
            f"Selected OpenAI api compatible {model_type} model: {model} url: {model_base_url}"
        )
        return ChatOpenAI(
            model=model,
            temperature=cfg.agent.code.temp,
            openai_api_base=model_base_url,
            openai_api_key="...",
        )
    logger.info(f"Selected Ollama {model_type} model: {model} url: {model_base_url}")
    return ChatOllama(
        model=model,
        temperature=cfg.agent.code.temp,
    )
