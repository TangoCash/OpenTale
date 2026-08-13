"""Configuration for the book generation system"""

import json
import os
from typing import Dict

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "model": "google/gemini-2.5-flash",
    "base_url": "http://127.0.0.1:1234/v1",
    "api_key": "",
    "temperature": 0.7,
    "top_p": 1.0,
    "seed": 42,
    "timeout": 1000,
    "max_tokens": 10000,
    "debug": False,
    "searxng_host": "",
    "research_agent_enabled": False,
}


def _load_config_file() -> Dict:
    """Load configuration from config.json, returning defaults if missing."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Merge with defaults so any missing keys get a default value
            merged = {**DEFAULT_CONFIG, **cfg}
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_CONFIG)


def save_config(config: Dict):
    """Persist configuration to config.json."""
    # Merge so we keep any keys that may have been added in later versions
    merged = {**DEFAULT_CONFIG, **config}
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def get_config() -> Dict:
    """Get the configuration for the agents from config.json."""

    cfg = _load_config_file()

    # Basic config for local LLM
    config_list = [
        {
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "api_key": cfg.get("api_key"),
        }
    ]

    # Common configuration for all agents
    agent_config = {
        "seed": int(cfg.get("seed", DEFAULT_CONFIG["seed"])),
        "temperature": float(cfg.get("temperature", DEFAULT_CONFIG["temperature"])),
        "top_p": float(cfg.get("top_p", DEFAULT_CONFIG["top_p"])),
        "config_list": config_list,
        "timeout": int(cfg.get("timeout", DEFAULT_CONFIG["timeout"])),
        "cache_seed": None,
        "max_tokens": int(cfg.get("max_tokens", DEFAULT_CONFIG["max_tokens"])),
    }

    return agent_config