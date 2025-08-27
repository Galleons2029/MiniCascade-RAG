# -*- coding: utf-8 -*-
# @Time   : 2025/8/16 22:38
# @Author : Galleons
# @File   : __init__.py.py

"""This file contains the prompts for the agent."""

import os
from datetime import datetime

from app.core.config import settings


def load_system_prompt(prompt_path: str, agent_name: str, **kwargs):
    """Load the system prompt from the file."""
    with open(os.path.join(os.path.dirname(__file__), prompt_path), "r") as f:
        return f.read().format(
            agent_name=agent_name,
            current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


SYSTEM_PROMPT = load_system_prompt(prompt_path="system.md",
                                   agent_name="chief agent"
                                   )
COORDINATOR_PROMPT = load_system_prompt(prompt_path="coordinator.md",
                                        agent_name="Task Coordinator Agent"
                                        )

