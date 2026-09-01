"""Shared test isolation settings.

Unit tests construct LangChain OpenAI clients but replace all model calls with
fakes.  A deterministic placeholder prevents client construction from reading
a developer's real ``.env`` file or failing in a clean CI environment.
"""

from __future__ import annotations

import os


os.environ["OPENAI_API_KEY"] = "test-key-not-a-real-credential"
