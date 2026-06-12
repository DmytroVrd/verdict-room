from __future__ import annotations

from pathlib import Path

from src.common.config import load_agent_credentials
from src.common.config import load_settings


def test_load_agent_credentials(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    path.write_text("arbiter:\n  agent_id: abc\n  api_key: secret\n", encoding="utf-8")
    result = load_agent_credentials(path)
    assert result["arbiter"].agent_id == "abc"
    assert result["arbiter"].api_key == "secret"


def test_load_settings_reads_turn_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEBATE_TURN_TIMEOUT", raising=False)
    path = tmp_path / ".env"
    path.write_text("DEBATE_TURN_TIMEOUT=120\n", encoding="utf-8")
    assert load_settings(path).debate_turn_timeout == 120.0
