from __future__ import annotations

from pathlib import Path

from src.common.config import load_agent_credentials


def test_load_agent_credentials(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    path.write_text("arbiter:\n  agent_id: abc\n  api_key: secret\n", encoding="utf-8")
    result = load_agent_credentials(path)
    assert result["arbiter"].agent_id == "abc"
    assert result["arbiter"].api_key == "secret"
