from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROLES, load_settings
from src.common.llm import LLMConfigurationError, available_model_specs, make_llm


def main() -> None:
    settings = load_settings()
    for role in ROLES:
        specs = available_model_specs(role, settings)
        if not specs:
            print(f"[WARN] {role}: no configured provider key")
            continue
        try:
            response = make_llm(role, settings).invoke(
                "Reply with exactly: pong", config={"run_name": f"smoke-{role}"}
            )
            print(
                f"[OK] {role} ({specs[0].provider}:{specs[0].model}): {response.content}"
            )
        except LLMConfigurationError as exc:
            print(f"[WARN] {role}: {exc}")
        except Exception as exc:
            print(f"[FAIL] {role}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
