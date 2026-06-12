from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_agent_credentials, load_settings


async def check_role(
    client: httpx.AsyncClient,
    role: str,
    api_key: str,
) -> tuple[str, bool, str]:
    try:
        response = await client.get(
            "/api/v1/agent/me",
            headers={"X-API-Key": api_key},
        )
        response.raise_for_status()
        payload = response.json()
        name = payload.get("name") or payload.get("data", {}).get("name") or "connected"
        return role, True, str(name)
    except Exception as exc:
        return role, False, f"{type(exc).__name__}: {exc}"


async def main() -> None:
    settings = load_settings()
    credentials = load_agent_credentials()
    async with httpx.AsyncClient(
        base_url=settings.band_rest_url,
        timeout=15,
    ) as client:
        results = await asyncio.gather(
            *(
                check_role(client, role, item.api_key)
                for role, item in credentials.items()
            )
        )
    for role, ok, detail in results:
        print(f"[{'OK' if ok else 'FAIL'}] {role}: {detail}")


if __name__ == "__main__":
    asyncio.run(main())
