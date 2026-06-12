from __future__ import annotations

import asyncio
import logging

from src.agents.arbiter import build_arbiter_agent
from src.agents.workers import build_worker_agent
from src.common.config import ROLES, load_agent_credentials, load_settings
from src.common.logging import configure_logging


async def main() -> None:
    configure_logging()
    settings = load_settings()
    credentials = load_agent_credentials()
    required = set(ROLES if settings.start_compliance else ROLES[:-1])
    missing = sorted(required - credentials.keys())
    if missing:
        raise RuntimeError(f"Missing Band credentials for: {', '.join(missing)}")

    agents = [build_arbiter_agent(credentials["arbiter"], settings)]
    for role in ROLES[1:]:
        if role == "compliance" and not settings.start_compliance:
            continue
        agents.append(build_worker_agent(role, credentials[role], settings))

    logging.getLogger(__name__).info("Starting %d Band agents", len(agents))
    await asyncio.gather(*(agent.run() for agent in agents))


if __name__ == "__main__":
    asyncio.run(main())
