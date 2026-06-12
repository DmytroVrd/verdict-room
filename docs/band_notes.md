# Band SDK notes

Verified on 2026-06-12 with `band-sdk==1.0.0`.

- Python requirement is 3.11+ in the published package.
- Import path is `band`, not the older `thenvoi` path still visible in some docs.
- Persistent lifecycle: `Agent.create(...)` then `await agent.run()`.
- `Agent.start()` initializes REST metadata, starts the adapter, then WebSocket
  processing. `Agent.stop(timeout=...)` supports graceful shutdown.
- LangGraph instructions belong in `custom_section`; custom LangChain tools use
  `additional_tools`.
- Built-in tool names use the `band_*` prefix in SDK 1.0.
- Each agent needs a distinct adapter instance and distinct Band credentials.
- Current identity smoke endpoint: `GET /api/v1/agent/me` with `X-API-Key`.
- Agent chat events: `POST /api/v1/agent/chats/{chat_id}/events`.
- Participant management:
  `GET|POST /api/v1/agent/chats/{chat_id}/participants`.

Live behavior still needs verification after real credentials are configured.
