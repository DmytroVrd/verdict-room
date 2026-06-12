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

## Live verification

Verified on 2026-06-12 with six remote agents:

- `scripts/smoke_band.py`: 6/6 identities authenticated.
- Six agents run concurrently in one asyncio process without WebSocket conflicts.
- `@Arbiter ping` returned a PONG in about one second.
- A full Band-room debate reached a saved JSON verdict.
- Dynamic Compliance recruitment succeeded through participant tools.
- Structured verdict events were accepted.
- Mention filtering is strict in practice: downstream handoffs must carry a
  compact case/evidence digest because agents cannot rely on room-wide history.
- Timeout logs confirmed the configured 120-second reminder and second
  120-second auto-advance windows. Band UI ordering can make reminders appear
  visually adjacent to a later handoff.
- A clean post-fix run on 2026-06-13 completed in about three minutes:
  Researcher replied in 31 seconds, Scout in 18 seconds, no timeout reminders
  fired, Compliance was recruited dynamically, and the verdict had no evidence
  gaps.
- An immediate repeat run also completed cleanly: Researcher replied in 42
  seconds, Scout in 14 seconds, no reminders or runtime errors fired, and the
  verdict again had no evidence gaps.
- Tool-capable LangGraph workers can finish successfully without calling
  `band_send_message`. Verdict Room now requires that tool in the prompt and
  deterministically sends the final model text when the call is omitted.
- Band REST serializes real mentions as `@[[uuid]]`; case sanitization removes
  incoming transport tokens from the copied brief, while the platform may still
  show this representation when messages are fetched through the API.
