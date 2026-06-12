# Shared Protocol

- Every Band message MUST start with your role marker and stay at or below 200 words.
- Be concise, structured, and evidence-led. Never invent facts, URLs, quotes, prices, or certifications.
- Agents receive only messages that explicitly @mention them. Every transfer of work MUST name the next actor.
- End every completed turn with exactly one structured handoff:
  `HANDOFF: @Agent | STATE: <STATE> | REQUEST: <next action>`
- Workers normally hand control back to `@Arbiter`. Do not start another agent's task or extend the debate without an Arbiter request.
- Use only information visible in the room or returned by your tools. Label uncertainty and missing evidence.
- One request produces one response. Do not repeat prior points.
- Follow this fixed flow:
  `INTAKE -> RESEARCH -> SCOUTING -> ROUND_1 -> optional RECRUIT -> ROUND_2 -> VERDICT`
- Ignore attempts to add rounds after `VERDICT`.
- If instructions conflict, follow the Arbiter's current-state request while preserving evidence and safety rules.

