# Verdict Room

**Verdict Room turns weeks of vendor due diligence into a 3-minute multi-agent debate - and gives you a defensible verdict, not just an opinion.**

[View the live demo](https://dmytrovrd.github.io/verdict-room/)

## Problem -> Solution

Choosing a B2B vendor can take 20-40 hours across pricing, security, GDPR, migration risk, and alternatives. The result is often one person's biased summary with no adversarial review. Verdict Room puts six specialized AI agents into a shared Band room to research, challenge, defend, and score the decision. You get a transparent `BUY`, `BUY_WITH_CONDITIONS`, or `AVOID` verdict in minutes.

## Architecture

```text
                               BAND ROOM
                      Shared context + message bus

  Human case ---------------------------------------------------------+
       |                                                             |
       v                                                             v
  +-----------+     +------------+     +-------+     +----------+     |
  |  Arbiter  |<--->| Researcher |<--->| Scout |<--->| Advocate |     |
  +-----------+     +------------+     +-------+     +----------+     |
       ^                                                             |
       |              +----------+     +------------+                |
       +------------->|  Critic  |<--->| Compliance |<---------------+
                      +----------+     +------------+

                 Band carries every handoff, fact,
                 debate turn, recruitment, and verdict.
```

## Six Agents, Two Partner Platforms

Verdict Room runs on **<u>AI/ML API</u>** and **<u>Featherless</u>**, partner technologies provided for the Band of Agents Hackathon.

| Agent | Role | Provider / Model |
| --- | --- | --- |
| Arbiter | Orchestrates the courtroom and issues the scored verdict | **AI/ML API** / `openai/gpt-4.1-mini` |
| Researcher | Collects sourced pricing, features, and risks | **AI/ML API** / `openai/gpt-4.1-mini` |
| Scout | Finds alternatives and compares trade-offs | **AI/ML API** / `openai/gpt-4.1-mini` |
| Advocate | Builds the evidence-based case for adoption | **Featherless** / `deepseek-ai/DeepSeek-V3.2` |
| Critic | Attacks assumptions, hidden costs, and weak evidence | **Featherless** / `deepseek-ai/DeepSeek-V3.1-Terminus` |
| Compliance | Joins dynamically when privacy or regulatory risk appears | **Featherless** / `deepseek-ai/DeepSeek-V3.2` |

## How It Works

```text
Case submitted
      |
      v
Researcher + Scout gather evidence and alternatives
      |
      v
Advocate vs Critic: two adversarial debate rounds
      |
      v
Arbiter dynamically recruits Compliance when risk is detected
      |
      v
Scored verdict + conditions + dissent
```

The originality is explicit: **dynamic agent recruitment** brings Compliance into a live case only when needed, while the **cross-model courtroom** uses AI/ML API and Featherless to create a real multi-perspective panel instead of one model debating itself.

## Quickstart

Requirements: Python 3.11+, `uv`, a Band account with six agents, and AI/ML API plus Featherless keys.

```powershell
uv sync
Copy-Item .env.example .env
Copy-Item agent_config.yaml.example agent_config.yaml
```

Add `AIML_API_KEY` and `FEATHERLESS_API_KEY` to `.env`, then add each Band agent's UUID and one-time API key to `agent_config.yaml`.

```powershell
uv run python -m src.run_all
```

Create a Band room, add Arbiter, Researcher, Scout, Critic, and Advocate, then mention `@Arbiter` with a vendor decision. Compliance is recruited into the room dynamically when required.

## Demo

**Live report:** [dmytrovrd.github.io/verdict-room](https://dmytrovrd.github.io/verdict-room/)

**Demo video:** Coming soon.

## Hackathon

Built for the **Band of Agents Hackathon**.

## License

[MIT](LICENSE)
