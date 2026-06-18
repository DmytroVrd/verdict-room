# Voiceover text for AI TTS (ElevenLabs / OpenAI)

> Генеруй ПО СЕГМЕНТАХ (кожен блок окремо) — так легше синхронізувати зі
> слайдами в монтажі. Голос: Brian / Adam (US) або George (UK), neutral, calm.
> Speed ~1.0, stability ~50%. Експорт mp3, клади під відеоряд.
> "—" у тексті = коротка пауза (TTS читає природно).

---

### SEG 1 — Title
Verdict Room. An AI courtroom for buying decisions.

---

### SEG 2 — Problem
Every time a company picks a software vendor, one person spends twenty to forty hours on research — pricing, security, GDPR, migration. It drags on for two to four weeks. And in the end, you get one biased summary. No debate. No second opinion. Procurement teams do this dozens of times a year. We turned that into a debate between AI agents.

---

### SEG 3 — Solution
Here is how it works. You drop a purchase case into a Band room, and six agents take over. A Researcher gathers the facts. A Scout finds alternatives. An Advocate argues for the deal. A Critic argues against it. And an Arbiter runs the debate and issues a scored verdict. Like an investment committee — but in minutes, not weeks.

---

### SEG 4 — Demo (DETAILED, over the live Band recording, ~2:05)
> Score 75 = the canonical recorded run. If you record a different run, swap the
> number to match the screen. Generate this as ONE mp3, or split by the ↘ marks.

Here is a real case. A fifty-person EU software company is choosing its main workspace — Notion, Confluence, or Slite. The priorities: total cost, migration effort, security, and GDPR. I drop the case into a Band room and mention the Arbiter. ↘

The Arbiter opens the case and hands off to the Researcher. It gathers sourced facts — Notion's pricing tiers, from free up to twenty dollars per user on the Business plan; its GDPR data-processing addendum with standard contractual clauses for EU transfers; and its migration tools that import from Confluence, Trello, and Google Docs. Every claim comes with a real source link. ↘

The Scout compares the alternatives — Confluence, bundled with Jira but pricier at scale; and Slite, simpler, but with thinner enterprise coverage. ↘

Then the debate opens. The Critic attacks — per-seat pricing risks budget creep, migrating off Confluence carries hidden effort, and there's a real question of where EU data actually lives. ↘

Now watch this. The Critic flags a GDPR concern — and the Arbiter does something new. It discovers and pulls a Compliance specialist into the room, live, in the middle of the debate. Nobody wired it in advance. Agents recruiting agents. ↘

The Compliance agent reviews data residency, the processing addendum, and certifications — then feeds its assessment straight back into the debate. ↘

The Advocate answers point by point, citing the gathered evidence. After two full rounds, the Arbiter closes the case. ↘

The verdict: Buy with conditions — seventy-five out of a hundred. With concrete conditions — lock down final pricing, run a migration pilot — and an explicit dissent on unresolved security questions. All of this took about three minutes. Not three weeks.

---

### SEG 5 — Originality
Two things here are new. First — dynamic recruitment. The Arbiter pulls a new agent into a live debate the moment it is needed. Agents recruiting agents. Second — a cross-model courtroom. The debaters run on different models, across two providers. So it is a real panel — not one model talking to itself.

---

### SEG 6 — Architecture
The architecture. Six agents in one Band room. The tool-using agents — Arbiter, Researcher, and Scout — run on AI/ML API. The debate agents — Advocate, Critic, and Compliance — run on Featherless, on DeepSeek models. And Band is the coordination layer. Every handoff, every fact, every debate turn, and the final verdict — all of it happens inside one Band room.

---

### SEG 7 — Business
Who is this for? Procurement and operations teams at companies of fifty to five hundred people. They run vendor picks every month. They pay per verdict, or per seat. The procurement software market is around seven billion dollars — and the AI part is growing fastest.

---

### SEG 8 — Why Band
And this is exactly the use case Band itself highlights — procurement workflows where agents share context and recruit each other. We built exactly that.

---

### SEG 9 — Closing
Verdict Room. Six agents, one defensible verdict — built on Band. Thank you.

---

## Робочий процес
1. elevenlabs.io → sign up (free) → Voices → обери Brian/Adam/George.
2. Вставляй SEG 1…9 по черзі → Generate → Download кожен mp3 (назви seg1.mp3 …).
3. У відеоредакторі поклади mp3 під відповідний слайд / під Band-запис (SEG 4).
4. SEG 4 — найдовший; підлаштуй темп показу Band-прогону під цей голос.
5. Якщо символів не вистачить на free — згенеруй найдовші сегменти (2,4) там,
   а короткі можеш сам або іншим акаунтом. Або OpenAI TTS (дешево, без ліміту).
