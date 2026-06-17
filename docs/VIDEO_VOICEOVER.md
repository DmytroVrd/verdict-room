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

### SEG 4 — Demo (over the live Band recording)
Here is a real case. Should a fifty-person EU software company adopt Notion as their main workspace, instead of Confluence or Slite? The priorities — cost, migration, security, and GDPR. I send this to the Arbiter in our Band room. The Arbiter opens the case and hands it to the Researcher, who pulls real facts with sources — pricing, security, GDPR terms. Then the Scout compares the alternatives. Now the debate begins. The Critic attacks — hidden costs, migration pain, and a data-privacy risk. Now watch this. The Critic raises a GDPR concern, and the Arbiter does something new. It pulls a Compliance specialist into the room, live, in the middle of the debate. Nobody added it by hand. Agents recruiting agents. The Compliance agent reviews data residency and GDPR, and feeds its assessment back into the debate. The Advocate defends, point by point. And after two rounds, the Arbiter gives the final verdict — Buy with conditions, seventy-five out of one hundred — with clear conditions, and a dissent. All of this took about three minutes. Not three weeks.

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
