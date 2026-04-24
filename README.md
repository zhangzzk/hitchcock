<div align="center">

<img src="./docs/logo.png" width="760" alt="Hitchcock — AI cinema agent"/>

**Turn a book chapter — or any text — into a stylistically consistent AI-generated film.**

Hitchcock is a **fully CLI-based, agent-friendly** multi-stage pipeline that drafts a script, storyboards every shot, generates art, and renders scenes.

![Python](https://img.shields.io/badge/python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-v0.2%20·%20Phase%201-orange?style=flat-square)
![CLI-first](https://img.shields.io/badge/interface-CLI--first-000000?style=flat-square&logo=gnubash&logoColor=white)
![Agent-friendly](https://img.shields.io/badge/agent--friendly-JSON%20I%2FO-6e5494?style=flat-square)
[![Video: Seedance 2.0](https://img.shields.io/badge/video-Seedance%202.0-e63946?style=flat-square)](https://www.volcengine.com/product/seedance)
[![Image: Nano Banana Pro](https://img.shields.io/badge/image-Nano%20Banana%20Pro-4285f4?style=flat-square)](https://deepmind.google/technologies/gemini/)
[![LLM: Xiaomi MiMo](https://img.shields.io/badge/LLM-Xiaomi%20MiMo-ff6700?style=flat-square)](#)

[Contract spec (AGENTS.md)](./AGENTS.md)

</div>

---

## 🎬 Demo

<div align="center">

[<img src="./docs/demo_thumb.svg" width="760" alt="Watch the Hitchcock demo reel on Bilibili"/>](https://www.bilibili.com/video/BV1A7oGBUEja/)

*A 70-second epilogue generated end-to-end by Hitchcock — script → storyboard → scene art → Seedance 2.0 clips → ffmpeg concat. Source: one chapter of Chinese prose, ~1.8k characters.*

</div>

---

## ⚡ What it does

You bring a block of text — a book chapter, a screenplay fragment, a scene description. Hitchcock turns it into:

> **A 60–180s animated short**, stylistically consistent, scene-by-scene faithful to the source, with diegetic voice lines and zero hand-editing of intermediate artifacts.

Every stage is **gate-based**: `generate` → `show` → `refine --feedback "..."` → `approve`. Python code holds meta-prompts, schemas, and plumbing only. Every story-specific artifact — scene titles, camera choices, dialogue, scene-art prompts, Seedance prompts — is MIMO-authored and editable only through natural-language feedback.

---

## 🤖 Fully CLI · agent-drivable

There is no GUI, no web dashboard, no hand-edited YAML. A single `hitchcock` CLI exposes every stage, every refine knob, every render path — and emits `--json` everywhere so another AI can be the director:

- **One verb-noun command per action.** `hitchcock <stage> <generate|show|refine|approve>` — seven stages, four verbs. An agent can memorize the whole contract from one table.
- **Machine-readable state.** `hitchcock status -s <story> --json` returns per-stage state + a `next_action` field. A driver reads it, runs that command, loops.
- **Natural-language feedback as the only editing surface.** `refine --feedback "merge s03 and s04, keep all dialogue"`. No file-poking, no prompt-engineering per run.
- **Explicit error contract.** Every non-zero exit prints `hitchcock-error: <CODE>: <msg>` — parseable, recoverable. Full code list in [AGENTS.md §5](./AGENTS.md).
- **Scriptable end-to-end.** From source text to `reel.mp4` in one bash for-loop, or one sub-agent call.

See **[AGENTS.md](./AGENTS.md)** — the CLI contract is the spec, written *for* an AI driver.

---

## 🔄 Pipeline

| # | Stage | Input | Output |
|---|---|---|---|
| 0  | 🎯 **brief**       | director intent (free-form paragraph)    | structured answers + canon research via Gemini grounding |
| 0b | 🎨 **style**       | brief                                     | art-direction spec (palette / lighting / motifs / avoid) + anchor images |
| 0c | 👥 **cast**        | source text                               | auto-discovered characters + locations with T2I portraits |
| 1  | 📜 **script**      | everything above                          | structured Story JSON (scenes, beats, dialogue, emotional register) |
| 2  | 🎞 **storyboard**  | approved script                           | per-scene shot breakdown + Seedance-ready prompts |
| 3  | 🖼 **art**         | approved storyboard                       | Nano Banana Pro first-frame art (N candidates per shot) |
| 4  | 🎥 **render**      | approved art                              | Seedance 2.0 clips → ffmpeg concat → `reel.mp4` |

Each gate locks the next. All commands accept `--json` for machine-readable output.

---

## 🚀 Quickstart

```bash
cp .env.example .env        # fill in your API keys
pip install -e .

# Bootstrap reusable characters / locations into the bible (one-shot)
hitchcock design   path/to/character.txt
hitchcock location path/to/location.txt

# Gate-based story pipeline
hitchcock init          -s my-arc --source path/to/source.txt
hitchcock brief answer  -s my-arc --intent "faithful to the original, painterly Arcane style, 3-minute short"
hitchcock brief research -s my-arc && hitchcock brief approve -s my-arc
hitchcock style  generate -s my-arc && hitchcock style  approve -s my-arc
hitchcock cast   discover -s my-arc && hitchcock cast   build   -s my-arc && hitchcock cast approve -s my-arc
hitchcock script generate -s my-arc && hitchcock script approve -s my-arc
hitchcock storyboard generate -s my-arc && hitchcock storyboard approve -s my-arc
hitchcock art    generate -s my-arc --candidates 2
hitchcock art    pick     -s my-arc --scene s01 --shot sh01 --candidate 2
hitchcock art    approve  -s my-arc

# Render via Seedance 2.0 (Volcengine Ark)
hitchcock render seedance -s my-arc --scene s01,s02,s03,s04,s05
hitchcock render post     -s my-arc                        # → reel.mp4
```

**Hand-tuning a scene before it hits the video model:**
```bash
hitchcock render package  -s my-arc
# edit render/packages/<scene>/prompt.txt by hand
hitchcock render seedance -s my-arc --scene <csv> --use-package-prompt
```

**Refine at any stage, without touching code:**
```bash
hitchcock script refine     -s my-arc --feedback "merge s03 and s04; keep all dialogue"
hitchcock storyboard refine -s my-arc --scene s06 --feedback "both figures barefoot on same rock, over-shoulder"
hitchcock art refine        -s my-arc --scene s12 --shot sh01 --feedback "over-shoulder POV looking up at sky"
```

---

## 🏗 Architecture invariant

> **Python code holds meta-prompts + schemas + plumbing only.**
> Every story-specific artifact — titles, beats, VO lines, camera specs, scene-art prompts, Seedance prompts — is MIMO-authored and editable only via `refine --feedback "..."`.

Full CLI contract, data model on disk, error codes, and known-issues log: see **[AGENTS.md](./AGENTS.md)**.

---

## 🧰 Model stack

| Role | Model |
|---|---|
| LLM (every `refine` call) | **Xiaomi MiMo** — OpenAI-compatible |
| Canon research | **Gemini 2.5 Flash** with Google Search grounding |
| T2I — primary | **Nano Banana Pro** (Gemini 2.5 Flash Image) |
| T2I — fallback | gpt-image-2 · Doubao Seedream |
| Video — primary | **Seedance 2.0** via Volcengine Ark |
| Video — manual fallback | Jimeng Web UI packages |
| Storage | plain filesystem (`bible/<characters\|locations\|stories>/...`) |

---

## 📁 Repo layout

```
src/hitchcock/
├── cli.py                # CLI entry point — `hitchcock ...`
├── config.py             # settings / env loader
├── agents/               # one file per pipeline stage
│   ├── brief.py  style.py  cast.py
│   ├── script.py  storyboard.py
│   ├── scene_art.py  shot_gen.py
│   └── design.py  location.py  post.py  tts.py
├── bible/                # story-bible store + Pydantic schemas
├── llm/                  # MiMo + Gemini clients
├── image/                # Nano Banana · gpt-image-2 · Ark Seedream clients
└── video/                # Seedance 2.0 client
scripts/
└── runway_seedance_runner.py   # one-shot A/B runner via Runway's Seedance 2 endpoint
```

---

## 🙏 Acknowledgments

- **Arcane** by Fortiche Studio — primary style reference
- Teams behind **MiMo**, **Gemini**, **Seedance**, **Nano Banana Pro**, and **Jimeng**
