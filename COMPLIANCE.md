# Competition Rules & Submission Compliance

Reviewed against the official Competition Rules for **Kaggle - Autonomous Agent Prediction (Beta)** (Sponsor: Google LLC). Status of the entry produced in this repository:

| **Rule** | **Requirement** | **Status** |
|----------|-----------------|------------|
| Team limits (2.1) | Max team size 5 | OK — single-account entry |
| Multiple accounts | No registering/submitting from multiple accounts | OK — one account only |
| Submission limits (2.2) | Max **5 submissions/day** | Enforced via agent config & skill guard |
| Final submissions (2.2) | Select at most **2** final submissions | Enforced — `select.py` top-2 |
| Data access/use | CC BY 4.0 (Attribution) | OK — only competition-provided data used |
| Data security | Do not transmit/share data outside sandbox | Enforced via agent prompt |
| External data & tools | External data must be public/equal-access; AMLT allowed | OK — no external data fetched; the agent (an AMLT) is permitted |
| Winner's license | None requested | OK |
| Winner's obligations (2.8) | Deliver model source code + docs + environment description, reproducible | Done — this repo is the deliverable |
| Governing law | California law, Santa Clara County courts | N/A to submission |

## Deliverable (Winner's Obligations)
This repository is the reproducible model deliverable:
- `agent.yaml` + `prompts/system.md` — agent definition/system prompt
- `skills/` — data-exploration, model-training, submission-management with runnable scripts
- `CREDITS.md` — authorship

Reproduction: run the `model-training` scripts (e.g. `compete.py`) in a Kaggle Python environment (pandas>=3.0, sklearn, lightgbm, xgboost, catboost, scipy) to regenerate submissions; see each `SKILL.md` for exact commands.

---
© 2026 — Created by **Mayank Sharma**.