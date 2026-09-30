# FDE: Forward Deployed Engineering Projects

A growing collection of hands-on projects that simulate real Forward Deployed Engineer engagements: take a client's messy real-world problem, build a working solution within their constraints, measure it honestly, and document what I'd ask the client next.

All projects run locally where possible (Ollama, no cloud APIs) and use synthetic data, so they are simulations and should be read as baselines.

## Projects

| # | Project | What it does | Stack | Status |
|---|---|---|---|---|
| 1 | [QuickMart Ticket Cleaner](./quickmart-ticket-cleaner) | Cleans messy support tickets and classifies them with a local LLM, with ground-truth evaluation | Python, pandas, Ollama | Tuning-set results done; holdout pending |

## Conventions for every project

- Its own folder with its own `README.md`, `requirements.txt` and `.gitignore`
- A written client brief, constraints, and limitations section
- Measured results (not just a demo), including failure analysis
- A "questions I'd ask the client" section

## Author

**Abhishek Singh**
