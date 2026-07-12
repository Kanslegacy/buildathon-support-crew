# Multi-Agent Customer Support (CrewAI + Streamlit)

Weekly Buildathon submission - Gen AI Architect Program (Social Eagle)

A customer support system made of three agents that run **sequentially** using
**CrewAI**, with a **Streamlit** UI.

## Agents

| # | Agent | Role |
|---|-------|------|
| 1 | Assistant | Answers the user's query directly from its own knowledge |
| 2 | Web Search Assistant | Searches the web and answers from the results |
| 3 | Entry Agent | Writes the query + both answers to `answers.txt` and returns both answers to the UI |

## Project structure

```
buildathon-support-crew/
├── .streamlit/
│   └── config.toml       # light theme config
├── app.py                # single-file app: agents, tasks, crew, Streamlit UI
├── requirements.txt
├── .env.example           # template - copy values into real env vars, not into a file
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. Create project folder (already done if you cloned this repo)
cd buildathon-support-crew

# 2. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API keys as environment variables (never hard-code these)
export OPENAI_API_KEY="your-openai-key"     # macOS/Linux
export SERPER_API_KEY="your-serper-key"

# Windows PowerShell:
# $env:OPENAI_API_KEY="your-openai-key"
# $env:SERPER_API_KEY="your-serper-key"

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How it works

1. User enters a query, e.g. "How do I reset my password?"
2. **Assistant** answers directly from its own knowledge.
3. **Web Search Assistant** searches the web (via Serper) and answers.
4. **Entry Agent** saves the query + both answers to `answers.txt` and the
   Streamlit UI displays both answers.

## Notes

- API keys are read only from environment variables — see `.env.example`.
- `answers.txt` is created/appended in the project root each time you run a query.
- Keep `venv/`, `.env`, and `answers.txt` out of GitHub (see `.gitignore`).
