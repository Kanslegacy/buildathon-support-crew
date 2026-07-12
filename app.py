"""
Multi-Agent Customer Support System
Social Eagle - Gen AI Architect Program | Weekly Buildathon

Three agents run sequentially using CrewAI:
  1. Assistant            -> answers the query directly from its own knowledge
  2. Web Search Assistant -> searches the web and answers using the results
  3. Entry Agent           -> writes the query + both answers to a .txt file
                              and returns both answers for the UI

UI: Streamlit
API keys: read from environment variables only (never hard-coded).
"""

import os
import time
import datetime

import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai_tools import SerperDevTool


# --------------------------------------------------------------------------- #
# 1. Environment / API key setup
# --------------------------------------------------------------------------- #
# All keys are read from environment variables. Set them before running:
#   export OPENAI_API_KEY="..."
#   export SERPER_API_KEY="..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

OUTPUT_FILE = "answers.txt"

EXAMPLE_QUERIES = [
    "How do I reset a forgotten account password?",
    "What is the latest stable version of Python?",
    "How do I fix an HTTP 429 'Too Many Requests' error?",
]


# --------------------------------------------------------------------------- #
# 2. Custom tool for the Entry Agent
# --------------------------------------------------------------------------- #
@tool("Save Answers To File")
def save_answers_to_file(query: str, assistant_answer: str, web_search_answer: str) -> str:
    """
    Saves the user's query along with the Assistant's answer and the
    Web Search Assistant's answer to a local .txt file (answers.txt).

    Args:
        query: The original user query.
        assistant_answer: The answer produced by the Assistant agent.
        web_search_answer: The answer produced by the Web Search Assistant agent.

    Returns:
        A confirmation string once the file has been written.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"{'=' * 60}\n"
        f"Timestamp: {timestamp}\n"
        f"Query: {query}\n\n"
        f"[Assistant Answer]\n{assistant_answer}\n\n"
        f"[Web Search Assistant Answer]\n{web_search_answer}\n"
        f"{'=' * 60}\n\n"
    )
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

    return f"Saved query and both answers to {OUTPUT_FILE}"


# --------------------------------------------------------------------------- #
# 3. Agents
# --------------------------------------------------------------------------- #
def build_agents():
    assistant = Agent(
        role="Assistant",
        goal="Answer the user's query directly and accurately using your own knowledge.",
        backstory=(
            "You are a knowledgeable customer support assistant who answers "
            "questions clearly and concisely using what you already know, "
            "without needing to search the web."
        ),
        tools=[],
        verbose=True,
        allow_delegation=False,
    )

    web_search_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Search the web for up-to-date, relevant information about the "
            "user's query and produce a well-grounded answer from the results."
        ),
        backstory=(
            "You are a research-savvy support agent who verifies and enriches "
            "answers by searching the internet, then summarizing the most "
            "relevant findings for the customer."
        ),
        tools=[SerperDevTool()],
        verbose=True,
        allow_delegation=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Record the user's query and both answers into a text file, then "
            "present both answers back clearly."
        ),
        backstory=(
            "You are responsible for logging every support interaction. You "
            "always save the full record to disk before reporting the final "
            "answers back to the user."
        ),
        tools=[save_answers_to_file],
        verbose=True,
        allow_delegation=False,
    )

    return assistant, web_search_assistant, entry_agent


# --------------------------------------------------------------------------- #
# 4. Tasks
# --------------------------------------------------------------------------- #
def build_tasks(query: str, assistant: Agent, web_search_assistant: Agent, entry_agent: Agent,
                 on_step=None):
    """
    on_step: optional callable(step_label: str) invoked whenever a task
    finishes, used to drive live progress updates in the UI.
    """

    def _wrap(label):
        if on_step is None:
            return None

        def _cb(output):
            on_step(label)

        return _cb

    task_assistant = Task(
        description=(
            f"Answer the following customer query directly, using your own "
            f"knowledge only (do not search the web):\n\nQuery: {query}"
        ),
        expected_output="A clear, direct answer to the user's query.",
        agent=assistant,
        callback=_wrap("Assistant"),
    )

    task_web_search = Task(
        description=(
            f"Search the web for information relevant to the following "
            f"customer query, then write an answer grounded in what you "
            f"find:\n\nQuery: {query}"
        ),
        expected_output="A clear answer to the user's query, grounded in web search results.",
        agent=web_search_assistant,
        callback=_wrap("Web Search Assistant"),
    )

    task_entry = Task(
        description=(
            f"You have been given the original user query and two answers: "
            f"one from the Assistant and one from the Web Search Assistant. "
            f"Original query: {query}\n\n"
            "Use the 'Save Answers To File' tool to save the query, the "
            "Assistant's answer, and the Web Search Assistant's answer to "
            "the answers.txt file. Then, in your final answer, clearly "
            "present both answers labelled as 'Assistant Answer' and "
            "'Web Search Answer'."
        ),
        expected_output=(
            "Confirmation the file was saved, followed by both answers "
            "clearly labelled."
        ),
        agent=entry_agent,
        context=[task_assistant, task_web_search],
        callback=_wrap("Entry Agent"),
    )

    return task_assistant, task_web_search, task_entry


# --------------------------------------------------------------------------- #
# 5. Crew runner
# --------------------------------------------------------------------------- #
def run_crew(query: str, on_step=None):
    assistant, web_search_assistant, entry_agent = build_agents()
    task_assistant, task_web_search, task_entry = build_tasks(
        query, assistant, web_search_assistant, entry_agent, on_step=on_step
    )

    crew = Crew(
        agents=[assistant, web_search_assistant, entry_agent],
        tasks=[task_assistant, task_web_search, task_entry],
        process=Process.sequential,
        verbose=True,
    )

    crew_output = crew.kickoff()

    # Pull each task's individual output so the UI can show both answers
    # separately (not just the Entry Agent's final combined text).
    assistant_answer = crew_output.tasks_output[0].raw
    web_search_answer = crew_output.tasks_output[1].raw
    entry_summary = crew_output.tasks_output[2].raw

    return assistant_answer, web_search_answer, entry_summary


# --------------------------------------------------------------------------- #
# 6. Styling
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}

    .se-hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #A78BFA 0%, #F0ABFC 50%, #FBCFE8 100%);
        color: #3B0764;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.15);
    }
    .se-hero h1 {
        margin: 0 0 0.3rem 0;
        font-size: 2rem;
        font-weight: 800;
        color: #3B0764;
    }
    .se-hero p {
        margin: 0;
        opacity: 0.85;
        font-size: 1.02rem;
        color: #4C1D95;
    }
    .se-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.55);
        color: #3B0764;
        font-size: 0.78rem;
        margin-top: 0.7rem;
        margin-right: 0.4rem;
        border: 1px solid rgba(124,58,237,0.25);
    }

    .se-card {
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        border: 1px solid rgba(124,58,237,0.12);
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(124, 58, 237, 0.06);
        height: 100%;
    }
    .se-card h4 {
        margin-top: 0;
        margin-bottom: 0.6rem;
        color: #4C1D95;
    }

    .se-status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .se-ok { background-color: #16A34A; }
    .se-missing { background-color: #DC2626; }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(124,58,237,0.12);
    }
</style>
"""


def key_status_row(label: str, is_set: bool) -> str:
    dot = "se-ok" if is_set else "se-missing"
    text = "connected" if is_set else "missing"
    return f'<span class="se-status-dot {dot}"></span>**{label}** — {text}'


# --------------------------------------------------------------------------- #
# 7. Streamlit UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(
        page_title="Multi-Agent Customer Support",
        page_icon="🛟",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    missing_keys = [
        name for name, val in
        [("OPENAI_API_KEY", OPENAI_API_KEY), ("SERPER_API_KEY", SERPER_API_KEY)]
        if not val
    ]

    # ------------------------- Sidebar ------------------------- #
    with st.sidebar:
        st.markdown("### 🔗 Agent Pipeline")
        st.markdown(
            "1. 🧠 **Assistant** — answers from its own knowledge\n"
            "2. 🌐 **Web Search Assistant** — answers from live web search\n"
            "3. 📝 **Entry Agent** — logs both answers to `answers.txt`"
        )
        st.caption("Runs sequentially via CrewAI's `Process.sequential`.")

        st.divider()
        st.markdown("### 🔑 API Keys")
        st.markdown(key_status_row("OPENAI_API_KEY", bool(OPENAI_API_KEY)), unsafe_allow_html=True)
        st.markdown(key_status_row("SERPER_API_KEY", bool(SERPER_API_KEY)), unsafe_allow_html=True)
        if missing_keys:
            st.caption("Set missing keys as environment variables, then restart the app.")

        st.divider()
        st.markdown("### 📁 Session Log")
        st.caption(f"{len(st.session_state.history)} quer{'y' if len(st.session_state.history) == 1 else 'ies'} this session")

        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                file_bytes = f.read()
            st.download_button(
                "⬇️ Download answers.txt",
                data=file_bytes,
                file_name="answers.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.session_state.history:
            if st.button("🗑️ Clear session history", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        st.divider()
        st.caption("Social Eagle · Gen AI Architect Program\nCrewAI + Streamlit")

    # ------------------------- Hero header ------------------------- #
    st.markdown(
        """
        <div class="se-hero">
            <h1>🛟 Multi-Agent Customer Support</h1>
            <p>Ask a question and watch three CrewAI agents collaborate, in sequence, to answer it.</p>
            <span class="se-pill">⚙️ CrewAI · Sequential</span>
            <span class="se-pill">🎈 Streamlit UI</span>
            <span class="se-pill">🌐 Live Web Search</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if missing_keys:
        st.warning(
            "Missing environment variable(s): "
            f"{', '.join(missing_keys)}. Set them before running the app "
            "(see README / .env.example)."
        )

    # ------------------------- Query input ------------------------- #
    st.markdown("#### 💬 Ask a question")

    def _set_query(value: str):
        # Runs BEFORE the script reruns, i.e. before the text_input widget
        # below is re-instantiated for this run - safe to set here.
        st.session_state.query_input = value

    chip_cols = st.columns(len(EXAMPLE_QUERIES) + 1)
    chip_cols[0].caption("Try an example:")
    for i, example in enumerate(EXAMPLE_QUERIES):
        chip_cols[i + 1].button(
            example,
            key=f"example_{i}",
            use_container_width=True,
            on_click=_set_query,
            args=(example,),
        )

    query = st.text_input(
        "Enter your query or task",
        key="query_input",
        placeholder="e.g. How do I reset my password?",
        label_visibility="collapsed",
    )

    run_clicked = st.button("🚀 Get Answer", type="primary", disabled=bool(missing_keys))

    # ------------------------- Run crew ------------------------- #
    if run_clicked:
        if not query.strip():
            st.error("Please enter a query first.")
            return

        progress_steps = {"Assistant": "⏳", "Web Search Assistant": "⏳", "Entry Agent": "⏳"}
        start_time = time.time()

        with st.status("Running the multi-agent crew...", expanded=True) as status:
            step_placeholder = st.empty()

            def render_steps():
                lines = [f"{icon} {name}" for name, icon in progress_steps.items()]
                step_placeholder.markdown("  \n".join(lines))

            render_steps()

            def on_step(label):
                progress_steps[label] = "✅"
                render_steps()

            try:
                assistant_answer, web_search_answer, entry_summary = run_crew(
                    query, on_step=on_step
                )
            except Exception as e:
                status.update(label="Crew failed", state="error")
                st.error(f"Something went wrong while running the crew: {e}")
                return

            elapsed = time.time() - start_time
            status.update(label=f"Crew finished in {elapsed:.1f}s", state="complete")

        st.session_state.history.insert(
            0,
            {
                "query": query,
                "assistant_answer": assistant_answer,
                "web_search_answer": web_search_answer,
                "entry_summary": entry_summary,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed": elapsed,
            },
        )
        st.rerun()

    # ------------------------- Latest result ------------------------- #
    if st.session_state.history:
        latest = st.session_state.history[0]

        st.success(
            f"Answered in {latest['elapsed']:.1f}s · saved to `{OUTPUT_FILE}` · {latest['timestamp']}"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Assistant answer length", f"{len(latest['assistant_answer'].split())} words")
        m2.metric("Web search answer length", f"{len(latest['web_search_answer'].split())} words")
        m3.metric("Response time", f"{latest['elapsed']:.1f}s")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div class="se-card"><h4>🧠 Assistant Answer</h4>{latest['assistant_answer']}</div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="se-card"><h4>🌐 Web Search Assistant Answer</h4>{latest['web_search_answer']}</div>""",
                unsafe_allow_html=True,
            )

        with st.expander("📝 Entry Agent summary / file confirmation"):
            st.write(latest["entry_summary"])

        # ------------------------- Past queries ------------------------- #
        if len(st.session_state.history) > 1:
            st.markdown("#### 🕘 Previous queries this session")
            for item in st.session_state.history[1:]:
                with st.expander(f"{item['timestamp']} — {item['query']}"):
                    st.markdown("**🧠 Assistant:** " + item["assistant_answer"])
                    st.markdown("**🌐 Web Search:** " + item["web_search_answer"])
    else:
        st.info("Enter a query above and click **Get Answer** to see the crew in action.")


if __name__ == "__main__":
    main()