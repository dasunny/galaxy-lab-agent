#!/usr/bin/env python3
"""
agent.py — ODU Computational Genomics Lab Agent
================================================
AI agent connecting natural language to the lab's Galaxy instance.
Uses Claude as the reasoning engine and BioBlend for Galaxy API calls.

Usage:
    python agent.py
"""

import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone

import anthropic
import urllib3
import requests
from requests.adapters import HTTPAdapter
from bioblend.galaxy import GalaxyInstance
from sync import run_full_sync

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SKILL_DIR = Path(__file__).resolve().parent
CATALOG_PATH = SKILL_DIR / "galaxy_catalog.json"
WORKFLOW_CATALOG_PATH = SKILL_DIR / "workflow_catalog.json"
SKILL_MD_PATH = SKILL_DIR / "SKILL.md"
GALAXY_SKILLS_DIR = SKILL_DIR / "galaxy_skills"
OUTPUT_DIR = SKILL_DIR / "output"

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
gi = GalaxyInstance(
    url=os.environ.get("GALAXY_URL"),
    key=os.environ.get("GALAXY_API_KEY"),
    verify=False
)
gi.verify = False

# Patch all requests sessions to skip SSL verification
original_send = requests.Session.send
def patched_send(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_send(self, *args, **kwargs)
requests.Session.send = patched_send

# ---------------------------------------------------------------------------
# Startup sync
# ---------------------------------------------------------------------------

def load_context() -> dict:
    """Load all local catalogs into memory for injection into system prompt."""
    context = {}

    if CATALOG_PATH.exists():
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        context["tool_count"] = catalog.get("tool_count", 0)
        context["section_count"] = catalog.get("section_count", 0)
        context["sections"] = list(catalog.get("sections", {}).keys())[:20]
    else:
        context["tool_count"] = 0
        context["sections"] = []

    if WORKFLOW_CATALOG_PATH.exists():
        context["workflows"] = json.loads(
            WORKFLOW_CATALOG_PATH.read_text(encoding="utf-8")
        )
    else:
        context["workflows"] = {}

    if SKILL_MD_PATH.exists():
        context["skill_md"] = SKILL_MD_PATH.read_text(encoding="utf-8")
    else:
        context["skill_md"] = ""

    return context


def startup_sync() -> dict:
    """Run on agent start — sync Galaxy state, return loaded context."""
    print("Syncing with Galaxy server...")
    sync_data = run_full_sync(gi)
    context = load_context()
    context.update(sync_data)
    print(f"  Tools available: {context['tool_count']}")
    print(f"  Workflows loaded: {len(context['workflows'])}")
    print("Ready.\n")
    return context


# ---------------------------------------------------------------------------
# Tools Claude can call
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "search_tools",
        "description": "Search the local Galaxy tool catalog by keyword. Use this ONCE per user request with the most relevant search term. Do not call multiple times for the same request. Does not hit the Galaxy API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term, e.g. 'peak calling', 'alignment', 'quality control'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_workflows",
        "description": "List all workflows available on the Galaxy instance from the local workflow catalog.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_workflow_details",
        "description": "Get detailed information about a specific workflow by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "The Galaxy workflow ID"
                }
            },
            "required": ["workflow_id"]
        }
    },
    {
        "name": "list_histories",
        "description": "Fetch recent analysis histories from the Galaxy server. Makes a live API call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of histories to return (default: 10)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_history_details",
        "description": "Get the datasets and job status inside a specific history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "history_id": {
                    "type": "string",
                    "description": "The Galaxy history ID"
                }
            },
            "required": ["history_id"]
        }
    },
    {
        "name": "run_single_tool",
        "description": "Upload a file and submit a single Galaxy tool job. Submits immediately and returns — does not wait for completion. Always confirm with the user before calling this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tool_id": {
                    "type": "string",
                    "description": "The full Galaxy tool ID"
                },
                "input_path": {
                    "type": "string",
                    "description": "Local path to the input file"
                },
                "history_name": {
                    "type": "string",
                    "description": "Name for the new Galaxy history"
                }
            },
            "required": ["tool_id", "input_path", "history_name"]
        }
    },
    {
        "name": "list_active_jobs",
        "description": "Get a lightweight overview of recent Galaxy histories and their status. Use when the user asks what's running or how jobs are doing. Does not return full dataset details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recent histories to check (default: 10)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_job_details",
        "description": "Get detailed status of a specific Galaxy history — all datasets, states, errors. Use only when the user asks about a specific job or history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "history_id": {
                    "type": "string",
                    "description": "The Galaxy history ID to inspect"
                }
            },
            "required": ["history_id"]
        }
    },
]

# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def search_tools(query: str) -> str:
    """Search local catalog by keyword."""
    if not CATALOG_PATH.exists():
        return "Tool catalog not found. Run startup sync first."

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    query_lower = query.lower()
    results = []

    for tool in catalog.get("tools", []):
        name = (tool.get("name") or "").lower()
        desc = (tool.get("description") or "").lower()
        section = (tool.get("section") or "").lower()

        if query_lower in name or query_lower in desc or query_lower in section:
            results.append({
                "name": tool.get("name"),
                "id": tool.get("id"),
                "description": tool.get("description", "")[:100],
                "section": tool.get("section"),
            })

    if not results:
        return f"No tools found matching '{query}'"

    return json.dumps(results[:15], indent=2)


def list_workflows(context: dict) -> str:
    """Return workflow list from local catalog."""
    workflows = context.get("workflows", {})
    if not workflows:
        return "No workflows found in catalog."
    return json.dumps(list(workflows.values()), indent=2)


def get_workflow_details(workflow_id: str, context: dict) -> str:
    """Return details for a specific workflow."""
    workflows = context.get("workflows", {})
    if workflow_id in workflows:
        return json.dumps(workflows[workflow_id], indent=2)
    return f"Workflow {workflow_id} not found in catalog."


def list_histories(limit: int = 10) -> str:
    """Fetch recent histories from Galaxy — live API call."""
    histories = gi.histories.get_histories()[:limit]
    return json.dumps([
        {"name": h["name"], "id": h["id"]}
        for h in histories
    ], indent=2)


def get_history_details(history_id: str) -> str:
    """Fetch datasets inside a history — live API call."""
    datasets = gi.histories.show_history(history_id, contents=True)
    summary = []
    for ds in datasets:
        summary.append({
            "name": ds.get("name"),
            "state": ds.get("state"),
            "type": ds.get("history_content_type"),
        })
    return json.dumps(summary, indent=2)


def run_single_tool(tool_id: str, input_path: str, history_name: str) -> str:
    """Upload a file and submit a Galaxy tool job. Returns immediately after submission."""
    input_file = Path(input_path)
    if not input_file.exists():
        return json.dumps({
            "status": "error",
            "error": f"Input file not found: {input_path}"
        })

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{tool_id.split('/')[-2] if '/' in tool_id else tool_id}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Create history
        print(f"  Creating history: {history_name}")
        history = gi.histories.create_history(name=history_name)
        history_id = history["id"]

        # 2. Upload input
        print(f"  Uploading {input_file.name}...")
        upload = gi.tools.upload_file(str(input_file), history_id)
        dataset_id = upload["outputs"][0]["id"]

        # 3. Poll until upload ready only
        print("  Waiting for upload...")
        _poll_dataset(history_id, dataset_id, timeout=120)

        # 4. Submit tool — do not poll for completion
        print(f"  Submitting {tool_id}...")
        inputs = {"input_file": {"src": "hda", "id": dataset_id}}
        result = gi.tools.run_tool(history_id, tool_id, inputs)
        job_ids = [j["id"] for j in result.get("jobs", [])]
        output_ids = [o["id"] for o in result.get("outputs", [])]

        # 5. Save job tracking info locally
        job_info = {
            "history_id": history_id,
            "history_name": history_name,
            "tool_id": tool_id,
            "job_ids": job_ids,
            "output_ids": output_ids,
            "run_dir": str(run_dir),
            "input_file": str(input_file),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "status": "running"
        }
        (run_dir / "job_info.json").write_text(
            json.dumps(job_info, indent=2), encoding="utf-8"
        )

        return json.dumps({
            "status": "submitted",
            "message": "Job submitted successfully.",
            "history_name": history_name,
            "history_id": history_id,
            "job_ids": job_ids,
            "run_dir": str(run_dir),
            "tip": "Ask 'check job status' anytime to see progress."
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def list_active_jobs(limit: int = 10) -> str:
    """Lightweight overview of recent histories and their states."""
    try:
        histories = gi.histories.get_histories()[:limit]
        summary = []

        for h in histories:
            state_details = h.get("state_details", {})
            total = sum(state_details.values()) if state_details else 0
            running = state_details.get("running", 0)
            queued = state_details.get("queued", 0)
            ok = state_details.get("ok", 0)
            error = state_details.get("error", 0)

            if error > 0:
                overall = "error"
            elif running > 0:
                overall = "running"
            elif queued > 0:
                overall = "queued"
            elif ok == total and total > 0:
                overall = "complete"
            else:
                overall = "empty"

            summary.append({
                "name": h["name"],
                "id": h["id"],
                "status": overall,
                "running": running,
                "queued": queued,
                "ok": ok,
                "error": error,
            })

        return json.dumps(summary, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def get_job_details(history_id: str) -> str:
    """Full dataset-level detail for a specific history."""
    try:
        history = gi.histories.show_history(history_id)
        datasets = gi.histories.show_history(history_id, contents=True)

        detail = []
        for ds in datasets:
            detail.append({
                "name": ds.get("name"),
                "state": ds.get("state"),
                "type": ds.get("history_content_type"),
                "id": ds.get("id"),
            })

        states = [d["state"] for d in detail if d["type"] == "dataset"]
        if all(s == "ok" for s in states) and states:
            overall = "complete"
        elif any(s == "error" for s in states):
            overall = "error"
        elif any(s == "running" for s in states):
            overall = "running"
        elif any(s in ("queued", "new") for s in states):
            overall = "queued"
        else:
            overall = "unknown"

        return json.dumps({
            "name": history.get("name"),
            "id": history_id,
            "overall_status": overall,
            "datasets": detail
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _poll_dataset(history_id: str, dataset_id: str, timeout: int = 600):
    """Poll until a dataset is in 'ok' state."""
    start = time.time()
    while time.time() - start < timeout:
        ds = gi.datasets.show_dataset(dataset_id)
        state = ds.get("state", "")
        if state == "ok":
            return
        if state in ("error", "discarded", "failed_metadata"):
            raise RuntimeError(f"Dataset {dataset_id} failed with state: {state}")
        time.sleep(3)
    raise TimeoutError(f"Dataset {dataset_id} timed out after {timeout}s")


def _write_repro_bundle(run_dir: Path, tool_id: str, input_file: Path, outputs: list):
    """Write reproducibility bundle for a completed tool run."""
    repro = run_dir / "reproducibility"
    repro.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()

    (repro / "commands.sh").write_text(
        f"#!/usr/bin/env bash\n"
        f"# Tool: {tool_id}\n"
        f"# Date: {ts}\n"
        f"# Galaxy: {gi.base_url}\n\n"
        f"python agent.py\n"
        f"# Then ask: run {tool_id} on {input_file.name}\n",
        encoding="utf-8"
    )

    (repro / "environment.yml").write_text(
        f"galaxy_url: {gi.base_url}\n"
        f"tool_id: {tool_id}\n"
        f"date: {ts}\n",
        encoding="utf-8"
    )

    lines = []
    for fp in [input_file] + [run_dir / "outputs" / o for o in outputs]:
        p = Path(fp)
        if p.exists():
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{sha}  {p.name}")
    (repro / "checksums.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, tool_input: dict, context: dict) -> str:
    """Route tool calls to the right function."""
    if tool_name == "search_tools":
        return search_tools(tool_input["query"])
    elif tool_name == "list_workflows":
        return list_workflows(context)
    elif tool_name == "get_workflow_details":
        return get_workflow_details(tool_input["workflow_id"], context)
    elif tool_name == "list_histories":
        return list_histories(tool_input.get("limit", 10))
    elif tool_name == "get_history_details":
        return get_history_details(tool_input["history_id"])
    elif tool_name == "run_single_tool":
        return run_single_tool(
            tool_input["tool_id"],
            tool_input["input_path"],
            tool_input["history_name"]
        )
    elif tool_name == "list_active_jobs":
        return list_active_jobs(tool_input.get("limit", 10))
    elif tool_name == "get_job_details":
        return get_job_details(tool_input["history_id"])
    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(context: dict) -> str:
    """Build the system prompt from SKILL.md and loaded context."""
    workflow_summary = "\n".join([
        f"  - {wf['name']} (ID: {wf['id']}, steps: {wf['steps']})"
        for wf in context.get("workflows", {}).values()
    ])

    return f"""
{context.get("skill_md", "")}

## Current Server State (loaded at startup)

- Tools indexed: {context.get("tool_count", 0)} across {context.get("section_count", 0)} categories
- Available workflows:
{workflow_summary or "  None found"}

## Instructions
- For tool discovery questions, use search_tools — do not guess, do not call more than once per request
- For workflow questions, use list_workflows or get_workflow_details
- For history browsing, use list_histories or get_history_details
- For job monitoring, use list_active_jobs for general status, get_job_details only when asked about a specific job
- Before calling run_single_tool, always confirm with the user and summarize: tool name, input file, history name
- After submitting a job, immediately tell the user it is running, show the history name and ID, and remind them they can ask for status anytime — do not wait for completion
- Never fetch full job details for all jobs unless the user specifically asks
- Be concise and direct
- When you return tool results, summarize them clearly for a lab member
""".strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_message: str, context: dict, history: list) -> list:
    """Single turn of the agent loop. Takes and returns conversation history."""
    print(f"\nUser: {user_message}")
    print("-" * 40)

    history.append({"role": "user", "content": user_message})
    system_prompt = build_system_prompt(context)

    while True:
        response = claude.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=history
        )

        if response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [calling: {block.name}]")
                    result = execute_tool(block.name, block.input, context)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            history.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAssistant: {block.text}")
                    history.append({
                        "role": "assistant",
                        "content": block.text
                    })
            break

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    context = startup_sync()
    conversation_history = []

    print("ODU Genomics Lab Agent — type 'quit' to exit")
    print("=" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        conversation_history = run_agent(user_input, context, conversation_history)