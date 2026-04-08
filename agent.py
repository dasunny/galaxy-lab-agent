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
import time
from pathlib import Path
from datetime import datetime, timezone

import anthropic
import urllib3
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
gi.verify = False  # Disable SSL verification for self-signed certs
gi.session.verify = False

# ---------------------------------------------------------------------------
# Startup sync — runs once when agent starts
# ---------------------------------------------------------------------------

def load_context() -> dict:
    """Load all local catalogs into memory for injection into system prompt."""
    context = {}

    # Load tool catalog summary
    if CATALOG_PATH.exists():
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        context["tool_count"] = catalog.get("tool_count", 0)
        context["section_count"] = catalog.get("section_count", 0)
        context["sections"] = list(catalog.get("sections", {}).keys())[:20]
    else:
        context["tool_count"] = 0
        context["sections"] = []

    # Load workflow catalog
    if WORKFLOW_CATALOG_PATH.exists():
        context["workflows"] = json.loads(
            WORKFLOW_CATALOG_PATH.read_text(encoding="utf-8")
        )
    else:
        context["workflows"] = {}

    # Load SKILL.md
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
        "description": "Run a single Galaxy tool on an input file. Always confirm with the user before calling this. Use for testing or simple single-step analyses.",
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
]

# ---------------------------------------------------------------------------
# Tool executor
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
    return f"Unknown tool: {tool_name}"

def run_single_tool(tool_id: str, input_path: str, history_name: str) -> str:
    """Upload a file, run a single Galaxy tool, poll for completion, download results."""
    import hashlib
    from datetime import datetime, timezone

    input_file = Path(input_path)
    if not input_file.exists():
        return f"Error: input file not found at {input_path}"

    # Create output directory
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

        # 3. Poll until upload ready
        print("  Waiting for upload...")
        _poll_dataset(history_id, dataset_id)

        # 4. Run tool
        print(f"  Running {tool_id}...")
        inputs = {"input_file": {"src": "hda", "id": dataset_id}}
        result = gi.tools.run_tool(history_id, tool_id, inputs)
        output_ids = [o["id"] for o in result.get("outputs", [])]

        # 5. Poll until outputs ready
        print("  Waiting for job to complete...")
        for oid in output_ids:
            _poll_dataset(history_id, oid)

        # 6. Download outputs
        downloads_dir = run_dir / "outputs"
        downloads_dir.mkdir(exist_ok=True)
        downloaded = []

        for oid in output_ids:
            ds = gi.datasets.show_dataset(oid)
            ext = ds.get("extension", "dat")
            fname = f"{ds.get('name', oid)}.{ext}"
            out_path = downloads_dir / fname
            gi.datasets.download_dataset(
                oid,
                file_path=str(out_path),
                use_default_filename=False
            )
            downloaded.append(fname)
            print(f"  Downloaded: {fname}")

        # 7. Write reproducibility bundle
        _write_repro_bundle(run_dir, tool_id, input_file, downloaded)

        return json.dumps({
            "status": "success",
            "tool_id": tool_id,
            "history": history_name,
            "history_id": history_id,
            "outputs": downloaded,
            "output_dir": str(run_dir),
        }, indent=2)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def _poll_dataset(history_id: str, dataset_id: str, timeout: int = 600):
    """Poll until a dataset is in 'ok' state."""
    import time
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
    """Write reproducibility bundle for a tool run."""
    import hashlib
    from datetime import datetime, timezone

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
    (repro / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")    

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
- For tool discovery questions, use search_tools — do not guess
- For workflow questions, use list_workflows or get_workflow_details
- For history and job status, use list_histories or get_history_details
- Be concise and direct
- Before calling run_single_tool, always confirm with the user and summarize the action clearly
- For run_single_tool, find the correct full tool ID from the catalog first using search_tools
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
    conversation_history = []  # persists across turns

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