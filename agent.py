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
        "description": "Search the local Galaxy tool catalog by keyword. Use this for any question about what tools are available. Does not hit the Galaxy API.",
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
- For tool discovery questions, use search_tools — do not guess
- For workflow questions, use list_workflows or get_workflow_details
- For history and job status, use list_histories or get_history_details
- Be concise and direct
- When you return tool results, summarize them clearly for a lab member
""".strip()

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_message: str, context: dict):
    """Single turn of the agent loop."""
    print(f"\nUser: {user_message}")
    print("-" * 40)

    messages = [{"role": "user", "content": user_message}]
    system_prompt = build_system_prompt(context)

    while True:
        response = claude.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

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

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAssistant: {block.text}")
            break

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    context = startup_sync()

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

        run_agent(user_input, context)