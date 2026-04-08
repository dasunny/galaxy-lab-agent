#!/usr/bin/env python3
"""
sync.py — Startup sync for ODU Galaxy Lab Agent
================================================
Generates local catalogs from the live Galaxy server at startup.
Run once per session — results are cached locally for fast lookup.

Functions:
    sync_galaxy_catalog()     — fetch all tools → galaxy_catalog.json
    sync_curated_skills()     — generate galaxy_skills/ markdown files
    sync_workflow_catalog()   — fetch workflows → workflow_catalog.json
    run_full_sync()           — run all three in order
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

AGENT_DIR = Path(__file__).resolve().parent
CATALOG_PATH = AGENT_DIR / "galaxy_catalog.json"
WORKFLOW_CATALOG_PATH = AGENT_DIR / "workflow_catalog.json"
GALAXY_SKILLS_DIR = AGENT_DIR / "galaxy_skills"

# ---------------------------------------------------------------------------
# Priority tools to always include in curated skills
# ---------------------------------------------------------------------------

PRIORITY_TOOLS = {
    "fastqc", "trimmomatic", "cutadapt", "fastp", "multiqc",
    "bwa", "bowtie2", "hisat2", "minimap2", "star",
    "samtools", "picard", "bamtools", "bedtools",
    "freebayes", "gatk4", "bcftools",
    "featurecounts", "htseq_count", "deseq2", "edger", "stringtie", "salmon",
    "kraken2", "metaphlan", "bracken",
    "spades", "flye", "megahit",
    "prokka", "augustus",
    "macs2", "deeptools", "diffbind",
    "scanpy", "cellranger",
    "nanoplot", "medaka",
}

# ---------------------------------------------------------------------------
# Galaxy catalog
# ---------------------------------------------------------------------------

def sync_galaxy_catalog(gi) -> dict:
    """Fetch all tools from the Galaxy API and write galaxy_catalog.json."""
    print("  Fetching tool catalog from Galaxy...")

    raw_tools = gi.tools.get_tools()
    print(f"  Received {len(raw_tools)} raw tool entries")

    tools = []
    seen_ids = set()

    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        tool_id = t.get("id", "")
        if not tool_id or tool_id in seen_ids:
            continue
        if tool_id.startswith("__") or tool_id == "upload1":
            continue

        seen_ids.add(tool_id)
        tools.append({
            "id": tool_id,
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "version": t.get("version", ""),
            "section": t.get("panel_section_name", ""),
            "edam_topics": t.get("edam_topics", []) or [],
            "edam_operations": t.get("edam_operations", []) or [],
        })

    sections: dict[str, int] = {}
    for t in tools:
        sec = t.get("section") or "Uncategorized"
        sections[sec] = sections.get(sec, 0) + 1

    catalog = {
        "version": "1.0.0",
        "galaxy_url": str(gi.base_url),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool_count": len(tools),
        "section_count": len(sections),
        "sections": dict(sorted(sections.items(), key=lambda x: -x[1])),
        "tools": tools,
    }

    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Catalog saved — {len(tools)} tools, {len(sections)} categories")
    return catalog


# ---------------------------------------------------------------------------
# Curated skill profiles
# ---------------------------------------------------------------------------

def _tool_slug(tool_id: str) -> str:
    """Extract short tool name from a Galaxy tool ID."""
    parts = tool_id.strip("/").split("/")
    if len(parts) >= 2:
        return parts[-2].lower()
    return parts[-1].lower()


def _slugify(name: str) -> str:
    """Convert tool name to filename-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-") or "unnamed"


def _write_tool_profile(tool: dict, output_dir: Path):
    """Write a markdown profile for a single Galaxy tool."""
    name = tool.get("name", "Unknown")
    tool_id = tool.get("id", "")
    version = tool.get("version", "?")
    desc = tool.get("description", "")
    section = tool.get("section", "")

    slug = _slugify(name)
    path = output_dir / f"{slug}.md"

    lines = [
        f"# {name}",
        "",
        f"**Galaxy Tool ID**: `{tool_id}`",
        f"**Version**: {version}",
        f"**Category**: {section}",
        "",
    ]

    if desc:
        lines.extend([f"> {desc}", ""])

    lines.extend([
        "## Example Query",
        "",
        f'> "Run {name} on my data"',
        "",
        "## Run via Agent",
        "",
        "```",
        f"search for {name.lower()}",
        "```",
        "",
        "---",
        f"*Generated from {tool.get('galaxy_url', 'ODU Galaxy instance')}*",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_curated_skills(catalog: dict):
    """Generate markdown profiles for priority tools into galaxy_skills/."""
    print("  Generating curated skill profiles...")

    GALAXY_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    tools = catalog.get("tools", [])
    selected = []
    selected_slugs = set()

    # Priority tools first
    for tool in tools:
        slug = _tool_slug(tool.get("id", ""))
        if slug in PRIORITY_TOOLS and slug not in selected_slugs:
            selected.append(tool)
            selected_slugs.add(slug)

    # Fill remaining up to 200 from catalog
    for tool in tools:
        if len(selected) >= 200:
            break
        slug = _tool_slug(tool.get("id", ""))
        if slug not in selected_slugs:
            selected.append(tool)
            selected_slugs.add(slug)

    for tool in selected:
        _write_tool_profile(tool, GALAXY_SKILLS_DIR)

    # Write index
    index_lines = [
        "# Galaxy Skills Index",
        "",
        f"**{len(selected)} tools** profiled from ODU Galaxy instance.",
        "",
    ]
    for tool in sorted(selected, key=lambda t: t.get("name", "")):
        name = tool.get("name", "?")
        slug = _slugify(name)
        section = tool.get("section", "")
        index_lines.append(f"- [{name}]({slug}.md) — {section}")

    (GALAXY_SKILLS_DIR / "INDEX.md").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )

    print(f"  Generated {len(selected)} skill profiles → galaxy_skills/")


# ---------------------------------------------------------------------------
# Workflow catalog
# ---------------------------------------------------------------------------

def sync_workflow_catalog(gi) -> dict:
    """Fetch workflows from Galaxy and write workflow_catalog.json."""
    print("  Syncing workflows...")

    workflows = gi.workflows.get_workflows()
    catalog = {}

    for wf in workflows:
        details = gi.workflows.show_workflow(wf["id"])
        catalog[wf["id"]] = {
            "name": details["name"],
            "id": details["id"],
            "steps": len(details["steps"]),
            "owner": details.get("owner", ""),
            "inputs": details.get("inputs", {}),
        }

    WORKFLOW_CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2), encoding="utf-8"
    )
    print(f"  Workflows saved — {len(catalog)} found")
    return catalog


# ---------------------------------------------------------------------------
# Full sync
# ---------------------------------------------------------------------------

def run_full_sync(gi) -> dict:
    """Run all sync steps. Returns loaded context dict."""
    start = time.time()

    catalog = sync_galaxy_catalog(gi)
    sync_curated_skills(catalog)
    workflow_catalog = sync_workflow_catalog(gi)

    elapsed = round(time.time() - start, 1)
    print(f"  Sync complete in {elapsed}s")

    return {
        "tool_count": catalog.get("tool_count", 0),
        "section_count": catalog.get("section_count", 0),
        "sections": list(catalog.get("sections", {}).keys()),
        "workflows": workflow_catalog,
    }