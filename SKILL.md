---
name: galaxy-lab-agent
description: "AI agent for Dr. Sun's Computational Genomics Lab at ODU — Galaxy workflow discovery, execution, and monitoring"
version: 0.1.0
author: Dawson Fromm
license: MIT
---

# ODU Computational Genomics Lab Agent

## Context
This agent connects to the lab's Galaxy instance running on ODU's HPC cluster at
jsun-compute-0.cs.odu.edu. It supports any bioinformatics workflow the lab runs —
genomics, epigenomics, transcriptomics, or otherwise.

## What This Agent Can Do
1. **Discovery** — answer questions about available tools and workflows from local catalog
2. **Execution** — invoke Galaxy workflows via BioBlend
3. **Monitoring** — check job status and history
4. **Reporting** — generate reproducible output bundles for every run

## What Requires a Live API Call
- Invoking a workflow
- Checking job or dataset status
- Fetching history details
- Anything time-sensitive or that may have changed since last sync

## What Does NOT Require a Live API Call
- Tool discovery and search (use galaxy_catalog.json)
- Workflow structure questions (use workflow_catalog.json)
- Tool parameter details (use galaxy_skills/)
- General methodology questions

## Lab Infrastructure
- Galaxy instance: jsun-compute-0.cs.odu.edu
- HPC cluster: ODU research computing
- Galaxy version: 24.2

## Principles
- Never guess tool IDs — always use the catalog
- Always generate a reproducibility bundle for executions
- Prefer local catalog lookups over live API calls where possible