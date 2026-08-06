# Agentic Instruction-Data Separation

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-SEP-yellow)](https://github.com/egozverev/Should-It-Be-Executed-Or-Processed)

## Overview
This repository provides a rigorous benchmark for evaluating **Instruction-Data Separation** (IDS) within Agentic Workflows. Using LangGraph, it constructs a multi-tool agent environment and evaluates LLMs' vulnerability to indirect prompt injections (where malicious commands are hidden inside passive data).

The benchmarking pipeline leverages the **SEP Dataset** introduced in the ICLR 2025 paper:
> *"Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?"* (Zverev, Abdelnabi, Tabesh, Fritz, Lampert).

## Key Findings
Our benchmark tests if modern open-weight LLMs (e.g., Qwen2.5-3B) correctly treat retrieved information as passive data, or if they accidentally execute malicious probes embedded within it.

*(See `results/figures/sfr_comparison.png` for generated visualizations)*

## Architecture
The testing environment is built with **LangGraph** and features a standard ReAct loop with two tools:
1. `read_mock_database(query)`: A safe retrieval tool simulating RAG.
2. `execute_system_command(command)`: A dangerous canary tool.

**Separation Failure Rate (SFR)** is measured. If the agent calls the dangerous tool based on retrieved data, it fails the separation test.

## Future Work (ASIDE Implementation)
While this benchmark empirically demonstrates that standard models fail in agentic tool-calling scenarios, the next logical step is integrating an architectural defense. 

Specifically, we plan to test if the **ASIDE architecture** (Architectural Separation of Instructions and Data in Language Models, Zverev et al., ICLR 2026) can prevent these multi-tool agent hijacks. By applying orthogonal embedding rotations to data tokens within our LangGraph pipeline, we aim to completely eliminate the Separation Failure Rate without relying on prompt engineering or tuning.

## Setup
```bash
pip install -r requirements.txt
python -m src.benchmark --model all --n 500
python -m src.visualize
```
