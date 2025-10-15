# MLA

This repository contains the implementation of the coursework *“Development of a Multi-Agent System for Generation and Optimization of ML Code.”*

## 🧩 Overview
The project implements a **multi-agent system** designed to automatically generate, analyze, and optimize machine learning code based on textual task descriptions and datasets.

## 🚀 Usage

### 1. Prepare the Environment
To install the package, run:

```bash
pip install .
```

For development or editable installation, use:

```bash
pip install -e .
```

### 2. Run the Agent

You can start the agent using either a description file or a direct text prompt:

```bash
# 1  Set an LLM key
export OPENAI_API_KEY=<your‑key>
# 2 Set an LLM base url compatible with openai api (e.g https://.../v1, for example for ollama compatible api usage)
export OPENAI_BASE_URL=<base-url>
# 3 also for clear ollama api usage set OPENAI_BASE_URL to simple https://ip:port

agent data_dir=<path_to_data_dir> desc_file=<path_to_description_file>
# or
agent data_dir=<path_to_data_dir> desc="<task_description_text>"
```

For additional configuration parameters, see:

```
agent/utils/config.yaml
```
