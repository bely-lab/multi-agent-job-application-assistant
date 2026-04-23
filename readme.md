# Multi-Agent Job Application Assistant

This project demonstrates two AI agent frameworks:

## Task 10-1: Smolagents
Single-agent workflow with a custom Python tool for CV keyword analysis.

Features:
- Match CV skills with job keywords
- Detect missing skills
- Generate recruiter-style feedback

## Task 10-2: CrewAI
Multi-agent workflow using sequential task execution.

Agents:
1. Job Requirements Researcher
2. CV Reviewer
3. Cover Letter Writer

Features:
- Extract job requirements
- Compare CV with role needs
- Generate tailored cover letter

## Tech Stack

- Python
- Smolagents
- CrewAI
- Hugging Face Inference API

## Run

```bash
pip install -r requirements.txt
python smolagents_app.py
python crewai_app.py