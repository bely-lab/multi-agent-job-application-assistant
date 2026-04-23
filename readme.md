# CrewAI Job Application Assistant

A multi-agent AI project built with **CrewAI** and **Hugging Face Router API** for automating parts of the job application process.

The system uses multiple specialized AI agents working sequentially to analyze a job posting, review a candidate CV, and generate a tailored cover letter.

## Features

- Extracts required and preferred skills from job descriptions
- Reviews a CV against job requirements
- Identifies strengths, gaps, and recommendations
- Generates a customized cover letter
- Saves final output automatically in an `output/` folder

## Agents Used

### 1. Job Requirements Researcher
Analyzes the job description and summarizes hiring needs.

### 2. CV Reviewer
Compares the candidate CV with the job requirements.

### 3. Cover Letter Writer
Creates a personalized cover letter based on previous analysis.

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
