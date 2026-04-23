import os
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from crewai import Agent, Task, Crew, Process, BaseLLM


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"{path} is empty or invalid.")

    return data


def save_output(content: str) -> None:
    os.makedirs("output", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"output/final_result_{timestamp}.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(content))

    print(f"\nResult saved to {file_path}")


class HFRouterLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.2,
    ):
        super().__init__(model=model, temperature=temperature)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)

        return response.choices[0].message.content or ""

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 262144


def build_llm() -> HFRouterLLM:
    api_key = os.getenv("HF_TOKEN")

    if not api_key:
        raise ValueError("HF_TOKEN is not set.")

    return HFRouterLLM(
        model="moonshotai/Kimi-K2.6:novita",
        api_key=api_key,
        base_url="https://router.huggingface.co/v1",
        temperature=0.2,
    )


def build_agent(config: dict, llm: BaseLLM) -> Agent:
    return Agent(
        role=config["role"],
        goal=config["goal"],
        backstory=config["backstory"],
        llm=llm,
        verbose=True,
    )


if __name__ == "__main__":
    print("Starting Task 10-2 CrewAI Job Application Assistant...")

    agents_config = load_yaml("agents.yaml")
    tasks_config = load_yaml("tasks.yaml")

    cv_text = read_file("sample_cv.txt")
    job_text = read_file("sample_job_description.txt")

    llm = build_llm()

    job_researcher = build_agent(agents_config["job_researcher"], llm)
    cv_reviewer = build_agent(agents_config["cv_reviewer"], llm)
    cover_letter_writer = build_agent(agents_config["cover_letter_writer"], llm)

    task1 = Task(
        description=f"""
{tasks_config["extract_job_requirements"]["description"]}

Job description:
{job_text}
""",
        expected_output=tasks_config["extract_job_requirements"]["expected_output"],
        agent=job_researcher,
    )

    task2 = Task(
        description=f"""
{tasks_config["review_cv_against_job"]["description"]}

Candidate CV:
{cv_text}

Job description:
{job_text}
""",
        expected_output=tasks_config["review_cv_against_job"]["expected_output"],
        agent=cv_reviewer,
        context=[task1],
    )

    task3 = Task(
        description=f"""
{tasks_config["write_cover_letter"]["description"]}

Candidate CV:
{cv_text}

Job description:
{job_text}
""",
        expected_output=tasks_config["write_cover_letter"]["expected_output"],
        agent=cover_letter_writer,
        context=[task1, task2],
    )

    crew = Crew(
        agents=[job_researcher, cv_reviewer, cover_letter_writer],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n=== FINAL CREW RESULT ===\n")
    print(result)

    save_output(result)