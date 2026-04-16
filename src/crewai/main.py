#!/usr/bin/env python
import warnings

from crewai import Crew

from crewai.crew import Crewai

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = {
        "topic": "AI Agents",
    }
    Crewai().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI Agents",
    }
    try:
        Crewai().crew().train(n_iterations=int(5), filename="trained_agents_data.pkl", inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Crewai().crew().replay(task_id="")
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI Agents",
    }
    try:
        Crewai().crew().test(n_iterations=int(3), openai_model_name="gpt-4o-mini", inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
