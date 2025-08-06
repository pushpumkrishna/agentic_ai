import streamlit as st
import json
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, LLM, Process
import json


try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
    print("Imported !!")
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Not Imported ")


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()


class ResumeParsingAgent(Agent):
    """
    Resume Parsing Agent:

    Role: Extracts structured details from resumes (name, skills, education, work history).
    Purpose: Convert free-form content into clean, structured data.
    """
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="Resume Parser",
            backstory="I extract structured data from resumes with high accuracy.",
            goal="Parse resumes to extract skills, experience, education, certifications, and career gaps."
        )

    def execute_task(self, task: Task, context: list = None, tools: list = None):
        resume_content = task.description
        if not resume_content:
            raise ValueError("Resume content is required.")

        prompt = f"""
        You are a resume parser. Extract the following fields from the text:
        - Name
        - Education
        - Skills
        - Work experience (roles, companies, duration)

        Resume:
        {resume_content}
        """
        structured_data = self.llm.call([{"role": "user", "content": prompt}])
        return json.dumps({"resume": structured_data.strip()})