import os
import re
import sys
import traceback
import requests

from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is missing")

    numbered_code = "\n".join(
        f"{i}: {line}"
        for i, line in enumerate(code.splitlines(), start=1)
    )

    prompt = f"""
Analyze this Python code and traceback.

Identify the exact line number or line numbers in the user's code
where the error occurred.

Do not invent line numbers.

NUMBERED CODE:
{numbered_code}

TRACEBACK:
{traceback_text}
"""

    schema = ErrorAnalysis.model_json_schema()

    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "error_analysis",
                    "strict": True,
                    "schema": schema
                }
            }
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()
    ai_text = data["choices"][0]["message"]["content"]

    analysis = ErrorAnalysis.model_validate_json(ai_text)

    return analysis.error_lines


def extract_error_lines_from_traceback(traceback_text: str) -> List[int]:
    matches = re.findall(
        r'File "<string>", line (\d+)',
        traceback_text
    )

    return sorted(set(int(x) for x in matches))


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"]
        )

    except Exception:
        error_lines = extract_error_lines_from_traceback(
            execution["output"]
        )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
