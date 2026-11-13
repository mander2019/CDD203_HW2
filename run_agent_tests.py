"""Run JSON test questions through the interactive Chemical Q&A agent.

The agent is started once so that model loading happens only once. Each test
question is sent as a separate prompt, and the captured responses are written
to a CSV file for review.

Usage:
    python run_agent_tests.py
    python run_agent_tests.py --agent Homework2.py --output results.csv
"""

import argparse
import csv
import json
import subprocess
import statistics
import sys
import time
from pathlib import Path


PROMPT_MARKER = "Prompt (type 'quit' to exit):"


def format_duration(seconds: float) -> str:
    """Format a duration compactly for progress output."""
    rounded_seconds = int(round(seconds))
    minutes, remaining_seconds = divmod(rounded_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def load_cases(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return cases


def read_until_marker(stream) -> str:
    """Read from a stream until the interactive prompt marker is encountered."""
    content = []
    marker_length = len(PROMPT_MARKER)
    recent = ""
    while True:
        character = stream.read(1)
        if character == "":
            break
        recent += character
        if len(recent) > marker_length:
            content.append(recent[0])
            recent = recent[1:]
        if recent == PROMPT_MARKER:
            break
    return "".join(content).strip()


def run_agent(agent: Path, questions: list[str], timeout: int) -> list[str]:
    test_start = time.perf_counter()
    command = [sys.executable, str(agent)]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    # Consume model-loading output and the first prompt before submitting tests.
    read_until_marker(process.stdout)

    answers = []
    response_times = []
    total = len(questions)
    for completed_count, question in enumerate(questions, start=1):
        if process.stdin is None:
            raise RuntimeError("Could not open the agent's standard input.")
        response_start = time.perf_counter()
        process.stdin.write(question + "\n")
        process.stdin.flush()
        answers.append(read_until_marker(process.stdout))
        response_duration = time.perf_counter() - response_start
        response_times.append(response_duration)
        percent = completed_count / total * 100
        print(
            f"{completed_count}/{total} | {percent:.1f}% | "
            f"{format_duration(response_duration)}",
            flush=True,
        )

    process.stdin.write("quit\n")
    process.stdin.flush()
    process.stdin.close()
    process.wait(timeout=timeout)
    if process.returncode != 0:
        print(f"Agent exited with status {process.returncode}.", file=sys.stderr)
    total_duration = time.perf_counter() - test_start
    print(f"Total run time: {format_duration(total_duration)}", flush=True)
    if response_times:
        fastest_index, fastest_time = min(enumerate(response_times, start=1), key=lambda item: item[1])
        slowest_index, slowest_time = max(enumerate(response_times, start=1), key=lambda item: item[1])
        print("Response metrics:", flush=True)
        print(f"  Average: {format_duration(statistics.mean(response_times))}", flush=True)
        print(f"  Median:  {format_duration(statistics.median(response_times))}", flush=True)
        print(
            f"  Fastest: {format_duration(fastest_time)} (test {fastest_index}/{total})",
            flush=True,
        )
        print(
            f"  Slowest: {format_duration(slowest_time)} (test {slowest_index}/{total})",
            flush=True,
        )
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("Homework2.py"), help="Agent script to run")
    parser.add_argument("--cases", type=Path, default=Path("test_cases.json"), help="JSON test-case file")
    parser.add_argument("--output", type=Path, default=Path("test_results.csv"), help="CSV results file")
    parser.add_argument("--timeout", type=int, default=3600, help="Maximum runtime in seconds")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    questions = [case["question"] for case in cases]
    print(f"Running {len(questions)} test cases using {args.agent}...")
    answers = run_agent(args.agent, questions, args.timeout)

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "question", "correct_answer", "provided_answer"],
        )
        writer.writeheader()
        for case, answer in zip(cases, answers):
            writer.writerow(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "correct_answer": case["correct_answer"],
                    "provided_answer": answer,
                }
            )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
