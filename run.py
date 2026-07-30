"""CLI runner to test the graph without Streamlit."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from backend.graph import get_graph
from backend.state import ResearchState


def run_report(topic: str, audience: str = "ML Engineers", tone: str = "analytical"):
    """Run the full research pipeline and print the report."""
    graph = get_graph()

    initial = ResearchState(
        topic=topic,
        audience=audience,
        tone=tone,
    )

    print(f"\n{'='*60}")
    print(f"🔬 Research Intelligence Engine")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")

    for event in graph.stream(
        initial.model_dump(),
        stream_mode="updates",
        config={"recursion_limit": 50},
    ):
        for node_name, output in event.items():
            if isinstance(output, dict):
                status = output.get("status", "")
                logs = output.get("progress_log", [])
                if status:
                    print(f"[{node_name}] {status}")
                for log in logs:
                    print(f"  → {log}")

    # Print final state
    final = graph.invoke(
        initial.model_dump(),
        config={"recursion_limit": 50},
    )

    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    print(final.get("final_report", "No report generated"))
    if final.get("pdf_path"):
        print(f"\n✅ PDF saved to: {final['pdf_path']}")


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "LangGraph vs AutoGen: Technical Comparison for Production AI"
    run_report(topic)
