import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.router import question_router


def test_router():
    print(question_router.invoke({"question": "Show me the cost trend of our surveys"}))
    print(
        question_router.invoke(
            {
                "question": "Using the survey data response and issue behaviors, explore potential ways we could reduce cost."
            }
        )
    )
    print(
        question_router.invoke(
            {
                "question": "Analyze if there's any response patterns that could help us reduce survey costs"
            }
        )
    )
    print(
        question_router.invoke(
            {
                "question": "What is our current rates from our comm platform for surveys?"
            }
        )
    )

    print(
        question_router.invoke(
            {"question": "Explore ways we could be more effecient with survey methods?"}
        )
    )


if __name__ == "__main__":
    test_router()
