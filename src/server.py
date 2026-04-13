import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def create_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="meta-game-bargaining",
        name="Meta-Game Bargaining",
        description="Responds to bargaining prompts from the evaluator with valid JSON offers or accept/reject decisions.",
        tags=["bargaining", "negotiation", "a2a", "agentbeats"],
        examples=[
            "Evaluate an offer against a BATNA and reply with {\"accept\": true}.",
            "Return a valid allocation split with allocation_self and allocation_other.",
        ],
    )

    return AgentCard(
        name="Meta-Game Negotiator",
        description="A reliable bargaining responder for the Meta-Game evaluator with deterministic fallbacks and optional OpenAI assistance.",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    agent_card = create_agent_card(args.card_url or f"http://{args.host}:{args.port}/")

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
