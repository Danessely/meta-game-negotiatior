from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text
from openai import AsyncOpenAI


ACTION_RE = re.compile(r"Action:\s*(PROPOSE|ACCEPT_OR_REJECT)\b", re.IGNORECASE)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
OBSERVATION_RE = re.compile(r"Observation:\s*", re.IGNORECASE)

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_OPENAI_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "8"))


@dataclass(slots=True)
class NegotiationObservation:
    action: str
    role: str | None
    round_index: int
    max_rounds: int
    discount: float
    quantities: list[int]
    valuations_self: list[int]
    batna_self: int
    pending_offer_self: list[int] | None = None
    offer_value: int | None = None

    @property
    def total_value(self) -> int:
        return allocation_value(self.valuations_self, self.quantities)

    @property
    def current_offer_value(self) -> int | None:
        if self.offer_value is not None:
            return self.offer_value
        if self.pending_offer_self is None:
            return None
        return allocation_value(self.valuations_self, self.pending_offer_self)

    def is_complete(self) -> bool:
        return bool(
            self.action
            and self.quantities
            and self.valuations_self
            and len(self.quantities) == len(self.valuations_self)
        )


def allocation_value(values: Sequence[int], allocation: Sequence[int]) -> int:
    return int(sum(int(value) * int(amount) for value, amount in zip(values, allocation, strict=True)))


def coerce_int_list(raw: Any) -> list[int] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    try:
        return [int(float(item)) for item in raw]
    except (TypeError, ValueError):
        return None


def load_json_dict(payload: str) -> dict[str, Any] | None:
    text = payload.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def find_first_json_dict(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def extract_observation_dict(message_text: str) -> dict[str, Any]:
    observation_match = OBSERVATION_RE.search(message_text)
    if observation_match:
        observation_text = message_text[observation_match.end() :]
        for block in FENCED_JSON_RE.findall(observation_text):
            data = load_json_dict(block)
            if data is not None:
                return data
        data = find_first_json_dict(observation_text)
        if data is not None:
            return data

    for block in FENCED_JSON_RE.findall(message_text):
        data = load_json_dict(block)
        if data is not None and any(
            key in data for key in ("quantities", "valuations_self", "batna_self", "pending_offer", "offer_value")
        ):
            return data

    return find_first_json_dict(message_text) or {}


def parse_observation(message_text: str) -> NegotiationObservation | None:
    action_match = ACTION_RE.search(message_text)
    action = action_match.group(1).upper() if action_match else ""
    raw = extract_observation_dict(message_text)

    if not action and isinstance(raw.get("action"), str):
        candidate_action = str(raw["action"]).strip().upper()
        if candidate_action in {"PROPOSE", "ACCEPT_OR_REJECT"}:
            action = candidate_action

    if not action:
        return None

    quantities = coerce_int_list(raw.get("quantities")) or []
    valuations_self = coerce_int_list(raw.get("valuations_self") or raw.get("valuations")) or []
    batna_raw = raw.get("batna_self", raw.get("batna", 0))
    round_raw = raw.get("round_index", raw.get("round", 1))
    max_rounds_raw = raw.get("max_rounds", raw.get("rounds", 1))
    discount_raw = raw.get("discount", 1.0)

    try:
        batna_self = int(float(batna_raw))
    except (TypeError, ValueError):
        batna_self = 0

    try:
        round_index = max(1, int(float(round_raw)))
    except (TypeError, ValueError):
        round_index = 1

    try:
        max_rounds = max(1, int(float(max_rounds_raw)))
    except (TypeError, ValueError):
        max_rounds = 1

    try:
        discount = float(discount_raw)
    except (TypeError, ValueError):
        discount = 1.0

    pending_offer_self = None
    pending_offer = raw.get("pending_offer")
    if isinstance(pending_offer, dict):
        pending_offer_self = coerce_int_list(
            pending_offer.get("offer_allocation_self")
            or pending_offer.get("allocation_self")
            or pending_offer.get("current_offer")
        )

    offer_value = raw.get("offer_value")
    try:
        parsed_offer_value = None if offer_value is None else int(float(offer_value))
    except (TypeError, ValueError):
        parsed_offer_value = None

    return NegotiationObservation(
        action=action,
        role=raw.get("role"),
        round_index=round_index,
        max_rounds=max_rounds,
        discount=discount,
        quantities=quantities,
        valuations_self=valuations_self,
        batna_self=batna_self,
        pending_offer_self=pending_offer_self,
        offer_value=parsed_offer_value,
    )


class Agent:
    def __init__(self):
        self._model = DEFAULT_OPENAI_MODEL
        api_key = os.environ.get("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key, timeout=DEFAULT_OPENAI_TIMEOUT) if api_key else None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message) or ""
        response = await self.build_response(input_text)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(response, separators=(",", ":"))))],
            name="decision",
        )

    async def build_response(self, message_text: str) -> dict[str, Any]:
        observation = parse_observation(message_text)
        fallback = self._heuristic_response(observation)

        if observation is None or not observation.is_complete():
            return fallback

        llm_response = await self._maybe_get_llm_response(observation, fallback)
        if llm_response is None:
            return fallback

        return self._normalize_response(observation, llm_response, fallback)

    async def _maybe_get_llm_response(
        self,
        observation: NegotiationObservation,
        fallback: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self._client is None:
            return None

        if observation.action == "PROPOSE":
            response_shape = '{"allocation_self":[...],"allocation_other":[...],"reason":"..."}'
        else:
            response_shape = '{"accept":true|false,"reason":"..."}'

        prompt = {
            "observation": {
                "action": observation.action,
                "role": observation.role,
                "round_index": observation.round_index,
                "max_rounds": observation.max_rounds,
                "discount": observation.discount,
                "quantities": observation.quantities,
                "valuations_self": observation.valuations_self,
                "batna_self": observation.batna_self,
                "pending_offer_self": observation.pending_offer_self,
                "offer_value": observation.current_offer_value,
            },
            "fallback": fallback,
            "required_response": response_shape,
        }

        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a bargaining responder. "
                            "Reply with a single JSON object only, with no markdown and no explanation outside JSON. "
                            "Never use information not present in the observation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt),
                    },
                ],
            )
        except Exception:
            return None

        content = completion.choices[0].message.content
        if not content:
            return None
        return load_json_dict(content)

    def _heuristic_response(self, observation: NegotiationObservation | None) -> dict[str, Any]:
        if observation is None or not observation.is_complete():
            return {"action": "WALK", "reason": "Unrecognized or incomplete negotiation prompt."}

        if observation.action == "ACCEPT_OR_REJECT":
            offer_value = observation.current_offer_value
            if offer_value is None:
                return {"accept": False, "reason": "No valid offer to evaluate."}

            threshold = self._proposal_target_value(observation)
            accept = offer_value >= observation.batna_self and (
                observation.round_index >= observation.max_rounds or offer_value >= threshold
            )
            return {
                "accept": accept,
                "reason": (
                    f"Offer value {offer_value} compared to batna {observation.batna_self} "
                    f"and target {threshold}."
                ),
            }

        min_value = self._proposal_target_value(observation)
        keep_allocation = self._make_keep_allocation(observation.quantities, observation.valuations_self, min_value)
        other_allocation = [
            total - kept for total, kept in zip(observation.quantities, keep_allocation, strict=True)
        ]
        return {
            "allocation_self": keep_allocation,
            "allocation_other": other_allocation,
            "reason": f"Keep value {allocation_value(observation.valuations_self, keep_allocation)} with target {min_value}.",
        }

    def _proposal_target_value(self, observation: NegotiationObservation) -> int:
        total_value = observation.total_value
        batna = min(max(observation.batna_self, 0), total_value)
        current_offer_value = observation.current_offer_value or 0

        if observation.max_rounds <= 1:
            progress = 1.0
        else:
            progress = (observation.round_index - 1) / (observation.max_rounds - 1)
            progress = min(max(progress, 0.0), 1.0)

        surplus = max(total_value - batna, 0)
        surplus_fraction = 0.70 - 0.50 * progress
        target_value = batna + math.ceil(surplus * surplus_fraction)
        return min(total_value, max(batna, current_offer_value, target_value))

    def _make_keep_allocation(
        self,
        quantities: Sequence[int],
        valuations_self: Sequence[int],
        min_value: int,
    ) -> list[int]:
        keep = [int(quantity) for quantity in quantities]
        current_value = allocation_value(valuations_self, keep)
        concession_order = sorted(
            range(len(keep)),
            key=lambda index: (int(valuations_self[index]), -int(quantities[index]), index),
        )

        for item_index in concession_order:
            item_value = int(valuations_self[item_index])
            while keep[item_index] > 0 and current_value - item_value >= min_value:
                keep[item_index] -= 1
                current_value -= item_value

        return keep

    def _normalize_response(
        self,
        observation: NegotiationObservation,
        candidate: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if observation.action == "ACCEPT_OR_REJECT":
            decision = candidate.get("accept")
            if decision is None and isinstance(candidate.get("action"), str):
                action = candidate["action"].strip().upper()
                if action == "ACCEPT":
                    decision = True
                elif action in {"WALK", "COUNTEROFFER", "COUNTER_OFFER", "OFFER"}:
                    decision = False
            if not isinstance(decision, bool):
                return fallback
            offer_value = observation.current_offer_value
            if decision and (
                offer_value is None
                or offer_value < observation.batna_self
                or (
                    observation.round_index < observation.max_rounds
                    and offer_value < self._proposal_target_value(observation)
                )
            ):
                return fallback
            return {
                "accept": decision,
                "reason": str(candidate.get("reason", fallback.get("reason", ""))),
            }

        allocation_self = coerce_int_list(
            candidate.get("allocation_self") or candidate.get("allocation") or candidate.get("offer")
        )
        allocation_other = coerce_int_list(candidate.get("allocation_other"))
        if allocation_self is None:
            return fallback
        if allocation_other is None:
            allocation_other = [
                int(total) - int(kept)
                for total, kept in zip(observation.quantities, allocation_self, strict=True)
            ]

        if not self._is_valid_allocation(observation, allocation_self, allocation_other):
            return fallback

        if allocation_value(observation.valuations_self, allocation_self) < max(
            observation.batna_self,
            observation.current_offer_value or 0,
        ):
            return fallback

        return {
            "allocation_self": allocation_self,
            "allocation_other": allocation_other,
            "reason": str(candidate.get("reason", fallback.get("reason", ""))),
        }

    def _is_valid_allocation(
        self,
        observation: NegotiationObservation,
        allocation_self: Sequence[int],
        allocation_other: Sequence[int],
    ) -> bool:
        if len(allocation_self) != len(observation.quantities):
            return False
        if len(allocation_other) != len(observation.quantities):
            return False

        for total, kept, other in zip(observation.quantities, allocation_self, allocation_other, strict=True):
            if kept < 0 or other < 0 or kept + other != int(total):
                return False
        return True
