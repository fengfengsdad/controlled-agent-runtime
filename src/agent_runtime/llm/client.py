from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from agent_runtime.config import settings
from agent_runtime.models.schemas import Citation, DeliveryPlan, DeliveryTask
from agent_runtime.security.controls import redact_secrets

PLAN_SYSTEM_PROMPT = """You are a delivery planning assistant for engineering change requests.
Return ONLY valid JSON with keys: summary (string), risks (string[]), tasks (object[]).
Each task needs: title, owner_role, estimate_days, dependencies (string[]),
acceptance_criteria (string[]).
Use only the labelled retrieved context. After every factual claim in summary
and risks, attach an inline citation of the form [S1] using a label that appears
in the context. Do not invent labels. If the labelled evidence does not support
a claim, omit the claim rather than guessing.
Prompt version: {prompt_version}.
"""


class LLMClient(ABC):
    @abstractmethod
    def generate_plan(
        self,
        requirement: str,
        citations: list[Citation],
        tool_context: str = "",
        context_text: str = "",
    ) -> DeliveryPlan:
        raise NotImplementedError


class StubLLMClient(LLMClient):
    def generate_plan(
        self,
        requirement: str,
        citations: list[Citation],
        tool_context: str = "",
        context_text: str = "",
    ) -> DeliveryPlan:
        safe_req = redact_secrets(requirement)
        title_seed = re.sub(r"\s+", " ", safe_req).strip()[:80]
        labels = re.findall(r"\[S\d+\]", context_text)
        primary = labels[0] if labels else ""
        secondary = labels[1] if len(labels) > 1 else primary
        if primary:
            citation_note = (
                f"Retrieved {len(citations)} labelled context chunk(s) {primary}. "
                f"Idempotency and approval controls must stay aligned {primary}."
            )
            risks = [
                f"Ambiguous acceptance criteria may expand scope {primary}.",
                f"Missing production telemetry for post-change verification {secondary}.",
            ]
        else:
            citation_note = "No retrieval hits; planned from requirement only."
            risks = [
                "Ambiguous acceptance criteria may expand scope.",
                "Missing production telemetry for post-change verification.",
            ]
        tasks = [
            DeliveryTask(
                title=f"Clarify scope: {title_seed}",
                owner_role="tech_lead",
                estimate_days=0.5,
                acceptance_criteria=[
                    "Requirement boundaries confirmed",
                    "Non-goals documented",
                ],
            ),
            DeliveryTask(
                title="Implement controlled change with tests",
                owner_role="engineer",
                estimate_days=2.0,
                dependencies=["Clarify scope"],
                acceptance_criteria=[
                    "Unit and integration tests pass",
                    "Audit events emitted for write path",
                ],
            ),
            DeliveryTask(
                title="Roll out behind approval gate",
                owner_role="platform",
                estimate_days=1.0,
                dependencies=["Implement controlled change with tests"],
                acceptance_criteria=[
                    "Human approval recorded",
                    "Runbook updated",
                ],
            ),
        ]
        if "payment" in safe_req.lower() and primary:
            risks.append(f"Payment path changes require regression coverage {primary}.")
        elif "payment" in safe_req.lower():
            risks.append("Payment path changes require regression coverage.")
        summary = (
            f"Stub plan for the engineering change {primary or 'without retrieved evidence'}. "
            f"{citation_note} Tool context length={len(tool_context)}."
        )
        return DeliveryPlan(
            summary=summary,
            risks=risks,
            tasks=tasks,
            citations=citations,
            prompt_version=settings.prompt_version,
            model="stub",
        )


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate_plan(
        self,
        requirement: str,
        citations: list[Citation],
        tool_context: str = "",
        context_text: str = "",
    ) -> DeliveryPlan:
        labelled = context_text.strip() or "(none)"
        user_prompt = (
            f"Requirement:\n{redact_secrets(requirement)}\n\n"
            f"Retrieved context:\n{labelled}\n\n"
            f"Tool context:\n{tool_context or '(none)'}\n"
        )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": PLAN_SYSTEM_PROMPT.format(
                        prompt_version=settings.prompt_version
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        tasks = [DeliveryTask.model_validate(t) for t in data.get("tasks", [])]
        return DeliveryPlan(
            summary=str(data.get("summary", "")),
            risks=[str(r) for r in data.get("risks", [])],
            tasks=tasks,
            citations=citations,
            prompt_version=settings.prompt_version,
            model=self.model,
        )


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "openai_compatible":
        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY required for openai_compatible provider")
        return OpenAICompatibleLLMClient(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
        )
    return StubLLMClient()
