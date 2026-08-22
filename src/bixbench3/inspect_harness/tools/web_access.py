from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inspect_ai.tool import Tool, ToolError, tool


@tool
def request_web_access() -> Tool:
    async def execute(url: str, reason: str) -> str:
        """Ask the benchmark network gateway to provide one web resource.

        Use this whenever a required method dependency, reference resource, or
        documentation URL is unavailable through the package infrastructure
        already allowed by the proxy. Exact URLs and narrow URL prefixes are
        fetched into ``/workspace/approved_web``. Rare approved package
        infrastructure domains are added to the proxy allowlist.

        The paper, its supplements, deposited records, public ground-truth
        packages, processed result tables, and any resource that overlaps the
        requested outputs are unavailable. General-purpose search results are
        also unavailable.

        Args:
            url: The exact HTTP or HTTPS URL needed.
            reason: A short sentence identifying the method or reference need.

        Returns:
            A JSON decision. For ``workspace_file`` delivery, use ``local_path``.
            For ``allowlist`` delivery, retry the original command.
        """
        endpoint = os.environ.get("BIXBENCH3_BROKER_ENDPOINT")
        token = os.environ.get("BIXBENCH3_BROKER_TOKEN")
        if not endpoint or not token:
            raise ToolError("web access approval service is not configured for this run")
        payload = json.dumps({"url": url, "reason": reason}).encode()
        request = Request(
            endpoint.rstrip("/") + "/approve",
            data=payload,
            headers={
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=45) as response:
                text = response.read().decode("utf-8", errors="replace")
        except HTTPError as error:
            text = error.read().decode("utf-8", errors="replace")
            try:
                decision = json.loads(text)
            except json.JSONDecodeError as json_error:
                raise ToolError(f"web access approval service failed: {error}") from json_error
            return _serialize_decision(decision)
        except URLError as error:
            raise ToolError(f"web access approval service failed: {error}") from error
        try:
            decision = json.loads(text)
        except json.JSONDecodeError as error:
            raise ToolError(
                f"web access approval service returned invalid JSON: {error}"
            ) from error
        return _serialize_decision(decision)

    return execute


def _serialize_decision(value: object) -> str:
    if not isinstance(value, dict):
        raise ToolError("web access approval service returned a non-object response")
    return json.dumps(value, sort_keys=True)
