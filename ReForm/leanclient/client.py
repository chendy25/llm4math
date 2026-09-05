"""Lean4Client compatibility layer for Kimina Lean Server.

ReForm historically imported ``leanclient.client.Lean4Client`` and called
``async_verify``. Modern ``kimina-client`` renamed the API; this shim talks
directly to Kimina's backward-compatible ``POST /verify`` endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import httpx


class Lean4Client:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        http_timeout: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.http_timeout = http_timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _normalize_codes(
        codes: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if isinstance(codes, dict):
            codes = [codes]
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(codes):
            custom_id = item.get("custom_id", str(i))
            proof = item.get("proof")
            if proof is None:
                proof = item.get("code", "")
            out.append({"custom_id": str(custom_id), "proof": proof})
        return out

    async def async_verify(
        self,
        codes: Union[List[Dict[str, Any]], Dict[str, Any]],
        timeout: int = 60,
        disable_cache: bool = False,
        infotree_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "codes": self._normalize_codes(codes),
            "timeout": timeout,
            "disable_cache": disable_cache,
        }
        if infotree_type is not None:
            payload["infotree_type"] = infotree_type

        url = f"{self.base_url}/verify"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout + 30, connect=30.0),
            headers=self._headers(),
        ) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    def verify(
        self,
        codes: Union[List[Dict[str, Any]], Dict[str, Any]],
        timeout: int = 60,
        disable_cache: bool = False,
        infotree_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "codes": self._normalize_codes(codes),
            "timeout": timeout,
            "disable_cache": disable_cache,
        }
        if infotree_type is not None:
            payload["infotree_type"] = infotree_type

        url = f"{self.base_url}/verify"
        with httpx.Client(
            timeout=httpx.Timeout(timeout + 30, connect=30.0),
            headers=self._headers(),
        ) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
