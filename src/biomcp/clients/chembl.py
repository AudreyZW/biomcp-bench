"""ChEMBL bioactivity and compound data.

API notes worth knowing before changing this file:

* Pagination is `limit`/`offset` with a `page_meta.total_count`, so the caller
  can report how much was truncated rather than silently dropping rows.
* `/target/search?q=` returns intermittent HTTP 500s. `find_targets` therefore
  treats it as best-effort and falls back to a `pref_name__icontains` filter
  query, which is stable. Both were verified against the live API.
* Potency should be compared on `pchembl_value`, never on `standard_value`,
  which is reported in mixed units across assays.
* `standard_type` is case-sensitive and a wrong case fails *silently*: at the
  time of writing `standard_type=IC50` matches 26,600 rows for CHEMBL203 while
  `ic50` matches 0, both with HTTP 200. `canonical_assay_type` normalises the
  common ones so a lower-case argument does not look like "no data".

Methods here return upstream records unchanged. Projection happens at the
server boundary so that `evals/` can serve the same data shaped or raw and
attribute any difference to the shaping alone.
"""

from __future__ import annotations

from typing import Any

from .base import ApiClient, ApiError

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# ChEMBL's own spelling of the assay types worth filtering on. Keyed by the
# upper-case form so any casing a caller supplies can be mapped back.
_CANONICAL_ASSAY_TYPES = {
    t.upper(): t
    for t in (
        "IC50",
        "EC50",
        "AC50",
        "GI50",
        "Ki",
        "Kd",
        "MIC",
        "Potency",
        "Activity",
        "Inhibition",
    )
}


def canonical_assay_type(assay_type: str) -> tuple[str, bool]:
    """Map any casing of a known assay type to ChEMBL's spelling.

    Returns the value to send and whether it was recognised. Unknown types are
    passed through untouched -- ChEMBL has hundreds and this list covers only
    the common ones -- but the caller is told, so it can warn instead of
    reporting an empty result as though it were an absence of data.
    """
    canonical = _CANONICAL_ASSAY_TYPES.get(assay_type.strip().upper())
    return (canonical, True) if canonical else (assay_type, False)


class ChemblClient(ApiClient):
    def __init__(self, **kwargs: Any):
        kwargs.setdefault("namespace", "chembl")
        super().__init__(BASE_URL, **kwargs)

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        return await self.request_json("GET", path, params={**params, "format": "json"})

    async def find_targets(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Resolve a free-text protein or gene name to ChEMBL targets."""
        try:
            body = await self._get("target/search", {"q": query, "limit": limit})
        except ApiError:
            body = None  # documented upstream flakiness; fall through
        if not body or not body.get("targets"):
            body = await self._get("target", {"pref_name__icontains": query, "limit": limit})
        return (body or {}).get("targets") or []

    async def target_for_accession(self, accession: str) -> dict[str, Any] | None:
        """Find the ChEMBL target that corresponds to a UniProt accession.

        This is the ChEMBL side of the join. The accession filter is one of the
        queries that fails intermittently upstream, so `ApiError` is allowed to
        propagate and callers degrade rather than retrying forever here.
        """
        body = (
            await self._get(
                "target",
                {"target_components__accession": accession, "limit": 1},
            )
            or {}
        )
        targets = body.get("targets") or []
        return targets[0] if targets else None

    async def activities(
        self,
        target_chembl_id: str,
        *,
        assay_type: str | None = None,
        min_potency: float | None = None,
        limit: int = 25,
    ) -> tuple[list[dict[str, Any]], int]:
        """Bioactivity rows for a target, most potent first.

        Returns the shaped rows and the upstream total, so a caller can tell the
        model it is looking at 25 of 29,582 rather than implying completeness.
        """
        params: dict[str, Any] = {
            "target_chembl_id": target_chembl_id,
            "limit": min(limit, 100),
            "order_by": "-pchembl_value",
            "pchembl_value__isnull": "false",
        }
        if assay_type:
            params["standard_type"] = canonical_assay_type(assay_type)[0]
        if min_potency is not None:
            params["pchembl_value__gte"] = min_potency
        body = await self._get("activity", params) or {}
        rows = body.get("activities") or []
        total = int((body.get("page_meta") or {}).get("total_count") or len(rows))
        return rows, total

    async def molecules(self, chembl_ids: list[str]) -> list[dict[str, Any]]:
        if not chembl_ids:
            return []
        body = (
            await self._get(
                "molecule",
                {"molecule_chembl_id__in": ",".join(chembl_ids), "limit": len(chembl_ids)},
            )
            or {}
        )
        return body.get("molecules") or []

    async def similar_molecules(
        self, smiles: str, threshold: int = 70, limit: int = 25
    ) -> tuple[list[dict[str, Any]], int]:
        """Tanimoto similarity search. `threshold` is a percentage, 40-100."""
        threshold = max(40, min(100, int(threshold)))
        body = (
            await self.request_json(
                "GET",
                f"similarity/{smiles}/{threshold}",
                params={"format": "json", "limit": min(limit, 100)},
            )
            or {}
        )
        molecules = body.get("molecules") or []
        total = int((body.get("page_meta") or {}).get("total_count") or len(molecules))
        return molecules, total
