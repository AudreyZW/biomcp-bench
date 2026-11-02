"""UniProt name resolution.

UniProt is here because the other two services cannot reliably resolve a
protein name. ChEMBL's `/target/search?q=` and its `pref_name__icontains`
filter both return intermittent 500s (measured: five identical requests
returned 500, 500, timeout, timeout, 500 while `/target/CHEMBL203` stayed
healthy), and RCSB's full-text search ranks on text relevance rather than
identity, so "EGFR" can return structures of unrelated proteins that merely
mention it.

UniProt accessions are the join key both other services already index, so
resolving a name here once and carrying the accession is both more reliable
and more correct than searching each database by name.
"""

from __future__ import annotations

from typing import Any

from .base import ApiClient

BASE_URL = "https://rest.uniprot.org"

FIELDS = "accession,id,protein_name,gene_primary,organism_name"

# See `resolve` -- size=1 is not the first row of size=2 at this API.
MIN_FETCH = 5


class UniprotClient(ApiClient):
    def __init__(self, **kwargs: Any):
        kwargs.setdefault("namespace", "uniprot")
        super().__init__(BASE_URL, **kwargs)

    async def resolve(
        self,
        query: str,
        *,
        organism_id: int | None = 9606,
        reviewed_only: bool = True,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Resolve a protein or gene name to UniProt entries.

        Defaults to reviewed (Swiss-Prot) human entries, which is what makes
        the first hit usually the right one; pass `organism_id=None` to widen
        the search to other species, e.g. for a viral target.
        """
        clauses = [query]
        if organism_id is not None:
            clauses.append(f"organism_id:{organism_id}")
        if reviewed_only:
            clauses.append("reviewed:true")

        # Never ask for size=1. UniProt returns a *different* top hit at
        # size=1 than at size>=2: for "EGFR AND organism_id:9606 AND
        # reviewed:true", size=1 yields CBLC while size=2 yields EGFR first.
        # Over-fetch and slice so the ranking is the one the archive means.
        fetch = max(min(limit, 25), MIN_FETCH)
        body = await self.request_json(
            "GET",
            "/uniprotkb/search",
            params={
                "query": " AND ".join(clauses),
                "fields": FIELDS,
                "format": "json",
                "size": fetch,
            },
        )
        return ((body or {}).get("results") or [])[:limit]
