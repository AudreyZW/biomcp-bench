"""The three API clients, bundled with their shared lifecycle.

Each client gets its own rate limiter. They talk to independent services, so a
single shared limiter would make a slow ChEMBL throttle PDB requests that have
nothing to do with it — and it would give the three services a combined budget
none of them asked for.

The cache *is* shared. It is content-addressed and namespaced per client, so
there is no collision, and one directory is easier to pin for the benchmark.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from ..cache import RateLimiter, ResponseCache
from .base import ApiClient, ApiError
from .chembl import ChemblClient
from .pdb import PdbClient
from .uniprot import UniprotClient

__all__ = [
    "ApiClient",
    "ApiError",
    "ChemblClient",
    "ClientBundle",
    "PdbClient",
    "UniprotClient",
    "client_lifespan",
]

DEFAULT_MIN_INTERVAL = 0.2


@dataclass
class ClientBundle:
    uniprot: UniprotClient
    pdb: PdbClient
    chembl: ChemblClient

    @classmethod
    def build(
        cls,
        cache: ResponseCache | None = None,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ) -> ClientBundle:
        shared_cache = cache if cache is not None else ResponseCache()
        return cls(
            uniprot=UniprotClient(cache=shared_cache, limiter=RateLimiter(min_interval)),
            pdb=PdbClient(cache=shared_cache, limiter=RateLimiter(min_interval)),
            chembl=ChemblClient(cache=shared_cache, limiter=RateLimiter(min_interval)),
        )

    async def aclose(self) -> None:
        for client in (self.uniprot, self.pdb, self.chembl):
            await client.aclose()


def client_lifespan(bundle: ClientBundle):
    """An MCPServer lifespan that closes the bundle's connections on shutdown.

    Without this the httpx pools are never closed; `aclose` existed before but
    nothing called it, which is a leak that only shows up under a long-running
    server or a test suite that builds many of them.
    """

    @asynccontextmanager
    async def lifespan(_server: object) -> AsyncIterator[ClientBundle]:
        try:
            yield bundle
        finally:
            await bundle.aclose()

    return lifespan
