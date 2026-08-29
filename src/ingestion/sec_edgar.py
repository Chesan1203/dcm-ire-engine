from typing import Any, Dict
import aiohttp


class SECEdgarIngestor:
    """Handles asynchronous ingestion of corporate balance sheets from SEC EDGAR[cite: 1]."""

    def __init__(self, user_agent: str = "DCM-IRE_FormalProject f20240535@hyderabad.bits-pilani.ac.in") -> None:
        self.headers: Dict[str, str] = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._ticker_cik_cache: Dict[str, str] = {} #store ticker to CIK pairs

    async def _resolve_cik(
        self, ticker: str, session: aiohttp.ClientSession) -> Optional[str]:
        """Internal helper: Resolves ticker to a 10-digit zero-padded SEC CIK string."""
        ticker_clean = ticker.upper().strip()
        if not self._ticker_cik_cache:
            url = "https://www.sec.gov/files/company_tickers.json"
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                for item in data.values():
                    raw_cik = str(item["cik_str"])
                    self._ticker_cik_cache[item["ticker"].upper()] = (
                        raw_cik.zfill(10)
                    )
        return self._ticker_cik_cache.get(ticker_clean)

    async def fetch_company_facts(self, cik: str) -> Dict[str, Any]:
        # TODO: Asynchronously request 10-Q JSON via data.sec.gov API[cite: 1, 2]
        pass

    def extract_liabilities(self, facts_json: Dict[str, Any]) -> Dict[str, float]:
        # TODO: Parse short-term debt, long-term debt, cash, and current assets[cite: 1]
        pass

    def calculate_remaining_maturity(
        self, facts_json: Dict[str, Any]
    ) -> float:
        # TODO: Dynamically isolate remaining debt maturity T (e.g., 4.0 years)[cite: 1, 2]
        pass