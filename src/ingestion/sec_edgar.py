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

    async def fetch_company_facts(
        self, cik_or_ticker: str, session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
        """Asynchronously requests full XBRL company facts JSON for a given ticker or CIK."""
        should_close = False
        if session is None:
            session = aiohttp.ClientSession()
            should_close = True

        try:
            # Check if input is a raw CIK (10 digits) or needs resolution from ticker
            if cik_or_ticker.isdigit() and len(cik_or_ticker) == 10:
                cik = cik_or_ticker
            else:
                cik = await self._resolve_cik(cik_or_ticker, session)
                if not cik:
                    raise ValueError(f"Ticker '{cik_or_ticker}' could not be resolved to a valid SEC CIK.")

            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    raise ConnectionError(
                        f"SEC EDGAR API returned HTTP {response.status} for CIK {cik}"
                    )
                return await response.json()
        finally:
            if should_close:
                await session.close()

    def _extract_latest_fact(
        self,
        facts_data: Dict[str, Any],
        tag_names: list[str],
        form_types: tuple[str, ...] = ("10-Q", "10-K"),
    ) -> float:
        """Helper to safely traverse US-GAAP tags and retrieve the most recent reported numeric value."""
        us_gaap = facts_data.get("facts", {}).get("us-gaap", {})

        for tag in tag_names:
            if tag in us_gaap:
                units = us_gaap[tag].get("units", {}).get("USD", [])
                # Filter strictly for standard corporate periodic filings with reported values
                filtered = [
                    u
                    for u in units
                    if u.get("form") in form_types and "val" in u
                ]
                if filtered:
                    # Sort chronologically by the period end date descending
                    filtered.sort(
                        key=lambda x: x.get("end", "1900-01-01"), reverse=True
                    )
                    return float(filtered[0]["val"])

        return 0.0
    
    def extract_liabilities(self, facts_json: Dict[str, Any]) -> Dict[str, float]:
        # TODO: Parse short-term debt, long-term debt, cash, and current assets[cite: 1]
        pass

    def calculate_remaining_maturity(
        self, facts_json: Dict[str, Any]
    ) -> float:
        # TODO: Dynamically isolate remaining debt maturity T (e.g., 4.0 years)[cite: 1, 2]
        pass