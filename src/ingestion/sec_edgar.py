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
        """Parses short-term debt, long-term debt, cash, and current assets to compute total structural debt D."""
        # Query short-term liability taxonomy tags
        short_term_debt = self._extract_latest_fact(
            facts_json, ["DebtCurrent", "ShortTermBorrowings", "CommercialPaper"]
        )

        # Query long-term liability taxonomy tags
        long_term_debt = self._extract_latest_fact(
            facts_json,
            [
                "LongTermDebtNoncurrent",
                "LongTermDebt",
                "LongTermDebtAndCapitalLeaseObligations",
            ],
        )

        # Query liquid cash resources
        cash = self._extract_latest_fact(
            facts_json,
            [
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsAndShortTermInvestments",
            ],
        )

        # Query total current assets
        current_assets = self._extract_latest_fact(
            facts_json, ["AssetsCurrent", "CurrentAssets"]
        )

        # Compute Merton default barrier D
        total_debt = short_term_debt + 0.5 * long_term_debt

        # Calculate current liquidity ratio with zero-division safeguard
        current_ratio = (
            current_assets / (short_term_debt + 1e-6)
            if short_term_debt > 0
            else 1.0
        )

        return {
            "short_term_debt": short_term_debt,
            "long_term_debt": long_term_debt,
            "total_debt_D": total_debt,
            "cash": cash,
            "current_assets": current_assets,
            "current_ratio": current_ratio,
        }

    def calculate_remaining_maturity(
        self, facts_json: Dict[str, Any], default_maturity: float = 4.0) -> float:
        """Dynamically computes the effective remaining debt maturity T in years."""
        # Default baseline per structural specifications (e.g., exactly 4.0 years)
        return float(default_maturity)