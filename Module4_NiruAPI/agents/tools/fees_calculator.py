"""Fees Calculator - Kenyan statutory deductions and fees calculator"""

import re
from typing import Dict, Any, Optional, List
from loguru import logger


class FeesCalculatorTool:
    """Calculate Kenyan statutory deductions, taxes, and government fees"""

    name = "fees_calculator"
    description = "Calculate Kenyan statutory deductions: Housing Levy (1.5%), SHIF (2.75%), NSSF (tiered), PAYE (income tax), parking fees, KRA fines"

    def execute(self, query: str) -> Dict[str, Any]:
        try:
            query_lower = query.lower().strip()
            result = self._parse_and_calculate(query_lower)
            return result
        except Exception as e:
            logger.error(f"Fees calculator failed: {e}")
            return {"success": False, "error": str(e), "result": None, "breakdown": []}

    def _parse_and_calculate(self, query: str) -> Dict[str, Any]:
        income = self._extract_amount(query)
        breakdown = []

        # Housing Levy: 1.5% of gross income (capped discussion)
        if "housing" in query or "levy" in query or income:
            housing = round(income * 0.015, 2) if income else "Requires income amount"
            breakdown.append(
                {
                    "item": "Housing Levy (1.5%)",
                    "rate": "1.5% of gross income",
                    "amount": housing,
                }
            )

        # SHIF: 2.75% of gross household income
        if "shif" in query or "health" in query or "nhif" in query or income:
            shif = round(income * 0.0275, 2) if income else "Requires income amount"
            breakdown.append(
                {
                    "item": "SHIF (2.75%)",
                    "rate": "2.75% of gross household income",
                    "amount": shif,
                }
            )

        # NSSF: Tiered (upper limit ~KES 4,320)
        if "nssf" in query or "social" in query or income:
            nssf_tier1 = 360  # KES per month
            nssf_tier2 = 3960
            breakdown.append(
                {"item": "NSSF Tier 1", "rate": "KES 360/month", "amount": 360}
            )
            breakdown.append(
                {
                    "item": "NSSF Tier 2",
                    "rate": "Up to KES 3,960/month",
                    "amount": "Variable",
                }
            )

        # PAYE brackets
        if "paye" in query or "income tax" in query or "tax" in query or income:
            paye = self._calculate_paye(income) if income else "Requires income amount"
            breakdown.append(
                {
                    "item": "PAYE (Income Tax)",
                    "rate": "Progressive brackets (10-35%)",
                    "amount": paye,
                }
            )

        # Parking fees
        if "parking" in query or "kanjo" in query or "county" in query:
            breakdown.append(
                {
                    "item": "Nairobi Parking Fee",
                    "rate": "KES 300/day or ~KES 6,000/month",
                    "amount": "KES 300/day",
                }
            )
            if "monthly" in query:
                breakdown[-1]["amount"] = "KES 6,000/month"

        # Fuel Levy
        if "fuel" in query or "petrol" in query:
            breakdown.append(
                {
                    "item": "Fuel Levy",
                    "rate": "KES 18/litre (incl. Road Maintenance Levy)",
                    "amount": "KES 18/L",
                }
            )
            if income:
                breakdown.append(
                    {
                        "item": "VAT on Fuel (16%)",
                        "rate": "16% of fuel cost",
                        "amount": "KES 0",
                    }
                )

        # VAT
        if "vat" in query:
            rate = "16% (standard) or 8% (supplies) or 0% (exports)"
            breakdown.append(
                {"item": "VAT", "rate": rate, "amount": "Varies by goods/services"}
            )

        if not breakdown:
            return {
                "success": True,
                "result": "I can calculate: Housing Levy (1.5%), SHIF (2.75%), NSSF (tiered), PAYE, parking fees, fuel levy, and VAT. Please specify what you'd like calculated, optionally including an income amount.",
                "breakdown": [],
                "total": None,
                "note": "Specify an income figure to get exact amounts, e.g. 'calculate housing levy and SHIF on KES 100,000'",
            }

        total = sum(
            v for item in breakdown if isinstance((v := item["amount"]), (int, float))
        )

        return {
            "success": True,
            "result": (
                f"Computed {len(breakdown)} items. Total deductions: KES {total:,.2f}"
                if total
                else "Breakdown computed (some items need income amount)"
            ),
            "breakdown": breakdown,
            "total": total if total else None,
            "note": "These are estimates based on current Kenyan tax law. Consult KRA for exact figures.",
        }

    def _extract_amount(self, text: str) -> Optional[float]:
        patterns = [
            r"(?:kes|ksh|k\s*shs?\.?)\s*([\d,]+(?:\.\d+)?)",
            r"salary\s*(?:of|:)?\s*(?:kes|ksh)?\s*([\d,]+(?:\.\d+)?)",
            r"income\s*(?:of|:)?\s*(?:kes|ksh)?\s*([\d,]+(?:\.\d+)?)",
            r"on\s*(?:kes|ksh)?\s*([\d,]+(?:\.\d+)?)",
            r"(\d[\d,]*)\s*(?:kes|ksh|shillings?)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return float(m.group(1).replace(",", ""))
        return None

    def _calculate_paye(self, annual_income: float) -> Dict[str, Any]:
        brackets = [
            (288000, 0.10),  # First KES 288K @ 10%
            (100000, 0.25),  # Next KES 100K @ 25%
            (412000, 0.30),  # Next KES 412K @ 30%
            (float("inf"), 0.35),  # Above KES 800K @ 35%
        ]
        personal_relief = 2400 / 12  # KES 2,400 per year = KES 200/month

        monthly = annual_income / 12
        remaining = monthly
        total_tax = 0
        details = []

        for bracket_band, rate in brackets:
            if remaining <= 0:
                break
            taxable = min(remaining, bracket_band / 12)
            tax = round(taxable * rate, 2)
            total_tax += tax
            details.append(f"KES {taxable:,.2f} @ {rate*100:.0f}% = KES {tax:,.2f}")
            remaining -= taxable

        total_tax = round(max(0, total_tax - personal_relief), 2)
        net_pay = round(monthly - total_tax, 2)

        return {
            "gross_monthly": round(monthly, 2),
            "total_tax": total_tax,
            "net_pay": net_pay,
            "personal_relief_applied": personal_relief,
            "brackets": details,
        }
