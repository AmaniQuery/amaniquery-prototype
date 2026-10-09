"""
Calculator Tool - Safe Mathematical Expression Evaluator

Features:
- Safe expression evaluation (no code injection)
- Support for complex math operations
- Unit conversions (Kenya-specific: KES, acres, hectares)
- Financial calculations (tax, compound interest)
- Percentage and ratio calculations
- LLM-ready tool schema
"""

import math
import re
from typing import Dict, Any, Union, Optional, List
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from loguru import logger


@dataclass
class CalculationResult:
    """Result of a calculation."""

    expression: str
    result: Union[float, int, str, None]
    steps: List[str]
    success: bool
    error: Optional[str] = None


class CalculatorTool:
    """
    Safe mathematical calculator with extended functionality.

    Features:
    - Safe expression evaluation
    - Complex math operations
    - Unit conversions
    - Financial calculations
    - Kenya-specific conversions
    """

    name = "calculator"
    description = (
        "Perform mathematical calculations, unit conversions, and financial computations. "
        "Supports: basic math, percentages, compound interest, tax calculations, "
        "and Kenya-specific conversions (KES, acres, hectares)."
    )

    # Safe operations allowed in eval
    SAFE_OPERATIONS = {
        # Math functions
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        # Built-in
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "int": int,
        "float": float,
    }

    # Unit conversions
    CONVERSIONS = {
        # Length
        "km_to_miles": 0.621371,
        "miles_to_km": 1.60934,
        "m_to_feet": 3.28084,
        "feet_to_m": 0.3048,
        # Area
        "acres_to_hectares": 0.404686,
        "hectares_to_acres": 2.47105,
        "sqm_to_sqft": 10.7639,
        "sqft_to_sqm": 0.092903,
        # Currency (approximate rates - should be fetched for accuracy)
        "usd_to_kes": 129.0,  # Approximate
        "kes_to_usd": 1 / 129.0,
        "gbp_to_kes": 172.77,
        "kes_to_gbp": 1 / 172.77,
        "eur_to_kes": 151.63,
        "kes_to_eur": 1 / 151.63,
    }

    # Kenya tax brackets (2024)
    KENYA_TAX_BRACKETS = [
        (24000, 0.10),  # 10% on first 24,000
        (8333, 0.25),  # 25% on next 8,333
        (467667, 0.30),  # 30% on next 467,667
        (300000, 0.325),  # 32.5% on next 300,000
        (float("inf"), 0.35),  # 35% on remainder
    ]

    def __init__(self, precision: int = 10):
        """
        Initialize calculator.

        Args:
            precision: Decimal precision for results
        """
        self.precision = precision
        logger.info("CalculatorTool initialized")

    def execute(
        self,
        expression: str,
        precision: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression or perform a specific operation.

        Args:
            expression: Mathematical expression to evaluate
            precision: Decimal precision for results
            operation: Specific operation ('convert', 'tax', 'compound_interest', etc.)
            **kwargs: Additional arguments for specific operations

        Returns:
            Calculation result with steps
        """
        precision = precision or self.precision

        try:
            # Handle specific operations
            if operation:
                return self._handle_operation(
                    operation, expression, precision, **kwargs
                )

            # Standard expression evaluation
            result = self._safe_eval(expression, precision)

            return {
                "expression": expression,
                "result": result,
                "formatted": self._format_number(result),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Calculator error for '{expression}': {e}")
            return {
                "expression": expression,
                "result": None,
                "error": str(e),
                "success": False,
            }

    def _safe_eval(self, expression: str, precision: int) -> Union[float, int]:
        """Safely evaluate mathematical expression."""
        # Clean expression
        expression = expression.strip()

        # Replace common symbols
        expression = expression.replace("^", "**")
        expression = expression.replace("×", "*")
        expression = expression.replace("÷", "/")
        expression = expression.replace("%", "/100")

        # Validate expression (only allow safe characters)
        if not re.match(r"^[\d\s\+\-\*\/\.\(\)\,\w]+$", expression):
            raise ValueError(f"Invalid characters in expression: {expression}")

        # Evaluate with restricted namespace
        result = eval(expression, {"__builtins__": {}}, self.SAFE_OPERATIONS)

        # Round to precision
        if isinstance(result, float):
            result = round(result, precision)

        return result

    def _handle_operation(
        self, operation: str, expression: str, precision: int, **kwargs
    ) -> Dict[str, Any]:
        """Handle specific calculation operations."""
        operation = operation.lower()

        if operation == "convert":
            return self._convert_units(
                expression, kwargs.get("from_unit"), kwargs.get("to_unit"), precision
            )

        elif operation == "percentage":
            return self._calculate_percentage(
                float(expression),
                kwargs.get("of"),
                kwargs.get("type", "of"),  # 'of', 'increase', 'decrease'
                precision,
            )

        elif operation == "compound_interest":
            return self._compound_interest(
                principal=float(expression),
                rate=kwargs.get("rate", 0.1),
                time=kwargs.get("time", 1),
                n=kwargs.get("compounds_per_year", 12),
                precision=precision,
            )

        elif operation == "tax" or operation == "paye":
            return self._calculate_kenya_tax(float(expression), precision)

        elif operation == "loan":
            return self._calculate_loan(
                principal=float(expression),
                rate=kwargs.get("rate", 0.14),
                months=kwargs.get("months", 12),
                precision=precision,
            )

        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _convert_units(
        self,
        value: str,
        from_unit: Optional[str],
        to_unit: Optional[str],
        precision: int,
    ) -> Dict[str, Any]:
        """Convert between units."""
        value = float(value)

        if not from_unit or not to_unit:
            raise ValueError("Both from_unit and to_unit are required for conversion")

        conversion_key = f"{from_unit.lower()}_to_{to_unit.lower()}"

        if conversion_key in self.CONVERSIONS:
            result = value * self.CONVERSIONS[conversion_key]
        else:
            raise ValueError(f"Unknown conversion: {conversion_key}")

        result = round(result, precision)

        return {
            "expression": f"{value} {from_unit} to {to_unit}",
            "result": result,
            "formatted": f"{value} {from_unit} = {result} {to_unit}",
            "success": True,
        }

    def _calculate_percentage(
        self,
        percentage: float,
        of_value: Optional[float],
        calc_type: str,
        precision: int,
    ) -> Dict[str, Any]:
        """Calculate percentages."""
        if calc_type == "of" and of_value is not None:
            result = (percentage / 100) * of_value
            formatted = f"{percentage}% of {of_value} = {round(result, precision)}"
        elif calc_type == "increase" and of_value is not None:
            result = of_value * (1 + percentage / 100)
            formatted = f"{of_value} + {percentage}% = {round(result, precision)}"
        elif calc_type == "decrease" and of_value is not None:
            result = of_value * (1 - percentage / 100)
            formatted = f"{of_value} - {percentage}% = {round(result, precision)}"
        else:
            raise ValueError("Invalid percentage calculation parameters")

        return {
            "expression": f"{percentage}% {calc_type} {of_value}",
            "result": round(result, precision),
            "formatted": formatted,
            "success": True,
        }

    def _compound_interest(
        self, principal: float, rate: float, time: float, n: int, precision: int
    ) -> Dict[str, Any]:
        """Calculate compound interest."""
        # A = P(1 + r/n)^(nt)
        amount = principal * (1 + rate / n) ** (n * time)
        interest = amount - principal

        return {
            "expression": f"P={principal}, r={rate*100}%, t={time} years, n={n}",
            "result": round(amount, precision),
            "principal": principal,
            "interest": round(interest, precision),
            "formatted": f"Final Amount: KES {amount:,.2f} (Interest: KES {interest:,.2f})",
            "success": True,
        }

    def _calculate_kenya_tax(
        self, monthly_income: float, precision: int
    ) -> Dict[str, Any]:
        """Calculate Kenya PAYE tax."""
        tax = 0
        remaining = monthly_income
        breakdown = []

        for bracket_amount, rate in self.KENYA_TAX_BRACKETS:
            if remaining <= 0:
                break

            taxable = min(remaining, bracket_amount)
            tax_amount = taxable * rate
            tax += tax_amount
            remaining -= taxable

            breakdown.append(
                {
                    "bracket": f"KES {taxable:,.0f}",
                    "rate": f"{rate * 100}%",
                    "tax": f"KES {tax_amount:,.2f}",
                }
            )

        # Personal relief (2,400 per month)
        relief = 2400
        net_tax = max(0, tax - relief)

        return {
            "expression": f"Monthly income: KES {monthly_income:,.2f}",
            "result": round(net_tax, precision),
            "gross_tax": round(tax, precision),
            "relief": relief,
            "net_tax": round(net_tax, precision),
            "net_income": round(monthly_income - net_tax, precision),
            "breakdown": breakdown,
            "formatted": f"PAYE Tax: KES {net_tax:,.2f} (Net Income: KES {monthly_income - net_tax:,.2f})",
            "success": True,
        }

    def _calculate_loan(
        self, principal: float, rate: float, months: int, precision: int
    ) -> Dict[str, Any]:
        """Calculate loan repayment (reducing balance)."""
        monthly_rate = rate / 12

        # EMI formula: [P × r × (1+r)^n] / [(1+r)^n - 1]
        if monthly_rate > 0:
            emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / (
                (1 + monthly_rate) ** months - 1
            )
        else:
            emi = principal / months

        total_payment = emi * months
        total_interest = total_payment - principal

        return {
            "expression": f"Loan: KES {principal:,.2f} at {rate*100}% for {months} months",
            "result": round(emi, precision),
            "principal": principal,
            "monthly_payment": round(emi, precision),
            "total_payment": round(total_payment, precision),
            "total_interest": round(total_interest, precision),
            "formatted": f"Monthly Payment: KES {emi:,.2f} (Total Interest: KES {total_interest:,.2f})",
            "success": True,
        }

    def _format_number(self, value: Union[float, int]) -> str:
        """Format number for display."""
        if isinstance(value, float):
            if value.is_integer():
                return f"{int(value):,}"
            return f"{value:,.{self.precision}f}".rstrip("0").rstrip(".")
        return f"{value:,}"

    def get_tool_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression or value to calculate",
                    },
                    "operation": {
                        "type": "string",
                        "enum": [
                            "convert",
                            "percentage",
                            "compound_interest",
                            "tax",
                            "loan",
                        ],
                        "description": "Specific operation to perform",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit for conversion",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Target unit for conversion",
                    },
                    "rate": {
                        "type": "number",
                        "description": "Interest rate (decimal, e.g., 0.14 for 14%)",
                    },
                    "time": {
                        "type": "number",
                        "description": "Time period in years",
                    },
                    "months": {
                        "type": "integer",
                        "description": "Loan duration in months",
                    },
                },
                "required": ["expression"],
            },
        }

    def list_conversions(self) -> List[str]:
        """List available unit conversions."""
        return list(self.CONVERSIONS.keys())
