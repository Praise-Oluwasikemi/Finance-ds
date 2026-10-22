"""Naira formatting for reports and chart labels."""

NAIRA = "₦"


def naira(amount: float, decimals: int = 1) -> str:
    """Readable naira amount: naira(240_512_000) -> '₦240.5m', naira(-60e6) -> '-₦60.0m'."""
    sign = "-" if amount < 0 else ""
    a = abs(amount)
    for size, unit in ((1e12, "trn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if a >= size:
            return f"{sign}{NAIRA}{a / size:,.{decimals}f}{unit}"
    return f"{sign}{NAIRA}{a:,.0f}"
