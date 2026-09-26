"""Project-wide constants: study window, topic taxonomy, market tickers."""

# Final-stage study window (UTC). The end date is exclusive.
EVENT_START = "2026-02-15"
EVENT_END = "2026-04-27"
FREQ = "4h"

# Keyword rules for each oil-market topic. A headline is assigned to the first
# topic in TOPIC_PRIORITY whose keywords match on a word boundary.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "hormuz_shipping": [
        "hormuz",
        "strait of hormuz",
        "persian gulf",
        "tanker",
        "vessel",
        "shipping",
        "shipping lane",
        "blockade",
        "maritime",
        "gulf waters",
    ],
    "military_escalation": [
        "attack",
        "attacks",
        "strike",
        "strikes",
        "missile",
        "missiles",
        "drone",
        "drones",
        "troops",
        "military",
        "war",
        "conflict",
        "retaliation",
        "retaliate",
        "escalation",
        "airstrike",
        "bombing",
        "shelling",
    ],
    "sanctions_exports": [
        "sanction",
        "sanctions",
        "embargo",
        "export",
        "exports",
        "iranian oil",
        "oil exports",
        "crude exports",
        "oil revenue",
        "production",
        "output",
        "supply cut",
    ],
    "diplomacy_ceasefire": [
        "talks",
        "negotiation",
        "negotiations",
        "ceasefire",
        "peace",
        "deal",
        "agreement",
        "diplomacy",
        "diplomatic",
        "truce",
        "resume talks",
        "peace talks",
    ],
    "opec_gulf_supply": [
        "opec",
        "opec+",
        "saudi",
        "saudi arabia",
        "uae",
        "iraq",
        "kuwait",
        "qatar",
        "gulf producers",
        "production increase",
        "production cut",
        "spare capacity",
    ],
    "inventory_eia": [
        "inventory",
        "inventories",
        "stockpile",
        "stockpiles",
        "eia",
        "api",
        "refinery",
        "refineries",
        "gasoline",
        "diesel",
        "crude stocks",
        "drawdown",
        "build",
    ],
    "macro_dollar_fed": [
        "dollar",
        "fed",
        "federal reserve",
        "inflation",
        "interest rate",
        "rates",
        "rate cut",
        "rate hike",
        "yields",
        "treasury",
        "stock market",
        "risk appetite",
    ],
    "demand_china_growth": [
        "demand",
        "china",
        "chinese demand",
        "global growth",
        "recession",
        "slowdown",
        "economic growth",
        "manufacturing",
        "factory activity",
        "asia demand",
    ],
}

TOPIC_PRIORITY: list[str] = [
    "hormuz_shipping",
    "military_escalation",
    "sanctions_exports",
    "diplomacy_ceasefire",
    "inventory_eia",
    "opec_gulf_supply",
    "macro_dollar_fed",
    "demand_china_growth",
]

DEFAULT_TOPIC = "other_oil_relevant"

# Topics most directly tied to the 2026 US-Iran oil-price narrative; these drive
# the final feature set.
CORE_TOPICS: list[str] = [
    "inventory_eia",
    "diplomacy_ceasefire",
    "military_escalation",
    "hormuz_shipping",
    "sanctions_exports",
]

# Yahoo Finance tickers used for price and control features.
TICKERS: dict[str, str] = {
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "NATGAS": "NG=F",
    "SPY": "SPY",
    "UUP": "UUP",
    "VIX": "^VIX",
}

# Columns that must never be used as model inputs.
NON_FEATURE_COLS = ["timestamp_utc", "wti_close", "future_ret_next", "target_direction"]
