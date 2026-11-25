"""Runtime logic for the Streamlit app. Reads app_data/ only; no model training libraries."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from sillage.routine import STEPS, Consultation, RoutineResult, build_routine, launch_candidates

ROOT = Path(__file__).resolve().parents[2]
SKIN_LABEL = {"dry": "dry", "oily": "oily", "combination": "combination", "normal": "normal"}
_NUM = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class AppData:
    products: pd.DataFrame
    skin: pd.DataFrame
    returning: pd.DataFrame
    meta: dict
    results: dict
    clients: list[dict]

    @property
    def min_n(self) -> int:
        return int(self.meta["min_group_size"])

    def client(self, cid: str) -> dict:
        return next(c for c in self.clients if c["id"] == cid)


def load(root: Path = ROOT) -> AppData:
    d = root / "app_data"
    with open(root / "clients" / "synthetic_clients.yaml", encoding="utf-8") as f:
        clients = yaml.safe_load(f)["clients"]
    return AppData(
        products=pd.read_parquet(d / "products.parquet"),
        skin=pd.read_parquet(d / "skin_stats.parquet"),
        returning=pd.read_parquet(d / "returning_scores.parquet"),
        meta=json.loads((d / "meta.json").read_text(encoding="utf-8")),
        results=json.loads((d / "results_summary.json").read_text(encoding="utf-8")),
        clients=clients,
    )


@dataclass
class Reason:
    text: str
    numbers: list[float] = field(default_factory=list)


@dataclass
class View:
    path_label: str
    routine: RoutineResult
    reasons: dict[str, list[Reason]]           # product_id -> reasons
    launches: pd.DataFrame
    launch_note: str
    usual_spend: float | None


def guarded(data: AppData, skin_type: str) -> set[str]:
    s = data.skin
    return set(s[(s["skin_type"] == skin_type) & s["guard"]]["product_id"])


def skin_reason(data: AppData, product_id: str, skin_type: str) -> Reason:
    row = data.skin[(data.skin["product_id"] == product_id) & (data.skin["skin_type"] == skin_type)]
    label = SKIN_LABEL.get(skin_type, skin_type)
    if row.empty:
        return Reason(f"Fewer than {data.min_n} {label}-skin reviews: no skin-type figure shown.")
    n, likes = int(row["n"].iat[0]), int(row["likes"].iat[0])
    pct = round(100 * likes / n)
    return Reason(f"Liked by {pct}% of {n} {label}-skin reviewers (4-5 stars).", [pct, n])


def common_reasons(data: AppData, p: pd.Series, c: Consultation) -> list[Reason]:
    out = [skin_reason(data, p["product_id"], c.skin_type)]
    for concern in c.concerns:
        if concern in list(p["concerns"]):
            out.append(Reason(f"Sephora tag: Good for {concern}."))
    return out


def _reason_from_text(text: str) -> Reason:
    return Reason(text, [float(x) for x in _NUM.findall(text)])


def consultation(client: dict, **overrides) -> Consultation:
    c = Consultation(
        skin_type=client["skin_type"], concerns=list(client.get("concerns", [])),
        budget=float(client["budget_usd"]), avoid=list(client.get("avoid", [])),
        usual_spend=client.get("usual_spend_usd"),
    )
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def build_view(data: AppData, client: dict, c: Consultation) -> View:
    p = data.products
    established = p[~p["is_new"]]
    guard = guarded(data, c.skin_type)
    reasons: dict[str, list[Reason]] = {}
    usual = c.usual_spend

    if client["type"] == "returning":
        r = data.returning[data.returning["client_id"] == client["id"]].set_index("product_id")
        info = data.meta["returning_clients"][client["id"]]
        usual = usual or info.get("usual_spend")
        c.usual_spend = usual
        # Popularity breaks ties among products the model scores equally.
        scores = r["routine_score"] + 1e-6 * p.set_index("product_id")["pop_score"].reindex(r.index).fillna(0)
        path_label = f"Returning client: {info['routine_model_label']}"
    else:
        scores = established.set_index("product_id")["pop_score"]
        path_label = ("New client: consultation score (what clients liked most in the past year, "
                      "minus products her skin type clearly dislikes). Learned models need a purchase history.")
        r = None

    routine = build_routine(established, scores, c, guard, data.min_n)
    if routine.picks is not None:
        for _, row in routine.picks.iterrows():
            rs = []
            if r is not None and isinstance(r.loc[row["product_id"], "reason"], str):
                rs.append(_reason_from_text(r.loc[row["product_id"], "reason"]))
            elif pd.notna(row["likes_12m"]):
                n = int(row["likes_12m"])
                rs.append(Reason(f"{n} liked reviews (4-5 stars) in the 12 months to March 2023.", [n]))
            reasons[row["product_id"]] = rs + common_reasons(data, row, c)

    launches, note = launch_candidates(p, c, guard, data.min_n)
    if r is not None and len(launches):
        ls = r["launch_score"].reindex(launches["product_id"]).to_numpy()
        launches = launches.assign(launch_score=ls).sort_values("launch_score", ascending=False)
        note = (f"New at Sephora. Ordered by {data.meta['returning_clients'][client['id']]['launch_model_label']}; "
                "filtered on her stated concerns, skin type and ingredients to avoid.")
        for _, row in launches.head(3).iterrows():
            txt = r.loc[row["product_id"], "launch_reason"]
            reasons[row["product_id"]] = ([_reason_from_text(txt)] if isinstance(txt, str) else []) + common_reasons(data, row, c)
    else:
        for _, row in launches.head(3).iterrows():
            reasons[row["product_id"]] = common_reasons(data, row, c)
    return View(path_label, routine, reasons, launches, note, usual)


def message_facts(data: AppData, client: dict, view: View, language: str):
    from sillage.messaging.guardrails import Facts

    picks = view.routine.picks
    numbers, tags = [], []
    for pid in picks["product_id"]:
        for r in view.reasons.get(pid, []):
            numbers.extend(r.numbers)
            if r.text.startswith("Sephora tag: "):
                tags.append(r.text.removeprefix("Sephora tag: ").rstrip("."))
    return Facts(
        client_name=client["name"].split()[0], language=language, skin_type=client["skin_type"],
        products=[{k: (float(v) if k == "price_usd" else v) for k, v in row.items()}
                  for row in picks[["product_name", "brand_name", "step", "price_usd"]].to_dict("records")],
        total=float(view.routine.total), numbers=numbers, tags=sorted(set(tags)),
    )


def brand_gaps(data: AppData, brand: str) -> pd.DataFrame:
    """For one brand: per skin type and step, in-stock products not flagged for that skin
    type, and how many have a skin-type figure (>= min_n reviews). 0 = a gap."""
    p = data.products[(data.products["brand_name"] == brand) & ~data.products["out_of_stock"]]
    rows = []
    for skin in ["dry", "normal", "combination", "oily"]:
        g = guarded(data, skin)
        with_fig = set(data.skin[data.skin["skin_type"] == skin]["product_id"])
        for step in STEPS:
            sp = p[(p["step"] == step) & ~p["product_id"].isin(g)]
            rows.append({"skin type": skin, "step": step, "products": len(sp),
                         "with skin-type evidence": int(sp["product_id"].isin(with_fig).sum())})
    return pd.DataFrame(rows)
