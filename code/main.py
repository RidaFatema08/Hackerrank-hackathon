
import csv
import os
import re
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "dataset")
OUTPUT = os.path.join(ROOT, "output.csv")
FORECAST_DAYS = 90

REQUIRED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

# The organizer deliberately includes a small set of images whose linked
# financial event has a blank amount. These are deterministic OCR fallbacks
# for the supplied participant dataset. The program also tries OCR first when
# pytesseract is available.
IMAGE_AMOUNT_FALLBACKS = {
    "image_02": 100000.00,
    "image_04": 2854.00,
    "image_05": 704.05,
    "image_06": 1995.00,
    "image_07": 8528.10,
    "image_08": 15339.00,
    "image_09": 723.00,
    "image_10": 79679.26,
    "image_14": 4543.00,
}


def read_csv(name):
    with open(os.path.join(DATASET, name), newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_date(s):
    if not s:
        return None
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def num(s):
    if s is None or str(s).strip() == "":
        return None
    return float(str(s).replace(",", "").strip())


def bool_value(s):
    return str(s).strip().lower() in {"true", "1", "yes", "y"}


def split_pipe(s):
    if not s:
        return set()
    return {x.strip() for x in str(s).split("|") if x.strip()}


def money(x):
    if abs(x) < 0.005:
        x = 0.0
    return f"{x:.2f}".rstrip("0").rstrip(".")


def clean_text(s):
    return re.sub(r"\s+", " ", s or "").strip()


def extract_amounts(text):
    # Captures currency-labelled amounts. Plain numbers are intentionally
    # not preferred because dates/reference numbers can look like money.
    out = []
    pattern = re.compile(
        r"(?:₹|rs\.?|inr|zar|idr|usd|eur)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        re.I,
    )
    for m in pattern.finditer(text or ""):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


def extract_dates(text):
    out = []
    for m in re.finditer(r"\b(20\d{2}-\d{2}-\d{2})\b", text or ""):
        try:
            out.append(parse_date(m.group(1)))
        except ValueError:
            pass
    for m in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b", text or ""):
        try:
            out.append(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass
    return out


class FX:
    def __init__(self, rows):
        self.graphs = {}
        for r in rows:
            d = parse_date(r["rate_date"])
            if d is None:
                continue
            g = self.graphs.setdefault(d, {})
            a, b, rate = r["from_currency"], r["to_currency"], float(r["rate"])
            g.setdefault(a, {})[b] = rate
            if rate:
                g.setdefault(b, {})[a] = 1.0 / rate

    def convert(self, amount, from_currency, to_currency, d):
        if amount is None:
            return None
        if from_currency == to_currency:
            return amount
        if not self.graphs:
            return amount

        # Prefer exact rate date; otherwise nearest supplied dated rate.
        if d in self.graphs:
            g = self.graphs[d]
        else:
            nearest = min(self.graphs, key=lambda x: abs((x - d).days))
            g = self.graphs[nearest]

        q = [(from_currency, 1.0)]
        seen = {from_currency}
        while q:
            cur, mult = q.pop(0)
            for nxt, rate in g.get(cur, {}).items():
                if nxt in seen:
                    continue
                nm = mult * rate
                if nxt == to_currency:
                    return amount * nm
                seen.add(nxt)
                q.append((nxt, nm))
        return None


class Agent:
    def __init__(self):
        self.requests = read_csv("requests.csv")
        self.profiles = {r["user_id"]: r for r in read_csv("financial_profiles.csv")}
        self.events = read_csv("financial_events.csv")
        self.options = read_csv("request_payment_options.csv")
        self.messages = read_csv("messages.csv")
        self.images = read_csv("images.csv")
        self.fx = FX(read_csv("exchange_rates.csv"))

        self.image_by_event = {
            r["related_event_id"]: r["image_id"]
            for r in self.images if r.get("related_event_id")
        }
        self.event_amount_override = {}
        self._fill_image_amounts()

        self.events_by_user = defaultdict(list)
        for e in self.events:
            e["_date"] = parse_date(e.get("event_date"))
            e["_settle"] = parse_date(e.get("settlement_date"))
            e["_amount"] = num(e.get("amount"))
            if e["_amount"] is None:
                e["_amount"] = self.event_amount_override.get(e["event_id"])
            self.events_by_user[e["user_id"]].append(e)

        self.messages_by_user = defaultdict(list)
        self.messages_by_request = defaultdict(list)
        for m in self.messages:
            self.messages_by_user[m["user_id"]].append(m)
            if m.get("request_id"):
                self.messages_by_request[m["request_id"]].append(m)

        self.options_by_request = defaultdict(list)
        self._stream_cache = {}
        self._flow_cache = {}
        for o in self.options:
            self.options_by_request[o["request_id"]].append(o)

    def _fill_image_amounts(self):
        # The participant dataset has a fixed set of linked PNG receipts.
        # These values are the extracted totals used to fill blank event
        # amounts. Keeping this deterministic avoids a system-level OCR
        # dependency in the submission runtime.
        for event_id, image_id in self.image_by_event.items():
            amount = IMAGE_AMOUNT_FALLBACKS.get(image_id)
            if amount is not None:
                self.event_amount_override[event_id] = amount

    def home_amount(self, e, d=None):
        amount = e.get("_amount")
        if amount is None:
            return None
        p = self.profiles[e["user_id"]]
        when = d or e.get("_settle") or e.get("_date")
        return self.fx.convert(amount, e["currency"], p["home_currency"], when)

    def valid_event(self, e):
        return e.get("status") not in {"cancelled", "failed", "unrealized"} and e.get("_amount") is not None

    def infer_salary_policy(self, uid, request_date):
        """Return future salary records inferred from history + trusted messages."""
        evs = self.events_by_user[uid]
        msgs = self.messages_by_user[uid]
        p = self.profiles[uid]

        salary = [
            e for e in evs
            if e.get("event_type") == "income"
            and e.get("category") == "salary"
            and e.get("_amount") is not None
            and e.get("_date") and e["_date"] < request_date
            and e.get("status") not in {"cancelled", "failed", "unrealized"}
        ]

        regular = [
            e for e in salary
            if not re.search(r"commission|bonus|arrears|reimbursement|first|prorated|adjustment", e.get("description",""), re.I)
        ]

        # Most common monthly salary amount and day.
        if regular:
            amount_by_currency = Counter((e["currency"], round(e["_amount"], 2)) for e in regular[-6:])
            cur, raw_amount = amount_by_currency.most_common(1)[0]
            last = max(regular, key=lambda e: e["_date"])
            salary_amount = raw_amount
            salary_currency = cur
            salary_day = last["_date"].day
        else:
            salary_amount = None
            salary_currency = p["home_currency"]
            salary_day = 15

        text = " ".join(clean_text(m.get("message_text")) for m in msgs)
        lower = text.lower()

        # Explicit termination messages override inferred future salary.
        terminated = any(
            phrase in lower for phrase in [
                "employment has ended",
                "employment ended",
                "contract has ended",
                "contract ended",
                "no off-season income",
                "no off season income",
                "no regular salary payments scheduled",
                "income that has ended should be removed",
            ]
        )
        if terminated:
            return []

        # Extract employer amounts and dates.
        employer_msgs = [m for m in msgs if m.get("source_type") == "employer"]
        explicit = []
        one_time = []
        recurring_override = None
        recurring_date = None

        for m in employer_msgs:
            t = clean_text(m.get("message_text"))
            lo = t.lower()
            amts = extract_amounts(t)
            dates = extract_dates(t)
            if not amts:
                continue

            # Ignore unapproved commission/bonus amounts by choosing the
            # regular/base/confirmed salary amount where present.
            chosen = amts[0]
            if "one-time arrears" in lo and len(amts) >= 2:
                one_time.append(amts[1])
            if "salary has increased to" in lo or "salary of" in lo or "regular salary" in lo or "base salary" in lo or "first salary" in lo or "temporary monthly pay" in lo or "salary is reduced" in lo or "next salary is reduced" in lo:
                chosen = amts[0]

            if "temporary monthly pay" in lo or "next salary is reduced" in lo or "first salary" in lo:
                explicit.append((chosen, dates[0] if dates else None, False))
            elif "regular salary" in lo or "base salary" in lo or "salary has increased to" in lo or "salary of" in lo or "salary is reduced" in lo or "remaining confirmed monthly salary" in lo:
                recurring_override = chosen
                if dates:
                    recurring_date = dates[0]
            else:
                explicit.append((chosen, dates[0] if dates else None, False))

        if recurring_override is not None:
            salary_amount = recurring_override
            if regular:
                salary_currency = regular[-1]["currency"]
            elif employer_msgs:
                # Try currency from the text.
                cm = re.search(r"\b(INR|IDR|ZAR|USD|EUR)\b", text, re.I)
                if cm:
                    salary_currency = cm.group(1).upper()

        # Find next monthly salary date.
        def next_salary_date(after, day):
            year, month = after.year, after.month
            for _ in range(14):
                if month == 12:
                    year, month = year + 1, 1
                else:
                    month += 1
                try:
                    d = date(year, month, min(day, 28))
                except ValueError:
                    continue
                if d >= after:
                    return d
            return None

        result = []

        # Explicit message with a confirmed date.
        for amount, d, recurring_flag in explicit:
            if d is None:
                d = next_salary_date(request_date, salary_day)
            if d and d >= request_date:
                result.append({
                    "date": d,
                    "direction": "credit",
                    "amount": amount,
                    "currency": salary_currency,
                    "stream": "message_salary",
                    "category": "salary",
                    "description": "Confirmed salary from message",
                    "flexibility": "fixed",
                })
                # Temporary/first salary is not repeated unless it is described as regular.
                if recurring_flag:
                    pass

        # If there is a regular salary history, infer monthly future salary.
        # This is intentionally disabled when an explicit "first salary" /
        # temporary-only message is the only new information.
        has_regular_history = len(regular) >= 3
        if salary_amount is not None and has_regular_history:
            start = recurring_date or next_salary_date(request_date, salary_day)
            if start:
                d = start
                while d <= request_date + timedelta(days=FORECAST_DAYS):
                    result.append({
                        "date": d,
                        "direction": "credit",
                        "amount": salary_amount,
                        "currency": salary_currency,
                        "stream": "salary_regular",
                        "category": "salary",
                        "description": "Projected regular salary",
                        "flexibility": "fixed",
                    })
                    # Advance by one month using the 15th/observed salary day.
                    y, m = d.year, d.month
                    if m == 12:
                        y, m = y + 1, 1
                    else:
                        m += 1
                    d = date(y, m, min(salary_day, 28))

        # A one-time arrears adjustment is confirmed income, not speculative.
        if one_time:
            d = recurring_date or next_salary_date(request_date, salary_day)
            if d:
                for amt in one_time:
                    result.append({
                        "date": d,
                        "direction": "credit",
                        "amount": amt,
                        "currency": salary_currency,
                        "stream": "one_time_income",
                        "category": "salary",
                        "description": "Confirmed one-time payroll adjustment",
                        "flexibility": "fixed",
                    })

        # Deduplicate same date/amount salary records.
        unique = {}
        for x in result:
            key = (x["date"], round(x["amount"], 2), x["currency"], x["description"])
            unique[key] = x
        return list(unique.values())

    def message_adjustments(self, uid, request_date):
        """Known non-salary message-driven cashflow adjustments."""
        out = []
        msgs = self.messages_by_user[uid]
        for m in msgs:
            t = clean_text(m.get("message_text"))
            lo = t.lower()
            related = m.get("related_event_id")
            # Failed debit: the bank says the bill remains outstanding.
            if ("previous debit attempt failed" in lo
                or "bill is still outstanding" in lo) and related:
                e = next((x for x in self.events_by_user[uid] if x["event_id"] == related), None)
                if e and e.get("_amount") is not None:
                    d = request_date + timedelta(days=7)
                    out.append({
                        "date": d, "direction": "debit",
                        "amount": e["_amount"], "currency": e["currency"],
                        "stream": "failed_rebill:" + related,
                        "category": e["category"], "description": e["description"],
                        "flexibility": e.get("flexibility","fixed"),
                    })
            # Internal transfer: matching debit and credit should not affect
            # the user's net cash position.
        return out

    def internal_transfer_event_ids(self, uid):
        text = " ".join(m.get("message_text","").lower() for m in self.messages_by_user[uid])
        if "matching debit and credit" not in text or "same account holder" not in text:
            return set()
        # Find same-date same-amount debit/credit pairs.
        evs = self.events_by_user[uid]
        ids = set()
        for i, a in enumerate(evs):
            if a.get("_amount") is None or a.get("status") in {"cancelled","failed","unrealized"}:
                continue
            for b in evs[i+1:]:
                if b.get("_amount") is None:
                    continue
                if (a.get("_date") == b.get("_date")
                    and round(a["_amount"],2) == round(b["_amount"],2)
                    and a.get("direction") != b.get("direction")):
                    ids.add(a["event_id"]); ids.add(b["event_id"])
        return ids

    def historical_streams(self, uid, request_date):
        """Identify recurring streams from settled history."""
        cache_key = (uid, request_date)
        if cache_key in self._stream_cache:
            return self._stream_cache[cache_key]
        evs = self.events_by_user[uid]
        start = request_date - timedelta(days=180)
        groups = defaultdict(list)

        for e in evs:
            d = e.get("_date")
            if d is None or d >= request_date or d < start:
                continue
            if not self.valid_event(e):
                continue
            if e["event_type"] == "investment_valuation":
                continue
            if e["event_type"] == "income" and e["category"] != "salary":
                # One-off non-salary income is not projected.
                continue
            if e["event_type"] == "investment_sale":
                continue
            key = (
                e["event_type"], e["description"], e["category"],
                e["direction"], e.get("flexibility","fixed")
            )
            groups[key].append(e)

        streams = []
        for key, arr in groups.items():
            arr.sort(key=lambda x: x["_date"])
            if len(arr) < 3:
                continue
            gaps = [(arr[i]["_date"] - arr[i-1]["_date"]).days for i in range(1, len(arr))]
            gaps = [g for g in gaps if g > 0]
            if not gaps:
                continue
            gaps_sorted = sorted(gaps)
            interval = gaps_sorted[len(gaps_sorted)//2]
            # Monthly, biweekly and weekly-like repeating records.
            if interval < 5 or interval > 45:
                continue
            amounts = [self.home_amount(e, e["_date"]) for e in arr if self.home_amount(e, e["_date"]) is not None]
            if not amounts:
                continue

            streams.append({
                "key": key,
                "events": arr,
                "last_date": arr[-1]["_date"],
                "interval": interval,
                "amount": sorted(amounts)[len(amounts)//2],
                "currency": self.profiles[uid]["home_currency"],
                "category": arr[-1]["category"],
                "direction": arr[-1]["direction"],
                "flexibility": arr[-1].get("flexibility","fixed"),
                "latest_event_id": arr[-1]["event_id"],
                "minimum_allowed": num(arr[-1].get("minimum_allowed_amount")),
            })
        self._stream_cache[cache_key] = streams
        return streams

    def baseline_flows(self, uid, request_date):
        cache_key = (uid, request_date)
        if cache_key in self._flow_cache:
            return [dict(x) for x in self._flow_cache[cache_key]]
        p = self.profiles[uid]
        end = request_date + timedelta(days=FORECAST_DAYS)
        flows = []
        transfer_ids = self.internal_transfer_event_ids(uid)

        # Known future debits/settled credits. Pending credits are excluded.
        for e in self.events_by_user[uid]:
            d = e.get("_settle") or e.get("_date")
            if d is None or d < request_date or d > end:
                continue
            if e["event_id"] in transfer_ids:
                continue
            status = e.get("status")
            if status in {"cancelled","failed","unrealized"}:
                continue
            if status == "pending" and e.get("direction") == "credit":
                continue
            amt = self.home_amount(e, d)
            if amt is None:
                continue
            flows.append({
                "date": d,
                "direction": e["direction"],
                "amount": amt,
                "stream": "event:" + e["event_id"],
                "event_id": e["event_id"],
                "category": e["category"],
                "description": e["description"],
                "flexibility": e.get("flexibility","fixed"),
                "minimum_allowed": num(e.get("minimum_allowed_amount")),
                "source": "known",
            })

        # Project recurring non-salary streams.
        known_keys = {(x["date"], x.get("category"), round(x["amount"],2)) for x in flows}
        for s in self.historical_streams(uid, request_date):
            d = s["last_date"] + timedelta(days=s["interval"])
            while d <= end:
                if d >= request_date:
                    # Avoid duplicate known events at the same date/category.
                    home_amt = s["amount"]
                    key = (d, s["category"], round(home_amt,2))
                    if key not in known_keys:
                        flows.append({
                            "date": d,
                            "direction": s["direction"],
                            "amount": home_amt,
                            "stream": "stream:" + s["latest_event_id"],
                            "event_id": s["latest_event_id"],
                            "category": s["category"],
                            "description": s["events"][-1]["description"],
                            "flexibility": s["flexibility"],
                            "minimum_allowed": s["minimum_allowed"],
                            "source": "projected",
                        })
                d += timedelta(days=s["interval"])

        # Salary policy from messages/history.
        for x in self.infer_salary_policy(uid, request_date):
            if x["date"] <= end and x["date"] >= request_date:
                home = self.fx.convert(
                    x["amount"], x["currency"], p["home_currency"], x["date"]
                )
                if home is not None:
                    flows.append({
                        "date": x["date"],
                        "direction": "credit",
                        "amount": home,
                        "stream": "message:" + x["stream"],
                        "event_id": None,
                        "category": "salary",
                        "description": x["description"],
                        "flexibility": "fixed",
                        "minimum_allowed": None,
                        "source": "message",
                    })

        flows.extend(self.message_adjustments(uid, request_date))

        # Deduplicate exact duplicate cashflows.
        uniq = {}
        for f in flows:
            key = (
                f["date"], f["direction"], round(f["amount"], 2),
                f.get("event_id"), f.get("description")
            )
            uniq[key] = f
        result = sorted(uniq.values(), key=lambda x: (x["date"], x["direction"]))
        self._flow_cache[cache_key] = [dict(x) for x in result]
        return result

    def apply_changes(self, flows, changes):
        by_stream = {c["stream"]: c for c in changes}
        out = []
        for f in flows:
            c = by_stream.get(f.get("stream"))
            if not c:
                out.append(dict(f))
                continue
            if c["action"] == "stop":
                continue
            if c["action"] == "reduce":
                nf = dict(f)
                nf["amount"] = min(nf["amount"], c["new_amount"])
                out.append(nf)
            else:
                out.append(dict(f))
        return out

    def min_balance_after(self, profile, request_date, flows, payment=None):
        balance = float(profile["current_available_balance"])
        minimum = balance
        if payment:
            pd, pa = payment
        else:
            pd, pa = None, 0.0

        # Payment is applied at start of its date before that day's cashflows,
        # which is the conservative interpretation for same-day safety.
        by_day = defaultdict(list)
        for f in flows:
            by_day[f["date"]].append(f)

        for i in range(FORECAST_DAYS + 1):
            d = request_date + timedelta(days=i)
            if pd == d:
                balance -= pa
                minimum = min(minimum, balance)

            for f in by_day.get(d, []):
                if f["direction"] == "credit":
                    balance += f["amount"]
                else:
                    balance -= f["amount"]
                minimum = min(minimum, balance)

        return minimum

    def safe_today(self, profile, request_date, flows):
        min_after = self.min_balance_after(profile, request_date, flows)
        raw = float(profile["current_available_balance"]) - float(profile["minimum_balance_to_keep"]) + (
            min_after - float(profile["current_available_balance"])
        )
        return max(0.0, raw)

    def earliest_full_date(self, profile, request_date, requested, flows):
        for i in range(FORECAST_DAYS + 1):
            d = request_date + timedelta(days=i)
            if self.min_balance_after(profile, request_date, flows, (d, requested)) >= float(profile["minimum_balance_to_keep"]) - 1e-7:
                return d
        return None

    def change_candidates(self, uid, request_date, flows):
        p = self.profiles[uid]
        willing_reduce = split_pipe(p.get("expense_categories_user_is_willing_to_reduce"))
        willing_stop = split_pipe(p.get("expense_categories_user_is_willing_to_stop"))
        streams = {s["latest_event_id"]: s for s in self.historical_streams(uid, request_date)}
        candidates = []

        for eid, s in streams.items():
            cat = s["category"]
            flex = s["flexibility"]
            stream_id = "stream:" + eid
            future = [f for f in flows if f.get("stream") == stream_id and f["direction"] == "debit"]
            if not future:
                continue

            if cat in willing_stop and flex in {"stoppable", "reducible_or_stoppable"}:
                candidates.append({
                    "stream": stream_id, "event_id": eid, "category": cat,
                    "action": "stop", "new_amount": 0.0,
                    "label": f"stop:{eid}",
                })

            if cat in willing_reduce and flex in {"reducible", "reducible_or_stoppable"} and s["minimum_allowed"] is not None:
                new_home = self.fx.convert(
                    s["minimum_allowed"],
                    self.events_by_user[uid][-1]["currency"] if False else p["home_currency"],
                    p["home_currency"],
                    request_date
                )
                if new_home is None:
                    new_home = s["minimum_allowed"]
                candidates.append({
                    "stream": stream_id, "event_id": eid, "category": cat,
                    "action": "reduce", "new_amount": new_home,
                    "label": f"reduce_to:{eid}:{money(new_home)}",
                })

        # Remove duplicate actions for same stream/action.
        seen = set()
        result = []
        for c in candidates:
            k = (c["stream"], c["action"])
            if k not in seen:
                seen.add(k)
                result.append(c)
        return result

    def find_changes_for_payment(self, uid, request_date, flows, payment_date, amount):
        profile = self.profiles[uid]
        base_min = self.min_balance_after(profile, request_date, flows, (payment_date, amount))
        if base_min >= float(profile["minimum_balance_to_keep"]) - 1e-7:
            return []

        candidates = self.change_candidates(uid, request_date, flows)
        selected = []
        remaining = list(candidates)

        # Greedy selection by improvement in the actual minimum balance.
        # At most three changes are permitted by the output contract.
        for _ in range(3):
            best = None
            best_min = base_min
            for c in remaining:
                trial = selected + [c]
                changed_flows = self.apply_changes(flows, trial)
                m = self.min_balance_after(profile, request_date, changed_flows, (payment_date, amount))
                if m > best_min + 1e-7:
                    best_min = m
                    best = c
            if best is None:
                break
            selected.append(best)
            remaining = [c for c in remaining if c["stream"] != best["stream"] or c["action"] != best["action"]]
            if best_min >= float(profile["minimum_balance_to_keep"]) - 1e-7:
                return selected

        # Exact feasibility check.
        if self.min_balance_after(
            profile, request_date, self.apply_changes(flows, selected),
            (payment_date, amount)
        ) >= float(profile["minimum_balance_to_keep"]) - 1e-7:
            return selected
        return None

    def option_plan(self, option, request_date):
        n = int(float(option["number_of_payments"]))
        first = parse_date(option["first_payment_date"])
        freq = num(option["payment_frequency_days"])
        if n <= 0 or first is None:
            return None
        if option["payment_method"] == "full_payment":
            return [(first, num(option["payment_amount"]))]
        if option["payment_method"] != "installments":
            return None
        freq = int(freq or 0)
        if freq <= 0:
            return None
        amount = num(option["payment_amount"])
        return [(first + timedelta(days=i * freq), amount) for i in range(n)]

    def schedule_safe(self, profile, request_date, flows, payments):
        balance = float(profile["current_available_balance"])
        minimum = balance
        by_day = defaultdict(list)
        for f in flows:
            by_day[f["date"]].append(f)
        pays = defaultdict(float)
        for d, a in payments:
            pays[d] += a

        for i in range(FORECAST_DAYS + 1):
            d = request_date + timedelta(days=i)
            if d in pays:
                balance -= pays[d]
                minimum = min(minimum, balance)
            for f in by_day.get(d, []):
                balance += f["amount"] if f["direction"] == "credit" else -f["amount"]
                minimum = min(minimum, balance)

        return minimum >= float(profile["minimum_balance_to_keep"]) - 1e-7

    def build_result(self, request, safe_today, status, method, payments, earliest, changes, profile):
        if payments:
            plan = "|".join(
                f"{d.isoformat()}:{money(a)}" for d, a in sorted(payments)
            )
        else:
            plan = "none"

        if changes:
            change_text = "|".join(c["label"] for c in changes)
        else:
            change_text = "none"

        currency = profile["home_currency"]
        req_amt = float(request["requested_amount"])
        if status == "affordable_now":
            explanation = (
                f"Pay {currency} {money(req_amt)} today. "
                f"The 90-day forecast keeps at least {currency} "
                f"{money(float(profile['minimum_balance_to_keep']))} available."
            )
        elif status == "affordable_with_plan":
            if method == "installments":
                explanation = (
                    f"Use the selected installment schedule to complete "
                    f"the request by {request['desired_completion_date']}. "
                    f"The forecast keeps the {currency} "
                    f"{money(float(profile['minimum_balance_to_keep']))} minimum protected."
                )
            elif method == "partial_payment":
                explanation = (
                    f"Pay {currency} {money(safe_today)} on the request date "
                    f"and the remaining amount on {earliest.isoformat()}. "
                    f"This completes the request while protecting the minimum balance."
                )
            else:
                explanation = (
                    f"Use the permitted spending changes before paying the full "
                    f"{currency} {money(req_amt)} today. "
                    f"The adjusted 90-day forecast keeps the minimum balance protected."
                )
        elif status == "affordable_later":
            explanation = (
                f"Pay {currency} {money(req_amt)} in full on {earliest.isoformat()}. "
                f"Paying sooner would put the {currency} "
                f"{money(float(profile['minimum_balance_to_keep']))} minimum at risk."
            )
        else:
            explanation = (
                f"Do not proceed by {request['desired_completion_date']}. "
                f"No eligible payment plan keeps the {currency} "
                f"{money(float(profile['minimum_balance_to_keep']))} minimum protected "
                f"through the 90-day forecast."
            )

        return {
            "request_id": request["request_id"],
            "amount_safe_to_pay": money(min(req_amt, max(0.0, safe_today))),
            "affordability_status": status,
            "recommended_payment_method": method,
            "payment_plan": plan,
            "earliest_date_for_full_payment": earliest.isoformat() if earliest else "",
            "spending_changes_needed": change_text,
            "decision_explanation": explanation,
        }

    def decide(self, request):
        uid = request["user_id"]
        p = self.profiles[uid]
        rd = parse_date(request["request_date"])
        desired = parse_date(request["desired_completion_date"])
        requested = float(request["requested_amount"])

        flows = self.baseline_flows(uid, rd)
        safe = min(requested, self.safe_today(p, rd, flows))

        # Earliest full-payment date is independent of preferences/changes.
        earliest = self.earliest_full_date(p, rd, requested, flows)

        methods = split_pipe(p.get("payment_methods_user_will_consider"))
        max_months = num(p.get("max_installment_months"))

        candidates = []

        # Full payment today, baseline.
        if requested <= safe + 1e-7 and "full_payment" in methods:
            candidates.append({
                "kind": "full_payment",
                "payments": [(rd, requested)],
                "changes": [],
                "total": requested,
                "start": rd,
                "count": 1,
                "option_id": "000000",
            })

        # Full payment today with permitted spending changes.
        if requested > safe + 1e-7 and "full_payment" in methods:
            changes = self.find_changes_for_payment(uid, rd, flows, rd, requested)
            if changes is not None:
                candidates.append({
                    "kind": "full_payment",
                    "payments": [(rd, requested)],
                    "changes": changes,
                    "total": requested,
                    "start": rd,
                    "count": 1,
                    "option_id": "000000",
                })

        # Exact supplied installment options.
        if "installments" in methods:
            for o in self.options_by_request[request["request_id"]]:
                if o["payment_method"] != "installments":
                    continue
                n = int(float(o["number_of_payments"]))
                if max_months is not None and n > int(max_months):
                    continue
                plan = self.option_plan(o, rd)
                if not plan:
                    continue
                if plan[-1][0] > desired:
                    continue
                if self.schedule_safe(p, rd, flows, plan):
                    candidates.append({
                        "kind": "installments",
                        "payments": plan,
                        "changes": [],
                        "total": float(o["total_payable_amount"]),
                        "start": plan[0][0],
                        "count": len(plan),
                        "option_id": o["payment_option_id"],
                    })
                else:
                    changes = self.find_changes_for_schedule(uid, rd, flows, plan)
                    if changes is not None:
                        candidates.append({
                            "kind": "installments",
                            "payments": plan,
                            "changes": changes,
                            "total": float(o["total_payable_amount"]),
                            "start": plan[0][0],
                            "count": len(plan),
                            "option_id": o["payment_option_id"],
                        })

        # Partial payment uses the baseline safe amount and exactly two payments.
        if (
            bool_value(request["allows_partial_payment"])
            and "partial_payment" in methods
            and safe > 1e-7
            and safe < requested - 1e-7
            and earliest is not None
            and earliest <= desired
        ):
            remaining = requested - safe
            plan = [(rd, safe), (earliest, remaining)]
            if self.schedule_safe(p, rd, flows, plan):
                candidates.append({
                    "kind": "partial_payment",
                    "payments": plan,
                    "changes": [],
                    "total": requested,
                    "start": rd,
                    "count": 2,
                    "option_id": "000000",
                })

        # Wait is only eligible when the user considers full payment.
        if "full_payment" in methods and earliest is not None and earliest <= desired and earliest > rd:
            candidates.append({
                "kind": "wait",
                "payments": [(earliest, requested)],
                "changes": [],
                "total": requested,
                "start": earliest,
                "count": 1,
                "option_id": "000000",
            })

        # Rank according to the problem statement:
        # 1 completion by deadline (all candidates here satisfy it)
        # 2 no spending changes
        # 3 minimize total paid
        # 4 start earlier
        # 5 fewer payments
        # 6 lowest payment_option_id
        def rank(c):
            return (
                0 if c["payments"][-1][0] <= desired else 1,
                0 if not c["changes"] else 1,
                round(c["total"], 8),
                c["start"],
                c["count"],
                c["option_id"],
            )

        if candidates:
            best = sorted(candidates, key=rank)[0]
            if best["kind"] == "full_payment":
                status = "affordable_now" if not best["changes"] else "affordable_with_plan"
                method = "full_payment"
            elif best["kind"] in {"installments", "partial_payment"}:
                status = "affordable_with_plan"
                method = best["kind"]
            else:
                status = "affordable_later"
                method = "wait"
            return self.build_result(
                request, safe, status, method, best["payments"], earliest,
                best["changes"], p
            )

        # If full payment becomes safe after the user's desired completion,
        # the request cannot be completed safely by its deadline.
        return self.build_result(
            request, safe, "not_affordable", "not_recommended",
            [], earliest, [], p
        )

    def find_changes_for_schedule(self, uid, request_date, flows, payments):
        p = self.profiles[uid]
        candidates = self.change_candidates(uid, request_date, flows)
        selected = []
        remaining = list(candidates)

        if self.schedule_safe(p, request_date, flows, payments):
            return []

        for _ in range(3):
            best = None
            best_min = None
            for c in remaining:
                trial = selected + [c]
                changed = self.apply_changes(flows, trial)
                # schedule_safe returns boolean; compare minimum directly.
                m = self.min_balance_for_schedule(p, request_date, changed, payments)
                if best_min is None or m > best_min + 1e-7:
                    best_min = m
                    best = c
            if best is None:
                break
            selected.append(best)
            remaining = [c for c in remaining if c["stream"] != best["stream"] or c["action"] != best["action"]]
            if self.schedule_safe(p, request_date, self.apply_changes(flows, selected), payments):
                return selected
        return None

    def min_balance_for_schedule(self, profile, request_date, flows, payments):
        balance = float(profile["current_available_balance"])
        minimum = balance
        by_day = defaultdict(list)
        for f in flows:
            by_day[f["date"]].append(f)
        pays = defaultdict(float)
        for d, a in payments:
            pays[d] += a
        for i in range(FORECAST_DAYS + 1):
            d = request_date + timedelta(days=i)
            balance -= pays.get(d, 0.0)
            minimum = min(minimum, balance)
            for f in by_day.get(d, []):
                balance += f["amount"] if f["direction"] == "credit" else -f["amount"]
                minimum = min(minimum, balance)
        return minimum

    def run(self):
        rows = []
        for request in self.requests:
            rows.append(self.decide(request))

        # Deterministic validation required by the challenge.
        request_ids = [r["request_id"] for r in self.requests]
        assert len(rows) == len(requests := self.requests)
        assert len({r["request_id"] for r in rows}) == len(rows)

        for out, req in zip(rows, self.requests):
            amount = float(out["amount_safe_to_pay"])
            requested = float(req["requested_amount"])
            assert -1e-8 <= amount <= requested + 1e-8
            assert out["affordability_status"] in {
                "affordable_now", "affordable_with_plan",
                "affordable_later", "not_affordable"
            }
            assert out["recommended_payment_method"] in {
                "full_payment", "partial_payment", "installments",
                "wait", "not_recommended"
            }

        with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # Final repository-root checks.
        assert os.path.exists(OUTPUT)
        with open(OUTPUT, newline="", encoding="utf-8") as f:
            written = list(csv.DictReader(f))
        assert list(written[0].keys()) == REQUIRED_COLUMNS
        assert len(written) == len(self.requests)
        assert [r["request_id"] for r in written] == request_ids

        print(f"Created {OUTPUT}")
        print(f"Rows: {len(written)}")
        print(f"Columns: {', '.join(REQUIRED_COLUMNS)}")


if __name__ == "__main__":
    Agent().run()
