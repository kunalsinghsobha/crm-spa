# app.py
# SPA Cycle Time Dashboard — Streamlit
# Author: ChatGPT (for Kunal @ Sobha Group)
# Python 3.9+ compatible.
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import io
import re
from typing import Optional

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="SPA Cycle Time Dashboard", layout="wide", page_icon="📊")

# ------------------------------
# Settings & labels
# ------------------------------
TZ = "Asia/Dubai"
OOPS = "Oopsie! Data not present"

EXPECTED_COLS = {
    "project": "Project",
    "tower": "Tower Name",
    "unit": "Unit Name",
    "booking": "Booking Name",
    "rera": "RERA Number",
    "salesops_approval": "SalesOps Assurance Approval Date",
    "spa_eligibility": "SPA Eligibility Date",
    "floorplan_upload": "Floor Plan Upload Date",
    "registration": "Registration date",
    "spa_sent": "SPA Sent Date",
    "crm_assurance": "SPA Sent to CRM OPS Assurance Date",
    "spa_executed": "SPA Executed Date",
    "pre_registration": "Date of Pre-Registration Initiation",
}

DATE_COLS = [
    "salesops_approval",
    "spa_eligibility",
    "floorplan_upload",
    "registration",
    "spa_sent",
    "crm_assurance",
    "spa_executed",
    "pre_registration",
]

# ------------------------------
# Helpers
# ------------------------------

def _wrap_label(text: str, width: int = 20) -> str:
    """Insert line breaks every `width` characters (for compact KPI labels)."""
    if not isinstance(text, str):
        text = str(text)
    return "\n".join([text[i:i+width] for i in range(0, len(text), width)])

@st.cache_data(show_spinner=False)
def _robust_to_datetime(raw: pd.Series, dayfirst: bool = True) -> pd.Series:
    """
    Parse a mixed-format date/time column safely.
    - Accepts strings like '1/2/2024', '1/2/2024 3:45 PM'
    - Accepts Excel serials (numbers)
    Returns pandas datetime64[ns] with NaT for invalids (tz-naive).
    """
    s = pd.to_datetime(raw, dayfirst=dayfirst, errors="coerce")

    # If some values failed but are numeric (Excel serials), try converting those
    if not isinstance(raw, pd.Series):
        raw = pd.Series(raw)

    numeric_mask = s.isna() & pd.to_numeric(raw, errors="coerce").notna()
    if numeric_mask.any():
        s2 = pd.to_datetime(
            pd.to_numeric(raw[numeric_mask], errors="coerce"),
            unit="d",
            origin="1899-12-30",
            errors="coerce",
        )
        s.loc[numeric_mask] = s2

    # Ensure tz-naive to avoid mixing aware/naive later
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)

    return s


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a normalized-name map (kept for debugging)."""
    _ = {re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_"): c for c in df.columns}
    df._norm_map = _  # debug info if needed
    return df


def _find_col(df: pd.DataFrame, expected_name: str) -> Optional[str]:
    """Find a column in df that best matches expected_name (case/space/punct-insensitive)."""
    target = re.sub(r"[^a-z0-9]+", "_", expected_name.strip().lower()).strip("_")
    for c in df.columns:
        cand = re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
        if cand == target:
            return c
    for c in df.columns:
        cand = re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
        if target in cand or cand in target:
            return c
    return None


def _status_badge(status: str) -> str:
    color = {
        "On-Time": "green",
        "Watch": "orange",
        "Delayed": "red",
        "Critical": "purple",
        "Missing": "gray",
        "Pending": "blue",
    }.get(status, "gray")
    return f":{color}[{status}]"


def _severity_from_days(days: Optional[float], limits=(7, 14, 60)) -> str:
    """Severity for S3 pending: On-Time (≤7), Watch (8–14), Delayed (15–60), Critical (>60)."""
    if days is None or (isinstance(days, float) and np.isnan(days)):
        return "Missing"
    if days <= limits[0]:
        return "On-Time"
    if days <= limits[1]:
        return "Watch"
    if days <= limits[2]:
        return "Delayed"
    return "Critical"


def _date_label(dt) -> str:
    """Return a friendly label for timestamps or NaT."""
    return dt.strftime("%d/%m/%Y %I:%M %p") if (pd.notna(dt) and isinstance(dt, pd.Timestamp)) else OOPS


@st.cache_data(show_spinner=False)
def process_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load + standardize + compute cycle metrics. Cached for speed."""
    df_raw = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    df = df_raw.copy()
    df = _normalize_cols(df)

    # Resolve columns
    colmap = {key: _find_col(df, exp_name) for key, exp_name in EXPECTED_COLS.items()}

    # Soft validation for essential IDs
    missing_essentials = [k for k in ["project", "tower", "unit", "booking"] if colmap.get(k) is None]
    if missing_essentials:
        st.warning(
            f"Some identifier columns are missing: {missing_essentials}. "
            "The app will still run using available columns."
        )

    # Parse date columns
    for key in DATE_COLS:
        src = colmap.get(key)
        if src is None:
            df[key] = pd.NaT
        else:
            df[key] = _robust_to_datetime(df[src])

    # Copy through text columns
    for key in ["project", "tower", "unit", "booking", "rera"]:
        src = colmap.get(key)
        df[key] = df[src] if (src in df.columns) else None

    # Booking code validation
    df["booking_valid"] = df["booking"].astype(str).str.match(r"^B-\d+$", na=False)

    # Ensure we use tz-naive "today" to match tz-naive parsed dates
    today = pd.Timestamp.now(pytz.timezone(TZ)).normalize().tz_localize(None)

    # ---------- S1: SalesOps Approval -> SPA Eligibility (<= 14 days) ----------
    s1_start = df["salesops_approval"]
    s1_end = df["spa_eligibility"]
    s1_days = (s1_end - s1_start).dt.days
    df["s1_days"] = s1_days
    df["s1_status"] = np.where(
        s1_days.isna(), "Missing",
        np.where(s1_days <= 14, "On-Time", "Delayed")
    )
    df["s1_reason"] = np.select(
        [s1_days.isna(), s1_days <= 14, s1_days > 14],
        [
            "Eligibility date missing or SalesOps approval missing",
            "KYC/Payment completed within 14 days",
            "KYC/Payment pending beyond 14 days (disqualification risk)",
        ],
        default=""
    )

    # Trigger-ready date L = max(eligibility, floorplan upload, registration)
    trigger_df = pd.concat([df["spa_eligibility"], df["floorplan_upload"], df["registration"]], axis=1)
    df["trigger_ready_date"] = trigger_df.max(axis=1, skipna=True)

    # ---------- S2: Trigger-ready -> SPA Sent (<= 3 days); support pending ----------
    s2_start = df["trigger_ready_date"]
    s2_end = df["spa_sent"]
    s2_days = (s2_end - s2_start).dt.days
    df["s2_days"] = s2_days

    s2_pending = s2_end.isna() & s2_start.notna()
    s2_pending_days = pd.Series(np.nan, index=df.index, dtype="float")
    s2_pending_days.loc[s2_pending] = (today - s2_start.loc[s2_pending]).dt.days
    df["s2_pending_days"] = s2_pending_days

    df["s2_status"] = np.select(
        [
            s2_start.isna(),                               # cannot compute
            s2_end.notna() & (s2_days <= 3),              # completed on time
            s2_end.notna() & (s2_days > 3),               # completed late
            s2_pending & (df["s2_pending_days"] <= 3),    # pending within SLA
            s2_pending & (df["s2_pending_days"] > 3),     # pending beyond SLA
        ],
        ["Missing", "On-Time", "Delayed", "Pending", "Delayed"],
        default="Missing"
    )
    df["s2_reason"] = np.select(
        [
            s2_start.isna(),
            s2_end.notna() & (s2_days <= 3),
            s2_end.notna() & (s2_days > 3),
            s2_pending & (df["s2_pending_days"] <= 3),
            s2_pending & (df["s2_pending_days"] > 3),
        ],
        [
            "Cannot compute trigger date (eligibility/floorplan/registration missing)",
            "SPA sent within 3 days of being ready",
            "SPA sent after 3-day SLA; check capacity/floorplan/registration dependencies",
            "SPA pending but still within 3-day SLA",
            "SPA pending and beyond 3-day SLA; manpower or document readiness issue",
        ],
        default=""
    )

    # ---------- S3: SPA Sent -> CRM Ops Assurance (<=14; >60 Critical); support pending ----------
    s3_start = df["spa_sent"]
    s3_end = df["crm_assurance"]
    s3_days = (s3_end - s3_start).dt.days
    df["s3_days"] = s3_days

    s3_pending = s3_end.isna() & s3_start.notna()
    s3_pending_days = pd.Series(np.nan, index=df.index, dtype="float")
    s3_pending_days.loc[s3_pending] = (today - s3_start.loc[s3_pending]).dt.days
    df["s3_pending_days"] = s3_pending_days

    # Use completed days or pending days to derive severity (S3 only)
    s3_effective = s3_days.copy()
    s3_effective.loc[s3_pending] = df.loc[s3_pending, "s3_pending_days"]
    df["s3_severity"] = s3_effective.apply(_severity_from_days)

    df["s3_status"] = np.select(
        [
            s3_start.isna(),                                               # cannot start
            s3_end.notna() & (s3_days <= 14),                              # completed within 14
            s3_end.notna() & (s3_days > 14) & (s3_days <= 60),             # completed late
            (s3_end.notna() & (s3_days > 60)) | (s3_pending & (s3_pending_days > 60)),
            s3_pending & (s3_pending_days <= 14),                          # pending within target
            s3_pending & (s3_pending_days > 14) & (s3_pending_days <= 60), # pending late
        ],
        ["Missing", "On-Time", "Delayed", "Critical", "Pending", "Delayed"],
        default="Missing"
    )

    df["s3_reason"] = np.select(
        [
            s3_start.isna(),
            s3_end.notna() & (s3_days <= 14),
            s3_end.notna() & (s3_days > 14) & (s3_days <= 60),
            (s3_end.notna() & (s3_days > 60)) | (s3_pending & (s3_pending_days > 60)),
            s3_pending & (s3_pending_days <= 14),
            s3_pending & (s3_pending_days > 14) & (s3_pending_days <= 60),
        ],
        [
            "SPA Sent Date missing; cannot track customer return",
            "Customer signed & returned within 14 days",
            "Customer return beyond 14 days; follow-up needed",
            "Exceeds 60-day cutoff. Auto-cancellation risk — escalate immediately",
            "Customer return pending (within 14 days)",
            "Customer return pending beyond 14 days; risk rising",
        ],
        default=""
    )

    # ---------- S4: CRM Ops Assurance -> SPA Executed (<=3); support pending ----------
    s4_start = df["crm_assurance"]
    s4_end = df["spa_executed"]
    s4_days = (s4_end - s4_start).dt.days
    df["s4_days"] = s4_days

    s4_pending = s4_end.isna() & s4_start.notna()
    s4_pending_days = pd.Series(np.nan, index=df.index, dtype="float")
    s4_pending_days.loc[s4_pending] = (today - s4_start.loc[s4_pending]).dt.days
    df["s4_pending_days"] = s4_pending_days

    df["s4_status"] = np.select(
        [
            s4_start.isna(),
            s4_end.notna() & (s4_days <= 3),
            s4_end.notna() & (s4_days > 3),
            s4_pending & (s4_pending_days <= 3),
            s4_pending & (s4_pending_days > 3),
        ],
        ["Missing", "On-Time", "Delayed", "Pending", "Delayed"],
        default="Missing"
    )

    df["s4_reason"] = np.select(
        [
            s4_start.isna(),
            s4_end.notna() & (s4_days <= 3),
            s4_end.notna() & (s4_days > 3),
            s4_pending & (s4_pending_days <= 3),
            s4_pending & (s4_pending_days > 3),
        ],
        [
            "CRM Assurance Date missing; cannot track execution",
            "Executed within 3 days of CRM Assurance",
            "Execution beyond 3 days; internal maker-checker/signatory delay",
            "Execution pending (within 3-day SLA)",
            "Execution pending and beyond 3-day SLA; escalate",
        ],
        default=""
    )

    # ---------- Informational: Pre-Registration delta ----------
    pr_start = df["spa_eligibility"]
    pr_end = df["pre_registration"]
    df["pre_registration_days"] = (pr_end - pr_start).dt.days

    # Labels
    for key in DATE_COLS + ["trigger_ready_date"]:
        df[f"{key}_label"] = df[key].apply(_date_label)

    # Funnel flags
    df["funnel_eligible"] = df["spa_eligibility"].notna()
    df["funnel_sent"] = df["spa_sent"].notna()
    df["funnel_crm"] = df["crm_assurance"].notna()
    df["funnel_executed"] = df["spa_executed"].notna()

    # --------- Pending stage classification & pending days ----------
    df["pending_stage"] = None
    df["pending_days"] = np.nan
    df["pending_severity"] = "Missing"  # unified severity for filtering

    # S1 pending: eligibility missing (SalesOps may or may not exist)
    s1_pend_mask = df["spa_eligibility"].isna()
    df.loc[s1_pend_mask, "pending_stage"] = "S1"
    # If SalesOps exists we can compute days; else NaN
    s1_have_start = s1_pend_mask & df["salesops_approval"].notna()
    df.loc[s1_have_start, "pending_days"] = (today - df.loc[s1_have_start, "salesops_approval"]).dt.days
    # Severity for S1 (On-Time ≤14, Delayed >14)
    df.loc[s1_pend_mask & df["pending_days"].le(14, fill_value=np.nan), "pending_severity"] = "On-Time"
    df.loc[s1_pend_mask & df["pending_days"].gt(14, fill_value=False), "pending_severity"] = "Delayed"
    df.loc[s1_pend_mask & df["pending_days"].isna(), "pending_severity"] = "Missing"

    # S2 pending: trigger ready exists and spa_sent missing
    s2_pend_mask = df["trigger_ready_date"].notna() & df["spa_sent"].isna()
    df.loc[s2_pend_mask, "pending_stage"] = "S2"
    df.loc[s2_pend_mask, "pending_days"] = (today - df.loc[s2_pend_mask, "trigger_ready_date"]).dt.days
    df.loc[s2_pend_mask & df["pending_days"].le(3, fill_value=np.nan), "pending_severity"] = "On-Time"
    df.loc[s2_pend_mask & df["pending_days"].gt(3, fill_value=False), "pending_severity"] = "Delayed"

    # S3 pending: spa_sent exists and crm_assurance missing
    s3_pend_mask = df["spa_sent"].notna() & df["crm_assurance"].isna()
    df.loc[s3_pend_mask, "pending_stage"] = "S3"
    df.loc[s3_pend_mask, "pending_days"] = (today - df.loc[s3_pend_mask, "spa_sent"]).dt.days
    # Use S3 rich severity
    s3_sev = df.loc[s3_pend_mask, "pending_days"].apply(_severity_from_days)
    df.loc[s3_pend_mask, "pending_severity"] = s3_sev

    # S4 pending: crm_assurance exists and spa_executed missing
    s4_pend_mask = df["crm_assurance"].notna() & df["spa_executed"].isna()
    df.loc[s4_pend_mask, "pending_stage"] = "S4"
    df.loc[s4_pend_mask, "pending_days"] = (today - df.loc[s4_pend_mask, "crm_assurance"]).dt.days
    df.loc[s4_pend_mask & df["pending_days"].le(3, fill_value=np.nan), "pending_severity"] = "On-Time"
    df.loc[s4_pend_mask & df["pending_days"].gt(3, fill_value=False), "pending_severity"] = "Delayed"

    return df


def _kpi_row(filtered: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    delayed_s1 = (filtered["s1_status"] == "Delayed").sum()
    delayed_s2 = (filtered["s2_status"] == "Delayed").sum()
    delayed_s3 = ((filtered["s3_status"] == "Delayed") | (filtered["s3_status"] == "Critical")).sum()
    delayed_s4 = (filtered["s4_status"] == "Delayed").sum()
    missing_any = filtered[["s1_status", "s2_status", "s3_status", "s4_status"]].eq("Missing").any(axis=1).sum()

    c1.metric(_wrap_label("Delayed: Eligibility (S1)"), int(delayed_s1))
    c2.metric(_wrap_label("Delayed: SPA Sending (S2)"), int(delayed_s2))
    c3.metric(_wrap_label("Delayed/Critical: Customer Return (S3)"), int(delayed_s3))
    c4.metric(_wrap_label("Delayed: Execution (S4)"), int(delayed_s4))
    c5.metric(_wrap_label("Records with Missing Dates"), int(missing_any))


def _kpi_legend_box():
    st.markdown(
        """
> **Legend — Topline KPIs**
> - **Delayed: Eligibility (S1)** → `SPA Eligibility - SalesOps Approval > 14 days`.
> - **Delayed: SPA Sending (S2)** → `SPA Sent - TriggerReady > 3 days` **or** Pending > 3 days.  
>   • *TriggerReady* = latest of **SPA Eligibility**, **Floor Plan Upload**, **Registration**.
> - **Delayed/Critical: Customer Return (S3)** →  
>   • **Delayed**: `CRM Assurance - SPA Sent > 14 and ≤ 60 days` **or** Pending in that band.  
>   • **Critical**: `> 60 days` **or** Pending > 60 days.  
>   • **On-Time**: `≤ 14 days` (Watch band `8–14` used for severity tagging).
> - **Delayed: Execution (S4)** → `SPA Executed - CRM Assurance > 3 days` **or** Pending > 3 days.
> - **Records with Missing Dates** → Any stage status is **Missing** (insufficient dates to compute).
        """
    )


def _filters_ui(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("🔎 Filters")
        projects = sorted([p for p in df["project"].dropna().astype(str).unique() if p.strip()])
        towers = sorted([t for t in df["tower"].dropna().astype(str).unique() if t.strip()])
        units = sorted([u for u in df["unit"].dropna().astype(str).unique() if u.strip()])

        sel_projects = st.multiselect("Project", options=projects, default=[])
        sel_towers = st.multiselect("Tower Name", options=towers, default=[])
        sel_units = st.multiselect("Unit Name", options=units, default=[])

        # Executed vs Pending switch (this is the toggle you asked for)
        case_view = st.radio(
            "Cases",
            options=["Executed", "Pending"],
            horizontal=True,
            help="Executed = SPA Executed present; Pending = SPA Executed missing",
        )

        booking_search = st.text_input("Search Booking Code (e.g., B-22123)", value="").strip()
        delayed_stage = st.multiselect(
            "Show only delayed in stage(s)",
            options=["S1", "S2", "S3", "S4"],
            help="Filters to rows delayed/critical in selected stages (applies to both views)",
        )
        date_range_on = st.checkbox("Filter by **SPA Eligibility Date** range", value=False)
        date_min, date_max = None, None
        if date_range_on:
            min_date = pd.to_datetime("2000-01-01")
            max_date = pd.Timestamp.now(pytz.timezone(TZ)).date()
            date_min = st.date_input("From (SPA Eligibility)", value=min_date)
            date_max = st.date_input("To (SPA Eligibility)", value=max_date)

        # Extra filters when 'Pending' view
        pending_stage_sel = []
        min_pending_days = 0
        pending_sev_sel = []
        if case_view == "Pending":
            st.markdown("---")
            st.subheader("Pending filters")
            pending_stage_sel = st.multiselect(
                "Pending at stage(s)",
                options=["S1", "S2", "S3", "S4"],
                default=["S1", "S2", "S3", "S4"],
                help="Filter pending cases to specific stages",
            )
            min_pending_days = st.slider(
                "Show cases pending for at least (days)",
                min_value=0, max_value=180, value=0, step=1,
            )
            pending_sev_sel = st.multiselect(
                "Pending severity",
                options=["On-Time", "Watch", "Delayed", "Critical", "Missing"],
                default=[],
                help="S3 uses all bands; S1/S2/S4 use On-Time/Delayed/Missing",
            )

    # Base mask
    mask = pd.Series(True, index=df.index)
    if sel_projects:
        mask &= df["project"].astype(str).isin(sel_projects)
    if sel_towers:
        mask &= df["tower"].astype(str).isin(sel_towers)
    if sel_units:
        mask &= df["unit"].astype(str).isin(sel_units)
    if booking_search:
        patt = re.escape(booking_search)
        mask &= df["booking"].astype(str).str.contains(patt, case=False, na=False)
    if delayed_stage:
        conds = []
        for s in delayed_stage:
            col = f"{s.lower()}_status"
            if s == "S3":
                conds.append(df[col].isin(["Delayed", "Critical"]))
            else:
                conds.append(df[col].eq("Delayed"))
        if conds:
            stage_mask = conds[0]
            for c in conds[1:]:
                stage_mask |= c
            mask &= stage_mask
    if date_range_on and (date_min is not None) and (date_max is not None):
        s = df["spa_eligibility"].dt.date
        mask &= s.between(pd.to_datetime(date_min).date(), pd.to_datetime(date_max).date(), inclusive="both")

    # View-specific masks
    if case_view == "Executed":
        mask &= df["spa_executed"].notna()
    else:
        # Pending view
        mask &= df["spa_executed"].isna()               # << this hides active/executed rows
        if pending_stage_sel:
            mask &= df["pending_stage"].isin(pending_stage_sel)
        mask &= (df["pending_days"].fillna(-1) >= min_pending_days)
        if pending_sev_sel:
            mask &= df["pending_severity"].isin(pending_sev_sel)

    filtered = df[mask].copy()
    filtered["_case_view"] = case_view
    return filtered


def _compliance_bar(df: pd.DataFrame, stage: str):
    col = f"{stage.lower()}_status"
    if col not in df.columns:
        st.info("No data to chart.")
        return
    cats = ["On-Time", "Watch", "Pending", "Delayed", "Critical", "Missing"]
    s = df[col].value_counts().reindex(cats).fillna(0).reset_index()
    s.columns = ["Status", "Count"]
    chart = (
        alt.Chart(s)
        .mark_bar()
        .encode(
            x=alt.X("Status:N", sort=cats, title="Status"),
            y=alt.Y("Count:Q", title="Cases"),
            tooltip=["Status", "Count"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def _sla_legend_box():
    st.markdown(
        """
> **Legend — SLA Stages (S1–S4)**
> - **S1 — Eligibility**: *SalesOps Approval → SPA Eligibility* (target ≤ **14 days**).
> - **S2 — SPA Sending**: *Trigger-Ready → SPA Sent* (target ≤ **3 days**).  
>   • *Trigger-Ready* = latest of **SPA Eligibility**, **Floor Plan Upload**, **Registration**.
> - **S3 — Customer Return (CRM)**: *SPA Sent → CRM Ops Assurance* (target ≤ **14 days**; **>60 days = Critical**).
> - **S4 — Execution**: *CRM Ops Assurance → SPA Executed* (target ≤ **3 days**).
        """
    )


def _funnel(df: pd.DataFrame):
    # Renamed stages as requested
    stages = [
        "SPA Eligible to Sent to Customer",   # formerly "Eligible"
        "Customer signed and sent back",      # formerly "Sent"
        "CRM Ops sent for Execution",         # formerly "CRM"
        "SPA Executed",                       # formerly "Executed"
    ]
    vals = [
        int(df["funnel_eligible"].sum()),
        int(df["funnel_sent"].sum()),
        int(df["funnel_crm"].sum()),
        int(df["funnel_executed"].sum()),
    ]
    f = pd.DataFrame({"Stage": stages, "Count": vals})
    chart = (
        alt.Chart(f)
        .mark_bar()
        .encode(
            x=alt.X("Stage:N", sort=stages, title="Funnel Stage"),
            y=alt.Y("Count:Q", title="Cases"),
            tooltip=["Stage", "Count"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def _timeline_for_row(row: pd.Series):
    steps = [
        ("S1: Eligibility", row["salesops_approval"], row["spa_eligibility"]),
        ("S2: SPA Sending", row["trigger_ready_date"], row["spa_sent"]),
        ("S3: Customer Return (CRM)", row["spa_sent"], row["crm_assurance"]),
        ("S4: Execution", row["crm_assurance"], row["spa_executed"]),
    ]

    data = []
    for label, start, end in steps:
        if pd.isna(start) or pd.isna(end):
            if pd.notna(start):
                data.append({"Stage": label, "Start": start, "End": start + pd.Timedelta(days=0.1)})
            else:
                today = pd.Timestamp.now(pytz.timezone(TZ)).normalize().tz_localize(None)
                data.append({"Stage": label, "Start": today, "End": today + pd.Timedelta(days=0.1)})
        else:
            data.append({"Stage": label, "Start": start, "End": end})

    tdf = pd.DataFrame(data)
    chart = (
        alt.Chart(tdf)
        .mark_bar()
        .encode(
            y=alt.Y("Stage:N", title=""),
            x=alt.X("Start:T", title="Timeline"),
            x2="End:T",
            tooltip=[alt.Tooltip("Stage:N"), alt.Tooltip("Start:T"), alt.Tooltip("End:T")],
        )
        .properties(height=200)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**Timestamps**")
    c1, c2 = st.columns(2)
    with c1:
        st.write("SalesOps Approval:", _date_label(row["salesops_approval"]))
        st.write("SPA Eligibility:", _date_label(row["spa_eligibility"]))
        st.write("Trigger Ready (max):", _date_label(row["trigger_ready_date"]))
        st.write("SPA Sent:", _date_label(row["spa_sent"]))
    with c2:
        st.write("CRM Assurance:", _date_label(row["crm_assurance"]))
        st.write("SPA Executed:", _date_label(row["spa_executed"]))
        st.write("Pre-Registration Initiation:", _date_label(row["pre_registration"]))

    st.markdown("**Status & Reasons**")
    st.write("S1:", _status_badge(row["s1_status"]), "—", row.get("s1_reason") or OOPS)
    st.write("S2:", _status_badge(row["s2_status"]), "—", row.get("s2_reason") or OOPS)
    st.write(
        "S3:",
        _status_badge(row["s3_status"]),
        "—",
        row.get("s3_reason") or OOPS,
        "| Severity:",
        _status_badge(row.get("s3_severity", "Missing")),
    )
    st.write("S4:", _status_badge(row["s4_status"]), "—", row.get("s4_reason") or OOPS)


# ------------------------------
# Sidebar — File upload
# ------------------------------
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    st.caption("Expected columns are documented in the README. Parsing is robust even if some are missing.")

if uploaded is None:
    st.info("Please upload the latest 'Booking Report with Units and Towers' Excel file to begin.")
    st.stop()

# ------------------------------
# Pipeline
# ------------------------------
file_bytes = uploaded.read()
df = process_dataframe(file_bytes, uploaded.name)

st.title("📊 SPA Cycle Time Dashboard")
st.caption("Sobha Group — Cycle times, SLA compliance, and delayed case drilldowns.")

# Filters (includes Executed vs Pending switch + pending-stage filters)
filtered = _filters_ui(df)

# KPIs
_kpi_row(filtered)
_kpi_legend_box()  # KPI legend box

st.divider()

# Charts
c1, c2 = st.columns((1, 1))
with c1:
    st.subheader("Funnel Overview")
    _funnel(filtered)
    st.caption(
        "Stages shown as: **SPA Eligible to Sent to Customer** → **Customer signed and sent back** "
        "→ **CRM Ops sent for Execution** → **SPA Executed**."
    )
with c2:
    st.subheader("SLA Compliance Snapshot")
    stage = st.selectbox("Stage", options=["S1", "S2", "S3", "S4"], index=2, key="stage_sel")
    _compliance_bar(filtered, stage)
    _sla_legend_box()  # SLA stage legend under the snapshot

st.divider()

# When Pending view, surface a quick summary per pending stage
if len(filtered) and filtered["_case_view"].iloc[0] == "Pending":
    st.subheader("⏳ Pending Summary")
    pend = (
        filtered.assign(stage=lambda d: d["pending_stage"].fillna("Unclassified"))
        .groupby("stage", dropna=False)
        .agg(
            Cases=("booking", "count"),
            AvgPendingDays=("pending_days", "mean"),
            MaxPendingDays=("pending_days", "max"),
        )
        .reset_index()
    )
    pend["AvgPendingDays"] = pend["AvgPendingDays"].round(1)
    st.dataframe(pend, use_container_width=True, height=220)

# Table
st.subheader("Cases (filtered)")
view_cols = [
    "project", "tower", "unit", "booking", "booking_valid",
    "salesops_approval_label", "spa_eligibility_label", "floorplan_upload_label", "registration_label",
    "trigger_ready_date_label", "spa_sent_label", "crm_assurance_label", "spa_executed_label",
    "s1_days", "s1_status", "s2_days", "s2_status", "s3_days", "s3_status", "s3_severity", "s4_days", "s4_status",
    "pending_stage", "pending_days", "pending_severity"
]
present_cols = [c for c in view_cols if c in filtered.columns]
st.dataframe(filtered[present_cols], use_container_width=True, height=360)

# Download filtered
csv = filtered[present_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="spa_cycle_time_filtered.csv", mime="text/csv")

# Drilldown
st.subheader("🔍 Drilldown: Booking")
options = filtered["booking"].dropna().astype(str).unique().tolist()
sel_book = st.selectbox("Select Booking Code", options=["-- choose --"] + options)
if sel_book and sel_book != "-- choose --":
    row = filtered.loc[filtered["booking"].astype(str) == sel_book].head(1).squeeze()
    st.markdown(f"### {sel_book}")
    _timeline_for_row(row)

# Notes
with st.expander("ℹ️ Deployment & Notes"):
    st.markdown(
        """
**Deploy on Streamlit Cloud**
1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. Go to share.streamlit.io, connect your repo, and deploy.

**Column Expectations** (exact or similar names acceptable):
- Project, Tower Name, Unit Name, Booking Name, RERA Number
- SalesOps Assurance Approval Date (date)
- SPA Eligibility Date (date)
- Floor Plan Upload Date (date & time)
- Registration date (date)
- SPA Sent Date (date)
- SPA Sent to CRM OPS Assurance Date (date)
- SPA Executed Date (date)
- Date of Pre-Registration Initiation (date)

**SLA Rules Encoded**
- S1: SalesOps Approval → SPA Eligibility: **≤ 14 days**
- S2: Latest(Eligibility, Floor Plan Upload, Registration) → SPA Sent: **≤ 3 days**
- S3: SPA Sent → CRM Ops Assurance (Customer returned & forwarded): **≤ 14 days**, **>60 days = Critical**
- S4: CRM Ops Assurance → SPA Executed: **≤ 3 days**

**Missing Data Handling**
- Any missing date shows as *Oopsie! Data not present*.
- Metrics exclude NaNs but "Records with Missing Dates" is surfaced as a KPI.
        """
    )
