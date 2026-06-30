"""
Verification tests for supply_chain_lite_master.xlsx (v3.1 schema) and code fixes.
Run: pytest tests/test_master_v3.py -v
"""
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

V3_PATH = Path("data/raw/supply_chain_lite_master.xlsx")
DB_PATH = Path("outputs/supply_chain.db")
DISTILBERT_SRC = Path("src/agents/distilbert_signal.py")
TRAINING_DATA_SRC = Path("fine_tuning/generate_training_data.py")

NON_ELEC_KEYWORDS = [
    "golf", "slide", "athletic", "jersey", "titleist", "bridgestone",
    "under armour", "women's ignite", "men's compression", "kids' mercenary",
]
VALID_LABELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


class TestV3WorkbookExists:
    def test_file_exists(self):
        assert V3_PATH.exists(), f"v3.1 workbook not found at {V3_PATH}"

    def test_expected_sheets(self):
        xl = pd.ExcelFile(V3_PATH)
        expected = {
            "Lite Master", "Column Guide (Lite)", "Legend",
            "Ops KPI (Filled)", "Semiconductor Signals"
        }
        assert expected.issubset(set(xl.sheet_names))


class TestLiteMasterV3:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_excel(V3_PATH, sheet_name="Lite Master", header=1)

    def test_row_count(self, df):
        assert len(df) >= 10_000, f"Expected ≥10,000 rows, got {len(df):,}"

    def test_column_count(self, df):
        assert len(df.columns) == 32, \
            f"Expected 32 columns (30 original + 2 new), got {len(df.columns)}"

    def test_new_columns_present(self, df):
        assert "Known_Disruption_Event" in df.columns, \
            "Known_Disruption_Event column missing from Lite Master"
        assert "Known_Severity" in df.columns, \
            "Known_Severity column missing from Lite Master"

    def test_macro_event_stamp_2020(self, df):
        y2020 = df[df["Year"] == 2020]
        events = y2020["Known_Disruption_Event"].unique()
        assert "COVID-19 Pandemic" in events, \
            f"2020 rows should have 'COVID-19 Pandemic', found: {events}"

    def test_macro_event_stamp_2021(self, df):
        y2021 = df[df["Year"] == 2021]
        events = y2021["Known_Disruption_Event"].unique()
        assert "Global Chip Shortage + Texas Freeze" in events, \
            f"2021 rows should have chip shortage event, found: {events}"

    def test_macro_event_stamp_2022(self, df):
        y2022 = df[df["Year"] == 2022]
        events = y2022["Known_Disruption_Event"].unique()
        assert "Russia-Ukraine / US Export Controls" in events, \
            f"2022 rows should have Ukraine/CHIPS event, found: {events}"

    def test_baseline_years_have_no_event(self, df):
        baseline = df[df["Year"].isin([2015, 2016, 2017, 2018])]
        assert (baseline["Known_Disruption_Event"] == "—").all(), \
            "2015-2018 baseline rows should have Known_Disruption_Event='—'"

    def test_low_rows_in_2020_have_event_context(self, df):
        """The 222 confusing LOW rows in 2020-2021 must now carry event context."""
        low_2020 = df[
            (df["Year"] == 2020) &
            (df["Disruption_Event_Label"] == "LOW") &
            (df["Supply_Disruption_Index"] > 7.0)
        ]
        if len(low_2020) > 0:
            without_context = (low_2020["Known_Disruption_Event"] == "—").sum()
            assert without_context == 0, \
                f"{without_context} LOW rows in 2020 with SDI>7 have no macro event context"

    def test_year_coverage_continuous(self, df):
        years = sorted(df["Year"].dropna().unique().astype(int))
        missing = [y for y in range(2015, 2026) if y not in years]
        assert not missing, f"Missing years: {missing}"

    def test_no_contamination(self, df):
        def check(name):
            n = str(name).lower()
            return any(kw in n for kw in NON_ELEC_KEYWORDS)
        contaminated = df["Product_Name"].apply(check).sum()
        assert contaminated == 0, f"{contaminated} non-electronics rows found"

    def test_label_values_valid(self, df):
        invalid = set(df["Disruption_Event_Label"].dropna().unique()) - VALID_LABELS
        assert not invalid, f"Invalid labels: {invalid}"

    def test_canceled_rows_are_critical_gap_years(self, df):
        gap = df[df["Year"].isin([2019, 2020, 2021, 2022])]
        canceled = gap[gap["Delivery_Status"] == "Shipping canceled"]
        bad = canceled[canceled["Disruption_Event_Label"] != "CRITICAL"]
        assert len(bad) == 0, f"{len(bad)} canceled gap-year rows are NOT CRITICAL"

    def test_tier1_row_count(self, df):
        tier1 = df[df["Year"].isin([2015, 2016, 2017, 2018])]
        assert len(tier1) == 5459, f"Tier 1 should be 5,459 rows, got {len(tier1):,}"

    def test_tier3_row_count(self, df):
        tier3 = df[df["Year"].isin([2023, 2024, 2025])]
        assert len(tier3) == 2700, f"Tier 3 should be 2,700 rows, got {len(tier3):,}"

    def test_critical_count_improved(self, df):
        critical = (df["Disruption_Event_Label"] == "CRITICAL").sum()
        assert critical > 600, f"CRITICAL count {critical} should exceed 600 (was 416 in V2)"


class TestOpsKPIV3:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_excel(V3_PATH, sheet_name="Ops KPI (Filled)", header=1)

    def test_row_count(self, df):
        assert len(df) >= 2600, f"Expected ≥2,600 Ops KPI rows, got {len(df):,}"

    def test_date_range(self, df):
        dates = pd.to_datetime(df["Week_Start"])
        assert dates.min().year <= 2019
        assert dates.max().year >= 2025

    def test_no_nulls_in_key_cols(self, df):
        for col in ["Week_Start", "Demand_Actual", "Disruption_Flag"]:
            assert df[col].isna().sum() == 0, f"{col} has nulls"

    def test_expected_columns(self, df):
        expected = {
            "Week_Start", "SKU_ID", "Region", "Demand_Actual",
            "Disruption_Flag", "MAPE_Baseline", "MAPE_AI", "Disruption_Event_Label"
        }
        assert expected.issubset(set(df.columns))


class TestDistilBERTFix:
    def test_delivery_not_in_text_function(self):
        if not DISTILBERT_SRC.exists():
            pytest.skip("distilbert_signal.py not found")
        source = DISTILBERT_SRC.read_text()
        start = source.find("def build_distilbert_text(")
        assert start != -1, "build_distilbert_text not found"
        func_body = source[start:start + 1800]
        assert "Delivery:" not in func_body, \
            "FAIL: 'Delivery:' string still present in build_distilbert_text()"

    def test_known_disruption_event_in_text_function(self):
        """Known_Disruption_Event MUST be included in the DistilBERT input."""
        if not DISTILBERT_SRC.exists():
            pytest.skip("distilbert_signal.py not found")
        source = DISTILBERT_SRC.read_text()
        start = source.find("def build_distilbert_text(")
        func_body = source[start:start + 1800]
        assert "known_disruption_event" in func_body.lower(), \
            "FAIL: known_disruption_event not present in build_distilbert_text()"

    def test_sql_excludes_delivery_status(self):
        if not TRAINING_DATA_SRC.exists():
            pytest.skip("generate_training_data.py not yet created")
        source = TRAINING_DATA_SRC.read_text()
        start = source.find("def load_distilbert_data(")
        if start == -1:
            pytest.skip("load_distilbert_data not found")
        func_body = source[start:start + 2000]
        lines = [l for l in func_body.split("\n")
                 if "delivery_status" in l.lower() and not l.strip().startswith("#")]
        assert not lines, \
            f"delivery_status still in load_distilbert_data SQL: {lines}"

    def test_sql_includes_known_disruption_event(self):
        if not TRAINING_DATA_SRC.exists():
            pytest.skip("generate_training_data.py not yet created")
        source = TRAINING_DATA_SRC.read_text()
        start = source.find("def load_distilbert_data(")
        if start == -1:
            pytest.skip("load_distilbert_data not found")
        func_body = source[start:start + 2000]
        assert "known_disruption_event" in func_body.lower(), \
            "FAIL: known_disruption_event not selected in load_distilbert_data SQL"


class TestSQLiteReload:
    @pytest.fixture(scope="class")
    def conn(self):
        if not DB_PATH.exists():
            pytest.skip("supply_chain.db not found — run etl_loader first")
        return sqlite3.connect(DB_PATH)

    def test_lite_master_row_count(self, conn):
        count = conn.execute("SELECT COUNT(*) FROM lite_master").fetchone()[0]
        assert count >= 10_000, f"SQLite lite_master has {count:,} rows, expected ≥10,000"

    def test_new_columns_in_schema(self, conn):
        cols = [r[1] for r in conn.execute("PRAGMA table_info(lite_master)").fetchall()]
        assert "known_disruption_event" in cols, \
            "known_disruption_event column missing from SQLite lite_master schema"
        assert "known_severity" in cols, \
            "known_severity column missing from SQLite lite_master schema"

    def test_2020_rows_have_covid_event(self, conn):
        rows = conn.execute(
            "SELECT COUNT(*) FROM lite_master "
            "WHERE year = 2020 AND known_disruption_event = 'COVID-19 Pandemic'"
        ).fetchone()[0]
        assert rows > 0, "No 2020 rows have Known_Disruption_Event='COVID-19 Pandemic'"

    def test_no_contamination_in_sqlite(self, conn):
        rows = conn.execute(
            "SELECT COUNT(*) FROM lite_master WHERE "
            "LOWER(product_name) LIKE '%golf%' OR "
            "LOWER(product_name) LIKE '%titleist%' OR "
            "LOWER(product_name) LIKE '%under armour%'"
        ).fetchone()[0]
        assert rows == 0, f"{rows} non-electronics rows in SQLite"

    def test_canceled_rows_are_critical(self, conn):
        rows = conn.execute(
            "SELECT COUNT(*) FROM lite_master "
            "WHERE delivery_status='Shipping canceled' "
            "AND disruption_event_label!='CRITICAL'"
        ).fetchone()[0]
        assert rows == 0, f"{rows} 'Shipping canceled' rows are NOT CRITICAL"

    def test_ops_kpi_row_count(self, conn):
        count = conn.execute("SELECT COUNT(*) FROM ops_kpi").fetchone()[0]
        assert count >= 2600, f"SQLite ops_kpi has {count:,} rows, expected ≥2,600"

    def test_ops_kpi_date_range(self, conn):
        result = conn.execute(
            "SELECT MIN(week_start), MAX(week_start) FROM ops_kpi"
        ).fetchone()
        assert result[0][:4] <= "2019", f"ops_kpi starts too late: {result[0]}"
        assert result[1][:4] >= "2025", f"ops_kpi ends too early: {result[1]}"
