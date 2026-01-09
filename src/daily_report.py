"""
HueLight GA4/Google Ads 일일 리포트 생성기
"""

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.oauth2 import service_account
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
from google.ads.googleads.client import GoogleAdsClient
from jinja2 import Environment, FileSystemLoader, select_autoescape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

GA4_SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]
GA4_CREDENTIALS_PATH = ".secrets/ga4.json"
ADS_CREDENTIALS_PATH = ".secrets/google-ads.yaml"

REPORT_TITLE = "HueLight 퍼포먼스 대시보드"
FIXED_START_DATE = "2025-03-18"
TIMEZONE = "Asia/Seoul"
ACCENT_COLOR = "#2563eb"


def seoul_today() -> date:
    return datetime.now(ZoneInfo(TIMEZONE)).date()


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_int(value: float) -> str:
    return f"{int(round(value)):,}"


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def format_currency(value: float) -> str:
    return f"₩{value:,.0f}"


def iso_date_range(start: date, end: date) -> list[str]:
    days = (end - start).days + 1
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days)]


def parse_ga4_date(value: str) -> str:
    if len(value) == 8:
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


class GA4Client:
    def __init__(self, property_id: str):
        creds = service_account.Credentials.from_service_account_file(
            GA4_CREDENTIALS_PATH, scopes=GA4_SCOPES
        )
        self.client = BetaAnalyticsDataClient(credentials=creds)
        self.property_id = property_id

    def run_report(self, dimensions: list, metrics: list, start_date: str, end_date: str, limit=10000) -> list:
        dim_list = [Dimension(name=d) for d in dimensions] if dimensions else []
        request = RunReportRequest(
            property=f"properties/{self.property_id}",
            dimensions=dim_list,
            metrics=[Metric(name=m) for m in metrics],
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            limit=limit,
        )
        response = self.client.run_report(request)

        rows = []
        for row in response.rows:
            data = {}
            for i, dim in enumerate(dimensions):
                data[dim] = row.dimension_values[i].value
            for i, met in enumerate(metrics):
                val = row.metric_values[i].value
                try:
                    data[met] = int(val)
                except ValueError:
                    try:
                        data[met] = float(val)
                    except ValueError:
                        data[met] = val
            rows.append(data)
        return rows


class AdsClient:
    def __init__(self, customer_id: str):
        self.client = GoogleAdsClient.load_from_storage(ADS_CREDENTIALS_PATH)
        self.customer_id = customer_id
        self.service = self.client.get_service("GoogleAdsService")

    def run_query(self, query: str, fallback_query: str | None = None) -> list:
        try:
            response = self.service.search(customer_id=self.customer_id, query=query)
        except Exception:
            if not fallback_query:
                raise
            response = self.service.search(customer_id=self.customer_id, query=fallback_query)
        return list(response)


class ReportGenerator:
    def __init__(self, property_id: str, customer_id: str, start_date: str, end_date: str):
        self.ga4 = GA4Client(property_id)
        self.ads = AdsClient(customer_id)
        self.start_date = start_date
        self.end_date = end_date

    def collect_all_data(self) -> dict:
        print("데이터 수집 중...")
        start = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        all_dates = iso_date_range(start, end)
        last_30_start = max(start, end - timedelta(days=29))
        last_30_dates = iso_date_range(last_30_start, end)

        ga4_daily, ga4_has_data = self._get_ga4_daily_series(self.start_date, self.end_date, all_dates)
        ads_daily, ads_has_data, has_conv_value = self._get_ads_daily_series(self.start_date, self.end_date, all_dates)

        summary = self._build_summary(ga4_daily, ads_daily, last_30_dates, all_dates, has_conv_value)
        tables = self._build_tables(last_30_start.isoformat(), self.end_date)

        return {
            "summary": summary,
            "tables": tables,
            "charts": self._build_chart_data(ga4_daily, ads_daily, last_30_dates, ga4_has_data, ads_has_data),
        }

    def _get_ga4_daily_series(self, start_date: str, end_date: str, all_dates: list[str]) -> tuple[dict, bool]:
        rows = self.ga4.run_report(
            ["date"],
            ["sessions", "activeUsers"],
            start_date,
            end_date,
        )
        data = {d: {"sessions": 0, "activeUsers": 0} for d in all_dates}
        for row in rows:
            date_key = parse_ga4_date(row.get("date", ""))
            if date_key not in data:
                continue
            data[date_key]["sessions"] = row.get("sessions", 0)
            data[date_key]["activeUsers"] = row.get("activeUsers", 0)
        return data, bool(rows)

    def _get_ads_daily_series(self, start_date: str, end_date: str, all_dates: list[str]) -> tuple[dict, bool, bool]:
        query_with_value = (
            "SELECT segments.date, metrics.impressions, metrics.clicks, metrics.cost_micros, "
            "metrics.conversions, metrics.conversions_value "
            "FROM campaign "
            f"WHERE segments.date BETWEEN '{start_date}' AND '{end_date}' "
            "AND campaign.status != 'REMOVED'"
        )
        query_fallback = (
            "SELECT segments.date, metrics.impressions, metrics.clicks, metrics.cost_micros, "
            "metrics.conversions "
            "FROM campaign "
            f"WHERE segments.date BETWEEN '{start_date}' AND '{end_date}' "
            "AND campaign.status != 'REMOVED'"
        )
        rows = self.ads.run_query(query_with_value, fallback_query=query_fallback)

        data = {
            d: {
                "cost": 0.0,
                "impressions": 0,
                "clicks": 0,
                "conversions": 0.0,
                "conversion_value": 0.0,
            }
            for d in all_dates
        }
        has_conv_value = False
        for row in rows:
            date_key = row.segments.date
            if date_key not in data:
                continue
            data[date_key]["cost"] += row.metrics.cost_micros / 1_000_000
            data[date_key]["impressions"] += row.metrics.impressions
            data[date_key]["clicks"] += row.metrics.clicks
            data[date_key]["conversions"] += row.metrics.conversions
            conv_value = getattr(row.metrics, "conversions_value", None)
            if conv_value is not None:
                has_conv_value = True
                data[date_key]["conversion_value"] += conv_value
        return data, bool(rows), has_conv_value

    def _build_summary(self, ga4_daily: dict, ads_daily: dict, last_30_dates: list[str], all_dates: list[str], has_conv_value: bool) -> dict:
        def sum_series(dates: list[str]) -> dict:
            return {
                "ga4_sessions": sum(ga4_daily[d]["sessions"] for d in dates),
                "ga4_active_users": sum(ga4_daily[d]["activeUsers"] for d in dates),
                "ads_cost": sum(ads_daily[d]["cost"] for d in dates),
                "ads_impressions": sum(ads_daily[d]["impressions"] for d in dates),
                "ads_clicks": sum(ads_daily[d]["clicks"] for d in dates),
                "ads_conversions": sum(ads_daily[d]["conversions"] for d in dates),
                "ads_conversion_value": sum(ads_daily[d]["conversion_value"] for d in dates),
            }

        today_key = self.end_date
        today_ga4 = ga4_daily.get(today_key, {"sessions": 0, "activeUsers": 0})
        today_ads = ads_daily.get(today_key, {
            "cost": 0.0,
            "impressions": 0,
            "clicks": 0,
            "conversions": 0.0,
            "conversion_value": 0.0,
        })

        today_metrics = {
            "ga4_sessions": today_ga4["sessions"],
            "ga4_active_users": today_ga4["activeUsers"],
            "ads_cost": today_ads["cost"],
            "ads_impressions": today_ads["impressions"],
            "ads_clicks": today_ads["clicks"],
            "ads_conversions": today_ads["conversions"],
            "ads_ctr": safe_div(today_ads["clicks"], today_ads["impressions"]) * 100,
            "ads_cpc": safe_div(today_ads["cost"], today_ads["clicks"]),
            "ads_cpa": safe_div(today_ads["cost"], today_ads["conversions"]),
            "ads_roas": safe_div(today_ads["conversion_value"], today_ads["cost"]) if has_conv_value else None,
        }

        last_30_totals = sum_series(last_30_dates)
        last_30_totals["ads_ctr"] = safe_div(last_30_totals["ads_clicks"], last_30_totals["ads_impressions"]) * 100
        last_30_totals["ads_cpc"] = safe_div(last_30_totals["ads_cost"], last_30_totals["ads_clicks"])
        last_30_totals["ads_cpa"] = safe_div(last_30_totals["ads_cost"], last_30_totals["ads_conversions"])
        last_30_totals["ads_roas"] = (
            safe_div(last_30_totals["ads_conversion_value"], last_30_totals["ads_cost"]) if has_conv_value else None
        )

        all_totals = sum_series(all_dates)
        all_totals["ads_ctr"] = safe_div(all_totals["ads_clicks"], all_totals["ads_impressions"]) * 100
        all_totals["ads_cpc"] = safe_div(all_totals["ads_cost"], all_totals["ads_clicks"])
        all_totals["ads_cpa"] = safe_div(all_totals["ads_cost"], all_totals["ads_conversions"])
        all_totals["ads_roas"] = (
            safe_div(all_totals["ads_conversion_value"], all_totals["ads_cost"]) if has_conv_value else None
        )

        today_cards = self._format_cards(today_metrics, has_conv_value)
        last_30_cards = self._format_cards(last_30_totals, has_conv_value)
        all_time_cards = self._format_cards(all_totals, has_conv_value, include_avg=True, days=len(all_dates))

        return {
            "today_cards": today_cards,
            "last_30_cards": last_30_cards,
            "all_time_cards": all_time_cards,
            "total_days": len(all_dates),
        }

    def _format_cards(self, metrics: dict, has_conv_value: bool, include_avg: bool = False, days: int = 1) -> list[dict]:
        labels = [
            ("ga4_sessions", "GA4 세션", "회"),
            ("ga4_active_users", "GA4 활성 사용자", "명"),
            ("ads_cost", "광고 비용", "원"),
            ("ads_impressions", "광고 노출", "회"),
            ("ads_clicks", "광고 클릭", "회"),
            ("ads_conversions", "광고 전환", "건"),
            ("ads_ctr", "광고 클릭률", "%"),
            ("ads_cpc", "클릭당 비용", "원"),
            ("ads_cpa", "전환당 비용", "원"),
        ]
        if has_conv_value:
            labels.append(("ads_roas", "광고 수익률", "배"))

        cards = []
        for key, label, unit in labels:
            value = metrics.get(key)
            if value is None:
                continue
            if key in {"ads_cost", "ads_cpc", "ads_cpa"}:
                display = format_currency(value)
                unit_text = ""
            elif key == "ads_roas":
                display = format_float(value, 2)
                unit_text = unit
            elif key in {"ads_ctr"}:
                display = format_float(value, 2)
                unit_text = unit
            else:
                display = format_int(value)
                unit_text = unit

            sub = None
            if include_avg and key in {
                "ga4_sessions",
                "ga4_active_users",
                "ads_cost",
                "ads_impressions",
                "ads_clicks",
                "ads_conversions",
            }:
                avg_value = safe_div(metrics.get(key, 0), days)
                if key == "ads_cost":
                    sub = f"평균 {format_currency(avg_value)}/일"
                else:
                    sub = f"평균 {format_float(avg_value, 1)}{unit}/일"

            cards.append({
                "label": label,
                "value": display,
                "unit": unit_text,
                "sub": sub,
            })
        return cards

    def _build_tables(self, start_date: str, end_date: str) -> dict:
        landing_rows = self.ga4.run_report(
            ["landingPagePlusQueryString"],
            ["sessions", "activeUsers"],
            start_date,
            end_date,
            limit=100,
        )
        landing_rows.sort(key=lambda x: x.get("sessions", 0), reverse=True)
        landing_rows = landing_rows[:10]

        source_rows = self.ga4.run_report(
            ["sessionSource", "sessionMedium"],
            ["sessions", "activeUsers"],
            start_date,
            end_date,
            limit=200,
        )
        source_rows.sort(key=lambda x: x.get("sessions", 0), reverse=True)
        source_rows = source_rows[:10]

        campaign_query = (
            "SELECT campaign.name, metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions "
            "FROM campaign "
            f"WHERE segments.date BETWEEN '{start_date}' AND '{end_date}' "
            "AND campaign.status != 'REMOVED'"
        )
        campaign_rows = self.ads.run_query(campaign_query)

        campaigns = {}
        for row in campaign_rows:
            name = row.campaign.name
            if name not in campaigns:
                campaigns[name] = {
                    "campaign": name,
                    "impressions": 0,
                    "clicks": 0,
                    "cost": 0.0,
                    "conversions": 0.0,
                }
            campaigns[name]["impressions"] += row.metrics.impressions
            campaigns[name]["clicks"] += row.metrics.clicks
            campaigns[name]["cost"] += row.metrics.cost_micros / 1_000_000
            campaigns[name]["conversions"] += row.metrics.conversions

        campaign_list = []
        for campaign in campaigns.values():
            campaign["ctr"] = safe_div(campaign["clicks"], campaign["impressions"]) * 100
            campaign["cpc"] = safe_div(campaign["cost"], campaign["clicks"])
            campaign["cpa"] = safe_div(campaign["cost"], campaign["conversions"])
            campaign_list.append(campaign)
        campaign_list.sort(key=lambda x: x["cost"], reverse=True)
        campaign_list = campaign_list[:10]

        return {
            "landing_pages": {
                "headers": ["랜딩페이지", "세션", "활성 사용자"],
                "rows": [
                    [
                        row.get("landingPagePlusQueryString", "")[:80],
                        format_int(row.get("sessions", 0)),
                        format_int(row.get("activeUsers", 0)),
                    ]
                    for row in landing_rows
                ],
            },
            "source_medium": {
                "headers": ["소스", "미디엄", "세션", "활성 사용자"],
                "rows": [
                    [
                        row.get("sessionSource", ""),
                        row.get("sessionMedium", ""),
                        format_int(row.get("sessions", 0)),
                        format_int(row.get("activeUsers", 0)),
                    ]
                    for row in source_rows
                ],
            },
            "ads_campaigns": {
                "headers": ["캠페인", "비용", "노출", "클릭", "전환", "클릭률", "클릭당 비용", "전환당 비용"],
                "rows": [
                    [
                        row["campaign"],
                        format_currency(row["cost"]),
                        format_int(row["impressions"]),
                        format_int(row["clicks"]),
                        format_float(row["conversions"], 1),
                        f"{format_float(row['ctr'], 2)}%",
                        format_currency(row["cpc"]),
                        format_currency(row["cpa"]),
                    ]
                    for row in campaign_list
                ],
            },
        }

    def _build_chart_data(
        self,
        ga4_daily: dict,
        ads_daily: dict,
        last_30_dates: list[str],
        ga4_has_data: bool,
        ads_has_data: bool,
    ) -> list[dict]:
        charts = []
        chart_specs = [
            (
                "ga4_sessions_30d.png",
                "GA4 세션 추이 (최근 30일)",
                [ga4_daily[d]["sessions"] for d in last_30_dates],
                "세션",
                "count",
                ga4_has_data,
            ),
            (
                "ga4_active_users_30d.png",
                "GA4 활성 사용자 추이 (최근 30일)",
                [ga4_daily[d]["activeUsers"] for d in last_30_dates],
                "활성 사용자",
                "count",
                ga4_has_data,
            ),
            (
                "ads_cost_30d.png",
                "광고 비용 추이 (최근 30일)",
                [ads_daily[d]["cost"] for d in last_30_dates],
                "비용",
                "currency",
                ads_has_data,
            ),
            (
                "ads_conversions_30d.png",
                "광고 전환 추이 (최근 30일)",
                [ads_daily[d]["conversions"] for d in last_30_dates],
                "전환",
                "count",
                ads_has_data,
            ),
            (
                "ads_ctr_30d.png",
                "광고 클릭률 추이 (최근 30일)",
                [
                    safe_div(ads_daily[d]["clicks"], ads_daily[d]["impressions"]) * 100
                    for d in last_30_dates
                ],
                "클릭률",
                "percent",
                ads_has_data,
            ),
            (
                "ads_cpc_30d.png",
                "클릭당 비용 추이 (최근 30일)",
                [
                    safe_div(ads_daily[d]["cost"], ads_daily[d]["clicks"])
                    for d in last_30_dates
                ],
                "클릭당 비용",
                "currency",
                ads_has_data,
            ),
        ]

        for filename, title, values, y_label, value_type, has_data in chart_specs:
            charts.append({
                "filename": filename,
                "title": title,
                "values": values,
                "y_label": y_label,
                "value_type": value_type,
                "has_data": has_data,
                "dates": last_30_dates,
            })
        return charts

    def generate_charts(self, output_dir: Path, charts: list[dict]):
        output_dir.mkdir(parents=True, exist_ok=True)
        for chart in charts:
            path = output_dir / chart["filename"]
            self._plot_chart(
                path,
                chart["dates"],
                chart["values"],
                chart["title"],
                chart["value_type"],
                chart["has_data"],
            )

    def _plot_chart(
        self,
        path: Path,
        dates: list[str],
        values: list[float],
        title: str,
        value_type: str,
        has_data: bool,
    ):
        if not has_data:
            fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=150)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "데이터 없음",
                ha="center",
                va="center",
                fontsize=12,
                color="#6b7280",
            )
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return

        fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=150)
        ax.plot(dates, values, color=ACCENT_COLOR, linewidth=2)
        ax.fill_between(dates, values, color=ACCENT_COLOR, alpha=0.12)
        ax.set_title(title, fontsize=11, loc="left", pad=10)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        tick_step = max(1, len(dates) // 6)
        ax.set_xticks(dates[::tick_step])

        if value_type == "currency":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"₩{int(x):,}"))
        elif value_type == "percent":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def render_report(self, output_path: Path, context: dict):
        env = Environment(
            loader=FileSystemLoader("templates"),
            autoescape=select_autoescape(["html"]),
        )
        template = env.get_template("report_ko.html")
        html = template.render(**context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        print(f"리포트 저장됨: {output_path}")


def build_render_context(report_data: dict, chart_prefix: str, logo_png_path: str, logo_svg_path: str, start_date: str, end_date: str) -> dict:
    charts = []
    for chart in report_data["charts"]:
        charts.append({
            "title": chart["title"],
            "path": f"{chart_prefix}{chart['filename']}",
        })
    return {
        "report_title": REPORT_TITLE,
        "period_start": start_date,
        "period_end": end_date,
        "generated_at": datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M"),
        "summary": report_data["summary"],
        "tables": report_data["tables"],
        "charts": charts,
        "logo_png_path": logo_png_path,
        "logo_svg_path": logo_svg_path,
    }


def main():
    load_dotenv()
    property_id = os.getenv("PROPERTY_ID")
    customer_id = os.getenv("CUSTOMER_ID")
    if not property_id or not customer_id:
        print("오류: PROPERTY_ID와 CUSTOMER_ID가 필요합니다.")
        return

    start_date = FIXED_START_DATE
    end_date = seoul_today().isoformat()

    print(f"리포트 생성: {start_date} ~ {end_date}")
    generator = ReportGenerator(property_id, customer_id, start_date, end_date)
    report_data = generator.collect_all_data()

    report_dir = Path(f"reports/{end_date}")
    generator.generate_charts(report_dir, report_data["charts"])

    report_context = build_render_context(
        report_data,
        chart_prefix="",
        logo_png_path="../../assets/huelight-logo.png",
        logo_svg_path="../../assets/huelight-logo.svg",
        start_date=start_date,
        end_date=end_date,
    )
    generator.render_report(report_dir / "index.html", report_context)

    root_context = build_render_context(
        report_data,
        chart_prefix=f"reports/{end_date}/",
        logo_png_path="assets/huelight-logo.png",
        logo_svg_path="assets/huelight-logo.svg",
        start_date=start_date,
        end_date=end_date,
    )
    generator.render_report(Path("index.html"), root_context)
    print("\n✅ 완료: index.html 및 reports 히스토리 업데이트")


if __name__ == "__main__":
    main()
