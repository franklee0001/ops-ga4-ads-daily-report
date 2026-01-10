"""
HueLight GA4/Google Ads 일일 리포트 생성기
"""

import json
import os
import sys
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
from matplotlib import font_manager


def set_korean_font():
    preferred_fonts = []
    if sys.platform == "darwin":
        preferred_fonts.extend(["AppleGothic", "Apple SD Gothic Neo"])
    elif sys.platform.startswith("linux"):
        preferred_fonts.extend(["Noto Sans CJK KR", "NanumGothic"])

    preferred_fonts.extend([
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "Malgun Gothic",
        "DejaVu Sans",
    ])

    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in preferred_fonts if name in available), None)
    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()

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


def safe_date_range(start: date, end: date) -> list[str]:
    if start > end:
        return []
    return iso_date_range(start, end)


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
        last_7_end = end - timedelta(days=1)
        last_7_start = max(start, last_7_end - timedelta(days=6))
        last_7_dates = safe_date_range(last_7_start, last_7_end)
        last_30_end = end - timedelta(days=1)
        last_30_complete_start = max(start, last_30_end - timedelta(days=29))
        last_30_complete_dates = safe_date_range(last_30_complete_start, last_30_end)
        prev_7_end = last_7_start - timedelta(days=1)
        prev_7_start = max(start, prev_7_end - timedelta(days=6))
        prev_7_dates = safe_date_range(prev_7_start, prev_7_end) if last_7_dates else []

        ga4_daily, ga4_has_data = self._get_ga4_daily_series(self.start_date, self.end_date, all_dates)
        ads_daily, ads_has_data, has_conv_value = self._get_ads_daily_series(self.start_date, self.end_date, all_dates)

        summary = self._build_summary(ga4_daily, ads_daily, last_30_dates, all_dates, has_conv_value)
        tables = self._build_tables(last_30_start.isoformat(), self.end_date)
        ai_summary = self._build_ai_summary(
            ga4_daily,
            ads_daily,
            last_7_dates,
            prev_7_dates,
            tables.get("top_landing"),
            tables.get("top_campaign"),
        )
        geo_map = self._get_geo_map(last_7_start, last_7_end) if last_7_dates else {
            "has_data": False,
            "chart_json": "[]",
            "start": last_7_start.isoformat(),
            "end": last_7_end.isoformat(),
            "total_active": 0,
            "total_active_display": format_int(0),
        }
        keyword_tables = self._get_ads_keyword_tables(last_7_start, last_7_end, last_30_complete_start, last_30_end)
        search_terms = self._get_ads_search_term_waste(last_7_start, last_7_end)
        wasted_summary = self._build_wasted_summary(last_7_dates, ads_daily, keyword_tables, search_terms)
        conversion_definitions = self._get_conversion_definitions(last_30_complete_start, last_30_end)
        exec_summary = self._build_executive_summary(
            last_7_start,
            last_7_end,
            wasted_summary,
            keyword_tables,
            search_terms,
            tables.get("campaigns_raw"),
        )

        return {
            "summary": summary,
            "tables": tables,
            "charts": self._build_chart_data(ga4_daily, ads_daily, last_30_dates, ga4_has_data, ads_has_data),
            "ai_summary": ai_summary,
            "geo_map": geo_map,
            "keyword_tables": keyword_tables,
            "search_terms": search_terms,
            "wasted_summary": wasted_summary,
            "conversion_definitions": conversion_definitions,
            "exec_summary": exec_summary,
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

    def _build_ai_summary(
        self,
        ga4_daily: dict,
        ads_daily: dict,
        last_7_dates: list[str],
        prev_7_dates: list[str],
        top_landing: dict | None,
        top_campaign: dict | None,
    ) -> dict:
        def sum_ga4(dates: list[str]) -> dict:
            return {
                "sessions": sum(ga4_daily[d]["sessions"] for d in dates) if dates else 0,
                "active_users": sum(ga4_daily[d]["activeUsers"] for d in dates) if dates else 0,
            }

        def sum_ads(dates: list[str]) -> dict:
            return {
                "cost": sum(ads_daily[d]["cost"] for d in dates) if dates else 0.0,
                "impressions": sum(ads_daily[d]["impressions"] for d in dates) if dates else 0,
                "clicks": sum(ads_daily[d]["clicks"] for d in dates) if dates else 0,
                "conversions": sum(ads_daily[d]["conversions"] for d in dates) if dates else 0.0,
            }

        def delta_pct(current: float, prev: float) -> float | None:
            if prev <= 0:
                return None
            return (current - prev) / prev * 100

        def trend_text(current: float, prev: float, unit: str = "", is_currency: bool = False) -> str:
            if is_currency:
                current_text = format_currency(current)
            elif unit == "%":
                current_text = f"{format_float(current, 2)}%"
            else:
                current_text = format_int(current)
                if unit:
                    current_text = f"{current_text}{unit}"
            delta = delta_pct(current, prev)
            if delta is None:
                return f"{current_text} (비교 데이터 없음)"
            direction = "증가" if delta >= 0 else "감소"
            return f"{current_text} (전주 대비 {abs(delta):.1f}% {direction})"

        last_7_ga4 = sum_ga4(last_7_dates)
        prev_7_ga4 = sum_ga4(prev_7_dates)
        last_7_ads = sum_ads(last_7_dates)
        prev_7_ads = sum_ads(prev_7_dates)

        last_ctr = safe_div(last_7_ads["clicks"], last_7_ads["impressions"]) * 100
        prev_ctr = safe_div(prev_7_ads["clicks"], prev_7_ads["impressions"]) * 100
        last_cpc = safe_div(last_7_ads["cost"], last_7_ads["clicks"])
        prev_cpc = safe_div(prev_7_ads["cost"], prev_7_ads["clicks"])
        last_cpa = safe_div(last_7_ads["cost"], last_7_ads["conversions"])
        prev_cpa = safe_div(prev_7_ads["cost"], prev_7_ads["conversions"])

        summary_parts = [
            f"세션 {trend_text(last_7_ga4['sessions'], prev_7_ga4['sessions'], unit='회')}",
            f"활성 사용자 {trend_text(last_7_ga4['active_users'], prev_7_ga4['active_users'], unit='명')}",
            f"광고 비용 {trend_text(last_7_ads['cost'], prev_7_ads['cost'], is_currency=True)}",
            f"전환 {trend_text(last_7_ads['conversions'], prev_7_ads['conversions'], unit='건')}",
        ]
        summary_text = (
            f"최근 7일 기준 {', '.join(summary_parts)}입니다. "
            "전주 대비 변화를 기준으로 핵심 지표를 요약했습니다."
        )

        landing_text = "데이터 없음"
        if top_landing:
            landing_text = f"{top_landing.get('landingPagePlusQueryString', '-')[:60]} (세션 {format_int(top_landing.get('sessions', 0))})"

        campaign_text = "데이터 없음"
        if top_campaign:
            campaign_text = f"{top_campaign.get('campaign', '-')[:60]} (비용 {format_currency(top_campaign.get('cost', 0.0))})"

        insights = [
            f"세션/활성 사용자: {trend_text(last_7_ga4['sessions'], prev_7_ga4['sessions'], unit='회')} / "
            f"{trend_text(last_7_ga4['active_users'], prev_7_ga4['active_users'], unit='명')}",
            f"광고 효율: CTR {trend_text(last_ctr, prev_ctr, unit='%')}, "
            f"CPC {trend_text(last_cpc, prev_cpc, is_currency=True)}, "
            f"CPA {trend_text(last_cpa, prev_cpa, is_currency=True)}",
            f"상위 성과: 랜딩페이지 {landing_text}, 캠페인 {campaign_text}",
        ]

        return {
            "text": summary_text,
            "insights": insights,
        }

    def _get_geo_map(self, start_date: date, end_date: date) -> dict:
        start = start_date.isoformat()
        end = end_date.isoformat()
        if start_date > end_date:
            return {
                "has_data": False,
                "chart_json": "[]",
                "start": start,
                "end": end,
                "total_active": 0,
                "total_active_display": format_int(0),
            }
        rows = self.ga4.run_report(
            ["country"],
            ["activeUsers", "sessions"],
            start,
            end,
            limit=200,
        )
        data_rows = []
        total_active = 0
        for row in rows:
            country = row.get("country", "")
            active_users = row.get("activeUsers", 0)
            try:
                active_users = int(active_users)
            except ValueError:
                active_users = 0
            if not country:
                continue
            total_active += active_users
            data_rows.append([country, active_users])
        data_rows.sort(key=lambda x: x[1], reverse=True)
        chart_data = [["국가", "활성 사용자"]] + data_rows
        return {
            "has_data": bool(data_rows),
            "chart_json": json.dumps(chart_data, ensure_ascii=False),
            "start": start,
            "end": end,
            "total_active": total_active,
            "total_active_display": format_int(total_active),
        }

    def _get_ads_keyword_rows(self, start_date: date, end_date: date) -> list[dict]:
        if start_date > end_date:
            return []
        start = start_date.isoformat()
        end = end_date.isoformat()
        query = (
            "SELECT campaign.name, ad_group.name, ad_group_criterion.keyword.text, "
            "ad_group_criterion.keyword.match_type, metrics.impressions, metrics.clicks, "
            "metrics.ctr, metrics.average_cpc, metrics.cost_micros, metrics.conversions, "
            "metrics.cost_per_conversion "
            "FROM keyword_view "
            f"WHERE segments.date BETWEEN '{start}' AND '{end}' "
            "AND campaign.advertising_channel_type = 'SEARCH' "
            "AND ad_group_criterion.status != 'REMOVED'"
        )
        rows = self.ads.run_query(query)
        results = []
        for row in rows:
            keyword = row.ad_group_criterion.keyword
            match_type = str(keyword.match_type)
            results.append({
                "campaign": row.campaign.name,
                "ad_group": row.ad_group.name,
                "keyword": keyword.text,
                "match_type": match_type,
                "impressions": row.metrics.impressions,
                "clicks": row.metrics.clicks,
                "ctr": row.metrics.ctr,
                "avg_cpc_micros": row.metrics.average_cpc,
                "cost_micros": row.metrics.cost_micros,
                "conversions": row.metrics.conversions,
                "cpa_micros": row.metrics.cost_per_conversion,
            })
        return results

    def _format_match_type(self, match_type: str) -> str:
        mapping = {
            "BROAD": "확장",
            "PHRASE": "구문",
            "EXACT": "일치",
        }
        return mapping.get(match_type, "기타")

    def _format_keyword_table(self, rows: list[dict]) -> dict:
        headers = [
            "캠페인",
            "광고그룹",
            "키워드",
            "매칭",
            "노출",
            "클릭",
            "클릭률",
            "클릭당 비용",
            "비용",
            "전환",
            "전환율",
            "전환당 비용",
        ]
        table_rows = []
        for row in rows:
            cost = row["cost_micros"] / 1_000_000
            conversions = row["conversions"]
            ctr = row["ctr"] * 100 if row["ctr"] else safe_div(row["clicks"], row["impressions"]) * 100
            avg_cpc = row["avg_cpc_micros"] / 1_000_000 if row["avg_cpc_micros"] else safe_div(cost, row["clicks"])
            cpa = row["cpa_micros"] / 1_000_000 if row["cpa_micros"] else safe_div(cost, conversions)
            conv_rate = safe_div(conversions, row["clicks"]) * 100
            table_rows.append([
                row["campaign"],
                row["ad_group"],
                row["keyword"],
                self._format_match_type(row["match_type"]),
                format_int(row["impressions"]),
                format_int(row["clicks"]),
                f"{format_float(ctr, 2)}%",
                format_currency(avg_cpc),
                format_currency(cost),
                format_float(conversions, 1),
                f"{format_float(conv_rate, 2)}%",
                format_currency(cpa) if conversions else "-",
            ])
        return {"headers": headers, "rows": table_rows}

    def _get_ads_keyword_tables(
        self,
        last_7_start: date,
        last_7_end: date,
        last_30_start: date,
        last_30_end: date,
    ) -> dict:
        rows_7 = self._get_ads_keyword_rows(last_7_start, last_7_end)
        rows_30 = self._get_ads_keyword_rows(last_30_start, last_30_end)

        rows_7_sorted = sorted(rows_7, key=lambda r: r["cost_micros"], reverse=True)[:20]
        rows_30_sorted = sorted(rows_30, key=lambda r: r["cost_micros"], reverse=True)[:20]
        wasted_7 = [r for r in rows_7 if r["conversions"] == 0 and r["cost_micros"] > 0]
        wasted_7_sorted = sorted(wasted_7, key=lambda r: r["cost_micros"], reverse=True)[:20]

        return {
            "last_7": {
                "start": last_7_start.isoformat(),
                "end": last_7_end.isoformat(),
                "top_cost": self._format_keyword_table(rows_7_sorted),
                "wasted": self._format_keyword_table(wasted_7_sorted),
                "rows": rows_7,
            },
            "last_30": {
                "start": last_30_start.isoformat(),
                "end": last_30_end.isoformat(),
                "top_cost": self._format_keyword_table(rows_30_sorted),
                "rows": rows_30,
            },
        }

    def _get_ads_search_term_waste(self, start_date: date, end_date: date) -> dict:
        if start_date > end_date:
            return {"start": start_date.isoformat(), "end": end_date.isoformat(), "rows": [], "table": {"headers": [], "rows": []}}
        start = start_date.isoformat()
        end = end_date.isoformat()
        query = (
            "SELECT campaign.name, ad_group.name, search_term_view.search_term, "
            "metrics.clicks, metrics.cost_micros, metrics.conversions, metrics.cost_per_conversion "
            "FROM search_term_view "
            f"WHERE segments.date BETWEEN '{start}' AND '{end}' "
            "AND campaign.advertising_channel_type = 'SEARCH' "
            "AND metrics.conversions = 0 "
            "AND metrics.cost_micros > 0"
        )
        rows = self.ads.run_query(query)
        results = []
        for row in rows:
            results.append({
                "campaign": row.campaign.name,
                "ad_group": row.ad_group.name,
                "term": row.search_term_view.search_term,
                "clicks": row.metrics.clicks,
                "cost_micros": row.metrics.cost_micros,
                "conversions": row.metrics.conversions,
                "cpa_micros": row.metrics.cost_per_conversion,
            })
        results_sorted = sorted(results, key=lambda r: r["cost_micros"], reverse=True)[:20]
        table_rows = []
        for row in results_sorted:
            cost = row["cost_micros"] / 1_000_000
            cpa = row["cpa_micros"] / 1_000_000 if row["cpa_micros"] else safe_div(cost, row["conversions"])
            table_rows.append([
                row["campaign"],
                row["ad_group"],
                row["term"],
                format_int(row["clicks"]),
                format_currency(cost),
                format_float(row["conversions"], 1),
                format_currency(cpa) if row["conversions"] else "-",
            ])
        return {
            "start": start,
            "end": end,
            "rows": results,
            "table": {
                "headers": ["캠페인", "광고그룹", "검색어", "클릭", "비용", "전환", "전환당 비용"],
                "rows": table_rows,
            },
        }

    def _build_wasted_summary(self, last_7_dates: list[str], ads_daily: dict, keyword_tables: dict, search_terms: dict) -> dict:
        total_cost = sum(ads_daily[d]["cost"] for d in last_7_dates) if last_7_dates else 0.0
        wasted_cost = 0.0
        source = "검색어"
        if search_terms["rows"]:
            wasted_cost = sum(row["cost_micros"] for row in search_terms["rows"]) / 1_000_000
        else:
            source = "키워드"
            wasted_rows = keyword_tables["last_7"]["rows"]
            wasted_cost = sum(row["cost_micros"] for row in wasted_rows if row["conversions"] == 0 and row["cost_micros"] > 0) / 1_000_000

        wasted_share = safe_div(wasted_cost, total_cost) * 100 if total_cost else 0.0
        return {
            "start": search_terms["start"],
            "end": search_terms["end"],
            "total_cost": total_cost,
            "wasted_cost": wasted_cost,
            "wasted_share": wasted_share,
            "source": source,
            "total_cost_display": format_currency(total_cost),
            "wasted_cost_display": format_currency(wasted_cost),
            "wasted_share_display": f"{format_float(wasted_share, 1)}%",
        }

    def _get_conversion_definitions(self, start_date: date, end_date: date) -> dict:
        def format_type(value: str) -> str:
            mapping = {
                "WEBPAGE": "웹페이지",
                "UPLOAD_CLICKS": "오프라인 클릭",
                "UPLOAD_CALLS": "오프라인 전화",
                "UPLOAD_CALLS_CONVERSION": "오프라인 전화 전환",
                "APP_DOWNLOAD": "앱 다운로드",
                "APP_INSTALLS": "앱 설치",
                "APP_IN_APP_ACTION": "앱 내 전환",
                "PHONE_CALL_LEAD": "전화 리드",
                "SUBMIT_LEAD_FORM": "리드 폼 제출",
                "STORE_VISITS": "매장 방문",
                "IMPORT": "가져오기",
            }
            return mapping.get(value, "기타")

        def format_category(value: str) -> str:
            mapping = {
                "DEFAULT": "기본",
                "PURCHASE": "구매",
                "SUBMIT_LEAD_FORM": "리드",
                "CONTACT": "문의",
                "PAGE_VIEW": "페이지뷰",
                "SIGNUP": "가입",
                "DOWNLOAD": "다운로드",
                "ADD_TO_CART": "장바구니",
                "BEGIN_CHECKOUT": "결제 시작",
                "SUBSCRIBE": "구독",
            }
            return mapping.get(value, "기타")

        actions_query = (
            "SELECT conversion_action.name, conversion_action.type, conversion_action.category, "
            "conversion_action.status, conversion_action.primary_for_goal "
            "FROM conversion_action "
            "WHERE conversion_action.status = 'ENABLED'"
        )
        action_rows = self.ads.run_query(actions_query)
        actions = []
        for row in action_rows:
            action = row.conversion_action
            actions.append({
                "name": action.name,
                "type": format_type(str(action.type)),
                "category": format_category(str(action.category)),
                "status": "사용",
                "primary": "예" if action.primary_for_goal else "아니오",
            })

        conversions_query = (
            "SELECT segments.conversion_action_name, metrics.conversions "
            "FROM customer "
            f"WHERE segments.date BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'"
        )
        conversions_fallback = (
            "SELECT segments.conversion_action_name, metrics.conversions "
            "FROM campaign "
            f"WHERE segments.date BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}' "
            "AND campaign.status != 'REMOVED'"
        )
        conversion_rows = self.ads.run_query(conversions_query, fallback_query=conversions_fallback)
        conv_by_action = {}
        for row in conversion_rows:
            name = row.segments.conversion_action_name
            if not name:
                continue
            if name not in conv_by_action:
                conv_by_action[name] = {"conversions": 0.0}
            conv_by_action[name]["conversions"] += row.metrics.conversions

        conv_rows = []
        for name, data in conv_by_action.items():
            conv_rows.append({
                "name": name,
                "conversions": data["conversions"],
            })
        conv_rows.sort(key=lambda r: r["conversions"], reverse=True)
        conv_table_rows = [
            [
                row["name"],
                format_float(row["conversions"], 1),
            ]
            for row in conv_rows[:20]
        ]

        return {
            "actions": {
                "headers": ["전환 액션", "유형", "카테고리", "상태", "기본 목표"],
                "rows": [
                    [action["name"], action["type"], action["category"], action["status"], action["primary"]]
                    for action in actions
                ],
            },
            "performance": {
                "headers": ["전환 액션", "전환"],
                "rows": conv_table_rows,
            },
        }

    def _build_executive_summary(
        self,
        start_date: date,
        end_date: date,
        wasted_summary: dict,
        keyword_tables: dict,
        search_terms: dict,
        campaigns_raw: list[dict] | None,
    ) -> dict:
        date_range = f"{start_date.isoformat()} ~ {end_date.isoformat()}"
        wasted_line = (
            f"낭비 비용 {wasted_summary['wasted_cost_display']} "
            f"(전체 광고비 {wasted_summary['total_cost_display']} 중 {wasted_summary['wasted_share_display']})."
        )

        action_items = []
        wasted_items = search_terms["rows"] if search_terms["rows"] else [
            row for row in keyword_tables["last_7"]["rows"] if row["conversions"] == 0 and row["cost_micros"] > 0
        ]
        for item in wasted_items[:2]:
            label = item.get("term") or item.get("keyword") or "상위 낭비 항목"
            action_items.append(f"'{label}' 제외/정제")
        action_line = "즉시 조치: " + (", ".join(action_items) if action_items else "낭비 항목이 없습니다.")

        best_keyword = None
        for row in keyword_tables["last_7"]["rows"]:
            if row["conversions"] > 0:
                cost = row["cost_micros"] / 1_000_000
                cpa = safe_div(cost, row["conversions"])
                if best_keyword is None or cpa < best_keyword["cpa"]:
                    best_keyword = {"keyword": row["keyword"], "cpa": cpa}

        best_campaign = None
        if campaigns_raw:
            for row in campaigns_raw:
                if row["conversions"] > 0:
                    if best_campaign is None or row["cpa"] < best_campaign["cpa"]:
                        best_campaign = {"campaign": row["campaign"], "cpa": row["cpa"]}

        opportunity = "기회: "
        parts = []
        if best_keyword:
            parts.append(f"키워드 '{best_keyword['keyword']}' 전환당 비용 {format_currency(best_keyword['cpa'])}")
        if best_campaign:
            parts.append(f"캠페인 '{best_campaign['campaign']}' 전환당 비용 {format_currency(best_campaign['cpa'])}")
        opportunity += ", ".join(parts) if parts else "전환이 있는 키워드/캠페인이 없습니다."

        return {
            "range": date_range,
            "lines": [
                f"이번 주 결론({date_range}): 비용 대비 전환 성과를 점검해야 합니다.",
                wasted_line,
                action_line,
                opportunity,
            ],
        }

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
        top_landing = landing_rows[0] if landing_rows else None

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
        top_campaign = campaign_list[0] if campaign_list else None

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
            "top_landing": top_landing,
            "top_campaign": top_campaign,
            "campaigns_raw": campaign_list,
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


def build_render_context(
    report_data: dict,
    chart_prefix: str,
    logo_png_path: str,
    logo_svg_path: str,
    logo_url: str | None,
    start_date: str,
    end_date: str,
) -> dict:
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
        "logo_url": logo_url,
        "ai_summary": report_data["ai_summary"],
        "geo_map": report_data["geo_map"],
        "keyword_tables": report_data["keyword_tables"],
        "search_terms": report_data["search_terms"],
        "wasted_summary": report_data["wasted_summary"],
        "conversion_definitions": report_data["conversion_definitions"],
        "exec_summary": report_data["exec_summary"],
    }


def main():
    load_dotenv()
    property_id = os.getenv("PROPERTY_ID")
    customer_id = os.getenv("CUSTOMER_ID")
    logo_url = os.getenv("LOGO_URL", "").strip() or None
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
        logo_url=logo_url,
        start_date=start_date,
        end_date=end_date,
    )
    generator.render_report(report_dir / "index.html", report_context)

    root_context = build_render_context(
        report_data,
        chart_prefix=f"reports/{end_date}/",
        logo_png_path="assets/huelight-logo.png",
        logo_svg_path="assets/huelight-logo.svg",
        logo_url=logo_url,
        start_date=start_date,
        end_date=end_date,
    )
    generator.render_report(Path("index.html"), root_context)
    print("\n✅ 완료: index.html 및 reports 히스토리 업데이트")


if __name__ == "__main__":
    main()
