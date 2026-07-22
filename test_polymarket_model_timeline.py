import dataclasses
import datetime as dt
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import polymarket_model_timeline as timeline


def market(question, date_label, yes, volume=1000):
    return {
        "question": question,
        "groupItemTitle": date_label,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": f'["{yes}", "{1 - yes}"]',
        "volume": str(volume),
    }


class TimelineTests(unittest.TestCase):
    def test_html_defaults_to_thirty_days_of_history(self):
        self.assertEqual(timeline.parse_args(["--html"]).history_days, 30)
        self.assertEqual(timeline.parse_args(["--html", "--history-days", "0"]).history_days, 0)
        self.assertTrue(timeline.parse_args(["--discovery-report"]).discovery_report)

    def test_html_public_metadata_and_source_link(self):
        output = timeline.render_html(
            [],
            dt.datetime(2027, 7, 1, tzinfo=dt.UTC),
            site_url="https://example.com/model-timeline",
            source_url="https://github.com/example/model-timeline?a=1&b=2",
        )
        self.assertIn('<meta name="description"', output)
        self.assertIn(f"<title>{timeline.SITE_TITLE}</title>", output)
        self.assertIn(f'<meta property="og:title" content="{timeline.SITE_TITLE}">', output)
        self.assertIn(f"<h1>{timeline.SITE_TITLE}</h1>", output)
        self.assertIn('<link rel="canonical" href="https://example.com/model-timeline/">', output)
        self.assertIn('<meta property="og:url" content="https://example.com/model-timeline/">', output)
        self.assertIn("Source and reuse terms", output)
        self.assertIn("a=1&amp;b=2", output)
        self.assertIn("not affiliated with Polymarket, OpenAI, Anthropic, or Google", output)
        self.assertIn("All rights reserved; no license is granted for reuse", output)
        self.assertIn("Polymarket market data remains third-party content", output)

        args = timeline.parse_args(
            ["--html", "--site-url", "https://example.com", "--source-url", "https://github.com/example/repo"]
        )
        self.assertEqual(args.site_url, "https://example.com")
        self.assertEqual(args.source_url, "https://github.com/example/repo")

    def test_historical_event_uses_latest_daily_yes_price(self):
        market_data = market("Will GPT-7 be released by August 31, 2027?", "August 31", 0.9)
        market_data.update(
            {
                "clobTokenIds": '["yes-token", "no-token"]',
                "createdAt": "2027-06-01T00:00:00Z",
                "bestBid": 0.8,
                "bestAsk": 0.9,
                "spread": 0.1,
                "closed": True,
                "acceptingOrders": False,
                "closedTime": "2027-07-12T12:00:00Z",
            }
        )
        event = {
            "id": "history",
            "slug": "gpt-history",
            "title": "GPT-7 released by...?",
            "description": "OpenAI public model release",
            "createdAt": "2027-06-01T00:00:00Z",
            "markets": [market_data],
        }
        timestamp = int(dt.datetime(2027, 7, 10, 12, tzinfo=dt.UTC).timestamp())
        events = timeline.historical_events_at(
            [event], {"yes-token": [{"t": timestamp, "p": 0.42}]}, dt.date(2027, 7, 10)
        )
        self.assertEqual(len(events), 1)
        historical_market = events[0]["markets"][0]
        self.assertEqual(json.loads(historical_market["outcomePrices"]), ["0.42", "0.5800000000000001"])
        self.assertNotIn("bestBid", historical_market)
        self.assertNotIn("bestAsk", historical_market)
        self.assertFalse(historical_market["closed"])
        self.assertTrue(historical_market["acceptingOrders"])

    def test_html_history_embeds_daily_snapshots_and_controls(self):
        earlier = timeline.TimelineItem(
            "OpenAI", "GPT", "release", dt.date(2027, 9, 1), "around Sep 1", "median", "", 1, "x", "1",
            distribution=(
                timeline.DistributionPoint(dt.date(2027, 8, 1), 0.25, 0.25),
                timeline.DistributionPoint(dt.date(2027, 9, 1), 0.5, 0.25),
                timeline.DistributionPoint(dt.date(2027, 10, 1), 0.75, 0.25),
            ),
            median_date=dt.date(2027, 9, 1),
        )
        current = dataclasses.replace(earlier, sort_date=dt.date(2027, 8, 15), median_date=dt.date(2027, 8, 15))
        snapshots = [
            timeline.HistorySnapshot(dt.date(2027, 7, 1), (earlier,)),
            timeline.HistorySnapshot(dt.date(2027, 7, 2), (current,)),
        ]
        output = timeline.render_html([current], dt.datetime(2027, 7, 2, tzinfo=dt.UTC), snapshots)
        self.assertIn('id="history-slider"', output)
        self.assertIn('id="history-prev"', output)
        self.assertIn('id="history-play"', output)
        self.assertIn('id="history-latest"', output)
        self.assertIn('data-overview-filter="all"', output)
        self.assertIn('data-overview-filter="active"', output)
        self.assertIn('data-overview-filter="active" aria-pressed="true"', output)
        self.assertIn('<div id="summary-container" data-overview-filter="active">', output)
        self.assertIn('let overviewFilter = "active";', output)
        self.assertIn('applyOverviewFilter(overviewFilter)', output)
        self.assertIn('latest.addEventListener("click"', output)
        self.assertIn('}, 470);', output)
        self.assertIn("renderHistory(snapshots.length - 1)", output)
        self.assertIn("renderHistory", output)
        match = re.search(r'<script type="application/json" id="history-data">(.*?)</script>', output, re.S)
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertEqual([snapshot["date"] for snapshot in payload], ["2027-07-01", "2027-07-02"])
        self.assertIn("Sep 1, 2027", payload[0]["summary_html"])
        self.assertIn("Aug 15, 2027", payload[1]["summary_html"])
        self.assertIn("Forecast as of Jul 1, 2027", payload[0]["summary_html"])
        self.assertIn("Forecast as of Jul 2, 2027", payload[1]["summary_html"])
        self.assertIn('class="as-of-line"', payload[0]["cards_html"])
        self.assertIn("Forecast as of Jul 1, 2027", payload[0]["cards_html"])
        self.assertIn("Forecast as of Jul 2, 2027", payload[1]["cards_html"])

    def test_html_output_is_standalone_and_escaped(self):
        item = timeline.TimelineItem(
            provider="OpenAI",
            title="GPT <script>alert(1)</script>",
            kind="release & change",
            sort_date=dt.date(2027, 8, 1),
            when="around August 1",
            estimate="median < 50%",
            detail="A & B",
            volume=1234,
            url="https://polymarket.com/event/gpt?a=1&b=2",
            event_id="1",
            distribution=(
                timeline.DistributionPoint(dt.date(2027, 7, 31), 0.2, 0.2),
                timeline.DistributionPoint(dt.date(2027, 8, 31), 0.7, 0.5),
            ),
            median_date=dt.date(2027, 8, 19),
        )
        output = timeline.render_html([item], dt.datetime(2027, 7, 1, tzinfo=dt.UTC))
        self.assertTrue(output.startswith("<!doctype html>"))
        self.assertIn("GPT &lt;script&gt;alert(1)&lt;/script&gt;", output)
        self.assertIn("a=1&amp;b=2", output)
        self.assertNotIn("<script>alert(1)</script>", output)
        self.assertIn('class="prob-chart"', output)
        self.assertIn("CDF · probability released by date", output)
        self.assertIn("PDF · probability rate per day", output)
        self.assertIn("1.6% per day", output)
        self.assertNotIn("interval probability", output)
        self.assertIn('data-chart-tooltip="', output)
        self.assertIn('id="chart-tooltip"', output)
        self.assertIn('class="month-grid"', output)
        self.assertIn('class="summary-card"', output)
        self.assertIn("All markets", output)
        self.assertIn("Active only", output)
        self.assertIn('#summary-container[data-overview-filter="active"] .summary-plot-row.unavailable', output)
        self.assertIn('button.setAttribute("aria-pressed"', output)
        self.assertEqual(output.count('class="summary-plot-row '), 1)
        self.assertEqual(output.count('class="summary-table-row"'), 1)
        self.assertIn("Median Aug 19, 2027", output)
        self.assertIn('class="summary-iqr open-right"', output)
        self.assertIn("75th percentile: beyond the Aug 31, 2027 market horizon", output)
        self.assertIn('href="#detail-openai-gpt-script-alert-1-script"', output)
        self.assertIn('id="detail-openai-gpt-script-alert-1-script"', output)
        self.assertNotIn('<td><a href="https://polymarket.com/', output)

    def test_summary_distinguishes_median_horizon_and_deadline(self):
        median_item = timeline.TimelineItem(
            "OpenAI", "GPT", "release", dt.date(2027, 8, 1), "around Aug 1", "median", "", 1, "x", "1",
            distribution=(
                timeline.DistributionPoint(dt.date(2027, 7, 25), 0.25, 0.25),
                timeline.DistributionPoint(dt.date(2027, 8, 1), 0.5, 0.25),
                timeline.DistributionPoint(dt.date(2027, 8, 8), 0.75, 0.25),
            ),
            median_date=dt.date(2027, 8, 1),
        )
        horizon_item = timeline.TimelineItem(
            "Google", "Gemini", "release", dt.date(2027, 8, 31), "Median not established", "20%", "", 1, "y", "2",
            distribution=(timeline.DistributionPoint(dt.date(2027, 8, 31), 0.2, 0.2),),
        )
        deadline_item = timeline.TimelineItem(
            "Anthropic", "Claude access", "change", dt.date(2027, 7, 19), "by Jul 19", "Yes 60% / No 40%", "", 1, "z", "3",
        )
        self.assertEqual(timeline.summary_date_signal(median_item)[0], "median")
        self.assertEqual(timeline.summary_date_signal(horizon_item)[0], "horizon")
        self.assertEqual(timeline.summary_date_signal(deadline_item)[0], "deadline")
        output = timeline.render_summary_card([median_item, horizon_item, deadline_item])
        self.assertEqual(output.count('class="summary-plot-row '), 3)
        self.assertEqual(output.count('summary-plot-row google unavailable'), 1)
        self.assertEqual(output.count('summary-plot-row anthropic unavailable'), 1)
        self.assertEqual(output.count('class="summary-table-row"'), 1)
        self.assertEqual(output.count('class="summary-table-row unavailable"'), 2)
        self.assertIn("No median; 20% probability by Aug 31, 2027", output)
        self.assertIn('class="summary-month-grid"', output)
        self.assertIn("GPT — OpenAI. Median Aug 1, 2027", output)
        self.assertNotIn("Market horizon, no median", output)
        self.assertNotIn("Decision deadline", output)
        self.assertIn('class="summary-iqr', output)
        self.assertIn("25th percentile: on or before Jul 25, 2027", output)
        self.assertIn("75th percentile: Aug 8, 2027", output)

        output_with_cursor = timeline.render_summary_card(
            [median_item, horizon_item, deadline_item],
            axis_range=(dt.date(2027, 7, 1), dt.date(2027, 9, 1)),
            as_of=dt.date(2027, 7, 15),
        )
        self.assertIn('class="summary-as-of"', output_with_cursor)
        self.assertIn('style="left: 22.58%"', output_with_cursor)
        self.assertIn("Forecast as of Jul 15, 2027", output_with_cursor)
        self.assertIn("Vertical cursor: forecast date", output_with_cursor)

    def test_history_overview_keeps_stable_roster_order_and_greys_missing_rows(self):
        gpt = timeline.TimelineItem(
            "OpenAI", "GPT", "release", dt.date(2027, 9, 1), "around Sep 1", "median", "", 1, "x", "1",
            distribution=(
                timeline.DistributionPoint(dt.date(2027, 8, 1), 0.25, 0.25),
                timeline.DistributionPoint(dt.date(2027, 9, 1), 0.5, 0.25),
                timeline.DistributionPoint(dt.date(2027, 10, 1), 0.75, 0.25),
            ),
            median_date=dt.date(2027, 9, 1),
        )
        claude = dataclasses.replace(
            gpt,
            provider="Anthropic",
            title="Claude",
            sort_date=dt.date(2027, 8, 15),
            when="around Aug 15",
            url="y",
            event_id="2",
            median_date=dt.date(2027, 8, 15),
        )
        snapshots = [
            timeline.HistorySnapshot(dt.date(2027, 7, 1), (gpt,)),
            timeline.HistorySnapshot(dt.date(2027, 7, 2), (claude,)),
        ]
        output = timeline.render_html([claude], dt.datetime(2027, 7, 2, tzinfo=dt.UTC), snapshots)
        match = re.search(r'<script type="application/json" id="history-data">(.*?)</script>', output, re.S)
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))

        row_pattern = r'class="summary-plot-row [^"]+" data-summary-key="([^"]+)"'
        orders = [re.findall(row_pattern, snapshot["summary_html"]) for snapshot in payload]
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(len(orders[0]), 2)
        self.assertIn('data-summary-key="anthropic::claude"', payload[0]["summary_html"])
        self.assertIn('summary-plot-row anthropic unavailable', payload[0]["summary_html"])
        self.assertIn('summary-plot-row openai unavailable', payload[1]["summary_html"])
        self.assertIn('href="#detail-openai-gpt"', payload[0]["summary_html"])
        self.assertIn('id="detail-openai-gpt"', payload[0]["cards_html"])
        self.assertNotIn('href="#detail-anthropic-claude"', payload[0]["summary_html"])
        self.assertIn('href="#detail-anthropic-claude"', payload[1]["summary_html"])
        self.assertIn('id="detail-anthropic-claude"', payload[1]["cards_html"])
        self.assertEqual(payload[0]["summary_html"].count("summary-table-row"), 2)
        self.assertEqual(payload[1]["summary_html"].count("summary-table-row"), 2)

    def test_closed_event_disappears_after_its_close_date(self):
        closed_market = market("Will Anthropic extend Claude access by July 12, 2027?", None, 1.0)
        closed_market.update(
            {
                "clobTokenIds": '["yes-token", "no-token"]',
                "createdAt": "2027-07-01T00:00:00Z",
                "closed": True,
                "closedTime": "2027-07-12T12:00:00Z",
            }
        )
        event = {
            "id": "closed",
            "slug": "closed-access",
            "title": "Will Anthropic extend Claude access by July 12?",
            "description": "Anthropic access extension",
            "createdAt": "2027-07-01T00:00:00Z",
            "active": True,
            "closed": True,
            "closedTime": "2027-07-12T12:00:00Z",
            "markets": [closed_market],
        }
        histories = {
            "yes-token": [
                {"t": int(dt.datetime(2027, 7, 10, 12, tzinfo=dt.UTC).timestamp()), "p": 0.5},
                {"t": int(dt.datetime(2027, 7, 12, 12, tzinfo=dt.UTC).timestamp()), "p": 1.0},
            ]
        }
        self.assertEqual(len(timeline.historical_events_at([event], histories, dt.date(2027, 7, 10))), 1)
        self.assertEqual(timeline.historical_events_at([event], histories, dt.date(2027, 7, 13)), [])

    def test_month_boundaries_are_calendar_aligned(self):
        self.assertEqual(
            timeline.month_boundaries(dt.date(2027, 11, 20), dt.date(2028, 2, 10)),
            [dt.date(2027, 12, 1), dt.date(2028, 1, 1), dt.date(2028, 2, 1)],
        )

    def test_axis_ticks_omit_endpoints_close_to_month_labels(self):
        ticks = timeline.axis_tick_dates(dt.date(2026, 6, 20), dt.date(2027, 1, 2))
        self.assertNotIn(dt.date(2026, 6, 20), ticks)
        self.assertNotIn(dt.date(2027, 1, 2), ticks)
        self.assertEqual(ticks[0], dt.date(2026, 7, 1))
        self.assertEqual(ticks[-1], dt.date(2027, 1, 1))

    def test_axis_ticks_keep_endpoints_when_there_are_no_month_boundaries(self):
        start = dt.date(2026, 7, 10)
        end = dt.date(2026, 7, 20)
        self.assertEqual(timeline.axis_tick_dates(start, end), [start, end])

    def test_pdf_interval_mass_is_converted_to_daily_rate(self):
        points = (
            timeline.DistributionPoint(dt.date(2027, 7, 1), 0.0, 0.0),
            timeline.DistributionPoint(dt.date(2027, 7, 11), 0.5, 0.5),
            timeline.DistributionPoint(dt.date(2027, 7, 12), 0.7, 0.2),
        )
        rates = timeline.daily_pdf_rates(points)
        self.assertAlmostEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[1], 0.05)
        self.assertAlmostEqual(rates[2], 0.2)

    def test_isotonic_regression_smooths_inversion(self):
        actual = timeline.isotonic_non_decreasing([0.2, 0.7, 0.6, 0.9])
        for observed, expected in zip(actual, [0.2, 0.65, 0.65, 0.9]):
            self.assertAlmostEqual(observed, expected)

    def test_cumulative_market_conditions_on_expired_zero_deadline(self):
        event = {
            "id": "0",
            "slug": "gpt-monthly",
            "title": "GPT-7 released by...?",
            "description": "OpenAI public model release",
            "active": True,
            "closed": False,
            "volume": 5000,
            "markets": [
                market("Will GPT-7 be released by June 30, 2027?", "June 30", 0.0),
                market("Will GPT-7 be released by July 31, 2027?", "July 31", 0.68),
            ],
        }
        item = timeline.cumulative_item(event, "OpenAI", as_of=dt.date(2027, 7, 14))
        self.assertIsNotNone(item)
        self.assertEqual(item.sort_date, dt.date(2027, 7, 26))
        self.assertNotIn("June 30", item.detail)

    def test_cumulative_market_interpolates_median(self):
        event = {
            "id": "1",
            "slug": "gpt-test",
            "title": "GPT-7 released by...?",
            "description": "OpenAI public model release",
            "active": True,
            "closed": False,
            "volume": 5000,
            "markets": [
                market("Will GPT-7 be released by August 1, 2027?", "August 1", 0.25),
                market("Will GPT-7 be released by August 11, 2027?", "August 11", 0.75),
            ],
        }
        item = timeline.event_to_item(event)
        self.assertIsNotNone(item)
        self.assertEqual(item.sort_date, dt.date(2027, 8, 6))
        self.assertEqual(item.when, "around Aug 6, 2027")

    def test_exact_date_market_produces_discrete_median(self):
        event = {
            "id": "2",
            "slug": "gemini-test",
            "title": "Next Google Gemini Pro Model released on...?",
            "description": "Google public release",
            "active": True,
            "closed": False,
            "negRisk": True,
            "volume": 6000,
            "markets": [
                market("Will Gemini be released on July 20, 2027?", "July 20", 0.2),
                market("Will Gemini be released on July 21, 2027?", "July 21", 0.6),
                market("No release by July 21, 2027?", "No release by July 21", 0.2),
            ],
        }
        item = timeline.event_to_item(event)
        self.assertIsNotNone(item)
        self.assertEqual(item.sort_date, dt.date(2027, 7, 21))
        self.assertEqual(item.median_date, dt.date(2027, 7, 21))
        self.assertIn("80%", item.detail)

    def test_by_and_on_variants_are_combined(self):
        by_event = {
            "id": "4",
            "slug": "opus-by",
            "title": "Next Claude Opus released by...?",
            "description": "Anthropic public model release",
            "active": True,
            "closed": False,
            "volume": 10_000,
            "markets": [
                market("Will the next Claude Opus be released by July 20, 2027?", "July 20", 0.4, 5000),
                market("Will the next Claude Opus be released by July 25, 2027?", "July 25", 0.8, 5000),
            ],
        }
        on_event = {
            "id": "5",
            "slug": "opus-on",
            "title": "Next Claude Opus Model released on...?",
            "description": "Anthropic public model release",
            "active": True,
            "closed": False,
            "negRisk": True,
            "volume": 2000,
            "markets": [
                market("Will the next Claude Opus be released on July 21, 2027?", "July 21", 0.1),
                market("Will the next Claude Opus be released on July 22, 2027?", "July 22", 0.4),
                market("Will the next Claude Opus not be released by July 25, 2027?", "No release by July 25", 0.5),
            ],
        }
        items = timeline.build_timeline([by_event, on_event])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Next Claude Opus")
        self.assertEqual(len(items[0].sources), 2)
        self.assertIn("combined by/on", items[0].kind)
        self.assertEqual(items[0].distribution[-1].cdf, 0.8)

    def test_combined_by_and_on_market_interpolates_median(self):
        by_event = {
            "id": "combined-by",
            "slug": "gemini-pro-by",
            "title": "Next Google Gemini Pro Model released by...?",
            "description": "Google public model release",
            "active": True,
            "closed": False,
            "volume": 10_000,
            "markets": [
                market("Will Gemini Pro be released by August 8, 2027?", "August 8", 0.35),
                market("Will Gemini Pro be released by August 31, 2027?", "August 31", 0.78),
            ],
        }
        on_event = {
            "id": "combined-on",
            "slug": "gemini-pro-on",
            "title": "Next Google Gemini Pro Model released on...?",
            "description": "Google public model release",
            "active": True,
            "closed": False,
            "negRisk": True,
            "volume": 2_000,
            "markets": [
                market("Will Gemini Pro be released on July 31, 2027?", "July 31", 0.2),
                market("Will Gemini Pro be released on August 1, 2027?", "August 1", 0.1),
                market("No release by August 31, 2027?", "No release by August 31", 0.7),
            ],
        }

        item = timeline.combined_release_item(
            [by_event, on_event], "Google", as_of=dt.date(2027, 7, 17)
        )

        self.assertIsNotNone(item)
        self.assertEqual(item.median_date, dt.date(2027, 8, 16))
        self.assertEqual(item.when, "around Aug 16, 2027")

    def test_low_horizon_probability_does_not_invent_median(self):
        event = {
            "id": "6",
            "slug": "gemini-four",
            "title": "Gemini 4.0 released by...?",
            "description": "Google public model release",
            "active": True,
            "closed": False,
            "volume": 9000,
            "markets": [
                market("Gemini 4.0 released by June 30, 2027?", "June 30", 0.0),
                market("Gemini 4.0 released by July 31, 2027?", "July 31", 0.03),
            ],
        }
        item = timeline.combined_release_item([event], "Google", as_of=dt.date(2027, 7, 17))
        self.assertIsNotNone(item)
        self.assertIsNone(item.median_date)
        self.assertEqual(item.when, "Median not established")
        self.assertNotIn("after", item.when.lower())
        self.assertIn("3.0%", item.estimate)

    def test_binary_access_market_is_not_treated_as_release_date(self):
        event = {
            "id": "3",
            "slug": "claude-access",
            "title": "Will Anthropic extend Claude paid-plan access by July 19?",
            "description": "Anthropic access extension",
            "active": True,
            "closed": False,
            "volume": 7000,
            "markets": [market("Will Anthropic extend Claude paid-plan access by July 19, 2027?", None, 0.45)],
        }
        item = timeline.event_to_item(event)
        self.assertIsNotNone(item)
        self.assertEqual(item.kind, "consumer change (binary)")
        self.assertEqual(item.estimate, "Yes 45% / No 55%")

    def test_multi_deadline_consumer_change_is_rendered_as_cumulative_forecast(self):
        event = {
            "id": "7",
            "slug": "fable-removal",
            "title": "Anthropic removes paid plan access for Fable 5 by...?",
            "description": "Consumer access change",
            "active": True,
            "closed": False,
            "volume": 20_000,
            "markets": [
                market("Will Anthropic remove paid plan access for Fable 5 by July 19, 2027?", "July 19", 0.01),
                market("Will Anthropic remove paid plan access for Fable 5 by July 31, 2027?", "July 31", 0.04),
            ],
        }
        item = timeline.consumer_change_item(event, "Anthropic", as_of=dt.date(2027, 7, 18))
        self.assertIsNotNone(item)
        self.assertEqual(item.kind, "consumer change forecast (multi-deadline)")
        self.assertIsNone(item.median_date)
        self.assertEqual(item.distribution[-1].cdf, 0.04)
        self.assertIn("Only 4.0% probability", item.estimate)
        chart = timeline.render_probability_chart(item, as_of=dt.date(2027, 7, 18))
        self.assertIn("probability change occurred by date", chart)
        self.assertIn("change rate", chart)
        self.assertIn('class="as-of-line"', chart)
        self.assertIn("Forecast as of Jul 18, 2027", chart)
        self.assertIn("as of Jul 18", chart)

    def test_resolved_consumer_change_dominates_later_deadlines(self):
        resolved_market = market(
            "Will Anthropic announce permanent paid plan access for Fable 5 by July 24, 2027?",
            "July 24",
            1.0,
        )
        resolved_market.update({"closed": True, "closedTime": "2027-07-18T05:03:57Z"})
        event = {
            "id": "8",
            "slug": "fable-permanent",
            "title": "Anthropic announces permanent paid plan access for Fable 5 by...?",
            "description": "Consumer access change",
            "active": True,
            "closed": False,
            "volume": 12_000,
            "markets": [
                resolved_market,
                market(
                    "Will Anthropic announce permanent paid plan access for Fable 5 by July 31, 2027?",
                    "July 31",
                    0.995,
                ),
            ],
        }
        item = timeline.consumer_change_item(event, "Anthropic", as_of=dt.date(2027, 7, 18))
        self.assertIsNotNone(item)
        self.assertEqual(item.kind, "consumer change (resolved)")
        self.assertEqual(item.sort_date, dt.date(2027, 7, 18))
        self.assertEqual(item.estimate, "Yes — change occurred")
        self.assertFalse(item.distribution)

    def test_discovery_tracks_new_then_unchanged_supported_event(self):
        event = {
            "id": "700000",
            "slug": "claude-access",
            "title": "Will Anthropic extend Claude paid-plan access by July 19?",
            "description": "Anthropic access extension",
            "active": True,
            "closed": False,
            "markets": [market("Will Anthropic extend Claude paid-plan access by July 19, 2027?", None, 0.45)],
        }

        def fake_get(path, params=None, retries=3):
            if path == "/public-search":
                return {"events": [event], "pagination": {"hasMore": False}}
            if path == "/events/keyset":
                return {"events": [event], "next_cursor": "end"}
            if path == "/events" and params and params.get("slug"):
                return [event]
            raise AssertionError((path, params))

        with mock.patch.object(timeline, "get_json", side_effect=fake_get):
            first = timeline.discover_event_candidates(["Claude"], (), tag_slugs=())
            second = timeline.discover_event_candidates(["Claude"], (), first.state, tag_slugs=())
        self.assertEqual(len(first.events), 1)
        self.assertEqual(first.decisions[0].classification, "accepted")
        self.assertEqual(first.decisions[0].change, "new")
        self.assertEqual(second.decisions[0].change, "unchanged")

    def test_discovery_rejects_provider_hardware_and_ranking_events(self):
        hardware = {
            "id": "9",
            "slug": "openai-hardware",
            "title": "Will OpenAI launch a consumer hardware product by...?",
            "active": True,
            "closed": False,
            "markets": [market("Will OpenAI launch consumer hardware by December 31, 2027?", None, 0.5)],
        }
        ranking = {
            "id": "10",
            "slug": "best-model",
            "title": "Which company has the best AI model end of July?",
            "description": "OpenAI and Anthropic are among the options.",
            "active": True,
            "closed": False,
            "markets": [market("Will OpenAI have the best model by July 31, 2027?", None, 0.5)],
        }
        self.assertEqual(timeline.classify_event(hardware)[2], "rejected")
        self.assertEqual(timeline.classify_event(ranking)[2], "rejected")

    def test_discovery_state_round_trips_and_report_explains_rejections(self):
        state = timeline.empty_discovery_state()
        state["events"] = {"x": {"event_id": "1", "title": "Example"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            timeline.save_discovery_state(path, state)
            self.assertEqual(timeline.load_discovery_state(path), state)

        decision = timeline.DiscoveryDecision(
            "1", "x", "Best AI model", "OpenAI", "unsupported", "rejected", "new",
            "Outside consumer scope.", ("search:OpenAI",),
        )
        result = timeline.DiscoveryResult((), (decision,), state)
        report = timeline.render_discovery_report(result, dt.datetime(2027, 7, 1, tzinfo=dt.UTC))
        self.assertIn("Polymarket discovery audit", report)
        self.assertIn("| new | rejected | OpenAI | unsupported |", report)
        self.assertIn("Outside consumer scope", report)


if __name__ == "__main__":
    unittest.main()
