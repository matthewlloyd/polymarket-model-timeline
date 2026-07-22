# AI Model Release Timeline — Prediction Market Forecasts

`polymarket_model_timeline.py` discovers active Polymarket events about consumer-facing OpenAI, Anthropic, and Google model releases and access changes, then prints a chronological Markdown timeline.

It uses the [public, unauthenticated Gamma API](https://docs.polymarket.com/api-reference/introduction) and Python's standard library only. Discovery follows Polymarket's [public search and event-fetching pattern](https://docs.polymarket.com/quickstart/fetching-data).

```bash
git clone https://github.com/matthewlloyd/ai-model-release-timeline.git
cd ai-model-release-timeline
python3 polymarket_model_timeline.py
```

Useful options:

```bash
# JSON for another program or dashboard
python3 polymarket_model_timeline.py --json

# Standalone responsive HTML page with 30 days of daily history
python3 polymarket_model_timeline.py --html > timeline.html

# Change the history window, or disable it for a smaller file
python3 polymarket_model_timeline.py --html --history-days 90 > timeline.html
python3 polymarket_model_timeline.py --html --history-days 0 > timeline.html

# Add canonical/share metadata and a source link for a public deployment
python3 polymarket_model_timeline.py --html \
  --site-url https://example.github.io/ai-model-release-timeline/ \
  --source-url https://github.com/example/ai-model-release-timeline \
  > timeline.html

# Add a discovery term or force-include a new event
python3 polymarket_model_timeline.py --query "Claude Fable" \
  --slug https://polymarket.com/event/some-new-event

# Audit what discovery accepted, rejected, or could not classify
python3 polymarket_model_timeline.py --discovery-report

# Use a particular state file, or run without persistent discovery state
python3 polymarket_model_timeline.py --state-file ./discovery-state.json
python3 polymarket_model_timeline.py --no-state

# Suppress very thin markets
python3 polymarket_model_timeline.py --min-volume 1000
```

Discovery combines paginated public search, provider/AI tag feeds, a descending event-ID feed, canonical event refetching, and an optional manual slug list. It stores event fingerprints and lifecycle metadata in `~/.polymarket-model-timeline/discovery-state.json` by default, allowing later runs to distinguish new, changed, unchanged, closed, rejected, and unsupported candidates. `--discovery-report` shows every audited candidate and the reason for its classification instead of silently dropping unsupported shapes.

Matching “released by” and “released on” events are combined into one forecast: deadline quotes calibrate the CDF while exact-date quotes shape the PDF between those anchors. Multi-deadline access and pricing changes are treated as cumulative forecasts too. If an earlier cumulative deadline has already resolved Yes, the card is marked as a resolved consumer change rather than presenting later redundant deadlines as independent forecasts. Single-deadline access/pricing markets remain binary probabilities.

HTML reports include a daily date slider with previous, next, playback, and latest-date controls; changing the date rerenders the overview, summary table, forecast cards, and PDF/CDF charts from historical CLOB token prices. The overview uses one fixed roster and row order across the complete history window. An item that has no median or was unavailable on the selected date retains its row at reduced opacity, so playback never causes rows to jump or disappear. Use the overview's **All markets / Active only** toggle to retain those greyed rows or hide them for the selected date; the choice remains applied during playback. Clicking an available model or change title in either the overview plot or table jumps to its detail card; unavailable historical titles remain plain text when no card exists for the selected date. Recent closed relevant events are retained as historical sources: they can appear in snapshots before closure and remain as unavailable rows afterward. The overview shows established median forecast dates with provider-colored 25–75% IQR lines. Combined by-deadline/exact-date forecasts interpolate the median where the calibrated CDF crosses 50%; exact-date-only markets retain their discrete bucket median. Arrows mark quartiles that extend beyond the priced market horizon. Chart marks have detailed hover tooltips, and vertical gridlines align with calendar month boundaries. If the available horizon never reaches 50%, the report says that no median is established instead of inventing a date beyond the final market.

Run the tests with:

```bash
python3 -m unittest -v
```

## Publish with GitHub Pages

The included `.github/workflows/pages.yml` workflow tests the generator, refreshes the public Polymarket data every six hours, and deploys the resulting standalone page to GitHub Pages. It can also be run manually from the repository's **Actions** tab. A failed refresh does not replace the last successful deployment.

To publish it:

1. Create a public GitHub repository and push this directory to its `main` branch.
2. In **Settings → Pages**, choose **GitHub Actions** as the source.
3. Run the **Refresh and deploy dashboard** workflow, or wait for its next scheduled run.

No API keys or repository secrets are required. The workflow derives the normal GitHub Pages URL automatically. If you later configure a custom domain, add a repository Actions variable named `SITE_URL` containing its full public URL so canonical and sharing metadata use that domain.

The workflow persists `data/discovery-state.json` after successful refreshes. This preserves market lifecycle history between GitHub's temporary runners and ensures the public repository continues to receive activity. Generated HTML is deployed as an artifact and is not committed to the source branch.

## Copyright and reuse

Copyright © 2026 Matthew Lloyd. All rights reserved.

This repository is public so its calculations and methodology can be inspected and discussed. It is not open-source software, and no permission is granted to copy, modify, redistribute, or use the project code or original presentation commercially without prior written permission. See [`RIGHTS.md`](RIGHTS.md) for the complete notice.

Market questions, prices, event metadata, and other data obtained from Polymarket are third-party content and are not licensed under this project's copyright notice.
