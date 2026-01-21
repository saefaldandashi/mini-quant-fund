# Automated Reporting System

## Overview

The reporting system generates executive-quality reports for the quantitative investment bot:
- **Daily Reports**: Tactical monitoring and performance
- **Weekly Reports**: Trend analysis and attribution
- **Monthly Reports**: Strategic review and diagnostics

## Features

- ✅ Human-readable narratives
- ✅ Visual charts and tables
- ✅ PDF export (one-click)
- ✅ Deterministic generation (no look-ahead bias)
- ✅ Integration with all system components

## Usage

### CLI

```bash
# Generate daily report
python generate_report.py --type daily --date 2026-01-20

# Generate weekly report
python generate_report.py --type weekly --week 2026-W03

# Generate monthly report
python generate_report.py --type monthly --month 2026-01
```

### API

```bash
# Generate report via API
curl -X POST http://localhost:5000/api/reports/generate \
  -H "Content-Type: application/json" \
  -d '{"type": "daily", "date": "2026-01-20"}'

# List available reports
curl http://localhost:5000/api/reports/list

# Download report
curl http://localhost:5000/api/reports/download/daily_report_20260120.pdf
```

### UI

Use the "Generate Report" button in the web interface.

## Report Structure

### Daily Report

1. **Executive Snapshot**
   - Portfolio return (1d)
   - Benchmark comparison
   - Risk metrics
   - Regime classification

2. **Portfolio Overview**
   - Top holdings table
   - Holdings bar chart
   - Exposure breakdown

3. **Strategy Activity**
   - Strategy weights
   - Debate summary
   - Strategy contribution

4. **Performance Attribution**
   - Asset-level contribution
   - Strategy-level contribution
   - Transaction costs

5. **Macro & News Brief**
   - Top macro events
   - Macro indices changes
   - Risk sentiment

6. **Risk Monitoring**
   - Volatility metrics
   - Drawdown chart
   - Risk alerts

### Weekly Report

Similar structure but with:
- Weekly aggregations
- Trend analysis
- Rolling metrics
- Positioning evolution

### Monthly Report

Strategic focus:
- Full performance review
- Strategy effectiveness
- Regime analysis
- System diagnostics

## Output Location

Reports are saved to:
```
outputs/reports/
├── daily_report_YYYYMMDD.pdf
├── weekly_report_YYYY-Wxx.pdf
└── monthly_report_YYYY-MM.pdf
```

Chart images are saved to:
```
outputs/reports/charts/
```

## Dependencies

- `matplotlib` - Chart generation
- `weasyprint` - PDF export
- `pandas` - Data manipulation

## Configuration

Customize report sections in:
- `report_templates.py` - Report structure
- `charts.py` - Chart styling
- `narrative.py` - Text templates

## Testing

```bash
# Generate test report
python generate_report.py --type daily --date 2026-01-20

# Verify PDF was created
ls -lh outputs/reports/daily_report_*.pdf
```
