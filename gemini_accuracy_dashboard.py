"""
Gemini accuracy dashboard - visualizes auto-training metrics and performance.

Features:
- Real-time accuracy trends
- Confidence distribution charts
- Auto-enrollment success rates
- Retraining flag analysis
- Interactive dashboard with multiple views

Usage:
    python gemini_accuracy_dashboard.py
    
    Then visit http://localhost:5000 in your browser

Install requirements:
    pip install flask plotly pandas numpy
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

try:
    import flask
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
except ImportError:
    print("ERROR: Install required packages with: pip install flask plotly pandas")
    exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiAccuracyAnalyzer:
    """Analyzes Gemini auto-training audit logs and computes accuracy metrics."""

    def __init__(self, audit_log_path: Path):
        """
        Initialize analyzer.
        
        Args:
            audit_log_path: Path to auto_training_audit.log
        """
        self.audit_log_path = audit_log_path
        self.entries: List[Dict] = []
        self._load_audit_log()

    def _load_audit_log(self) -> None:
        """Load and parse audit log entries."""
        self.entries = []
        if not self.audit_log_path.exists():
            logger.warning(f"Audit log not found: {self.audit_log_path}")
            return

        try:
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self.entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error loading audit log: {e}")

    def get_summary_stats(self) -> Dict:
        """Get overall accuracy statistics."""
        stats = {
            "total_entries": len(self.entries),
            "auto_enrollments": 0,
            "enrollments_approved": 0,
            "enrollments_rejected": 0,
            "retraining_flags": 0,
            "pending_confirmations": 0,
            "avg_gemini_confidence": 0.0,
            "new_identities": 0,
        }

        confidence_scores = []

        for entry in self.entries:
            action = entry.get("action", "")

            if action == "auto_enroll":
                stats["auto_enrollments"] += 1
                conf = entry.get("confidence", 0.0)
                if isinstance(conf, (int, float)):
                    confidence_scores.append(conf)
            elif action == "enrollment_rejected":
                stats["enrollments_rejected"] += 1
            elif action == "enroll_pending":
                stats["pending_confirmations"] += 1
            elif action == "face_enrolled":
                stats["enrollments_approved"] += 1
            elif action == "flag_retraining":
                stats["retraining_flags"] += 1
            elif action == "new_identity_created":
                stats["new_identities"] += 1

        if confidence_scores:
            stats["avg_gemini_confidence"] = float(np.mean(confidence_scores))

        return stats

    def get_confidence_distribution(self, bins: int = 10) -> Dict:
        """Get distribution of Gemini confidence scores."""
        confidence_scores = []

        for entry in self.entries:
            if entry.get("action") in ["auto_enroll", "enroll_pending"]:
                conf = entry.get("confidence", 0.0)
                if isinstance(conf, (int, float)):
                    confidence_scores.append(conf)

        if not confidence_scores:
            return {"bins": [], "counts": [], "range": [0, 1]}

        hist, bin_edges = np.histogram(confidence_scores, bins=bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            "bins": bin_centers.tolist(),
            "counts": hist.tolist(),
            "range": [0, 1],
            "mean": float(np.mean(confidence_scores)),
            "median": float(np.median(confidence_scores)),
            "std": float(np.std(confidence_scores)),
        }

    def get_timeline_data(self, hours: int = 24) -> Dict:
        """Get accuracy metrics over time."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Group by hour
        hourly = defaultdict(lambda: {
            "auto_enrollments": 0,
            "approved": 0,
            "rejected": 0,
            "avg_confidence": [],
            "flags": 0,
        })

        for entry in self.entries:
            try:
                ts = datetime.fromisoformat(entry.get("timestamp", ""))
                if ts < cutoff:
                    continue

                hour_key = ts.strftime("%Y-%m-%d %H:00")
                action = entry.get("action", "")

                if action == "auto_enroll":
                    hourly[hour_key]["auto_enrollments"] += 1
                    conf = entry.get("confidence", 0.0)
                    if isinstance(conf, (int, float)):
                        hourly[hour_key]["avg_confidence"].append(conf)
                elif action == "face_enrolled":
                    hourly[hour_key]["approved"] += 1
                elif action == "enrollment_rejected":
                    hourly[hour_key]["rejected"] += 1
                elif action == "flag_retraining":
                    hourly[hour_key]["flags"] += 1
            except Exception:
                continue

        # Compute averages
        timeline = []
        for hour_key in sorted(hourly.keys()):
            data = hourly[hour_key]
            avg_conf = np.mean(data["avg_confidence"]) if data["avg_confidence"] else 0.0
            timeline.append({
                "timestamp": hour_key,
                "auto_enrollments": data["auto_enrollments"],
                "approved": data["approved"],
                "rejected": data["rejected"],
                "avg_confidence": float(avg_conf),
                "retraining_flags": data["flags"],
            })

        return timeline

    def get_identity_stats(self) -> Dict[str, Dict]:
        """Get statistics per identity."""
        stats = defaultdict(lambda: {
            "enrolled_faces": 0,
            "approvals": 0,
            "rejections": 0,
            "avg_confidence": [],
        })

        for entry in self.entries:
            label = entry.get("label", "UNKNOWN")
            action = entry.get("action", "")

            if action in ["auto_enroll", "enroll_pending"]:
                conf = entry.get("confidence", 0.0)
                if isinstance(conf, (int, float)):
                    stats[label]["avg_confidence"].append(conf)
            elif action == "face_enrolled":
                stats[label]["enrolled_faces"] += 1
                stats[label]["approvals"] += 1
            elif action == "enrollment_rejected":
                stats[label]["rejections"] += 1

        # Compute averages
        result = {}
        for label, data in stats.items():
            result[label] = {
                "enrolled_faces": data["enrolled_faces"],
                "approvals": data["approvals"],
                "rejections": data["rejections"],
                "avg_confidence": float(np.mean(data["avg_confidence"])) if data["avg_confidence"] else 0.0,
            }

        return result

    def get_recent_decisions(self, limit: int = 20) -> List[Dict]:
        """Get recent auto-training decisions."""
        recent = []
        for entry in reversed(self.entries[-limit:]):
            action = entry.get("action", "")
            if action in ["auto_enroll", "enroll_pending", "face_enrolled", "enrollment_rejected", "flag_retraining"]:
                recent.append({
                    "timestamp": entry.get("timestamp", ""),
                    "action": action,
                    "label": entry.get("label", "?"),
                    "confidence": entry.get("confidence", 0.0),
                })
        return list(reversed(recent))


# Initialize Flask app
app = flask.Flask(__name__)
AUDIT_LOG_PATH = Path(__file__).resolve().parent / "models" / "auto_training_audit.log"


@app.route("/")
def index():
    """Main dashboard view."""
    analyzer = GeminiAccuracyAnalyzer(AUDIT_LOG_PATH)
    stats = analyzer.get_summary_stats()

    # Confidence distribution chart
    conf_dist = analyzer.get_confidence_distribution()
    confidence_chart = go.Figure(data=[
        go.Bar(
            x=[f"{b:.1f}" for b in conf_dist["bins"]],
            y=conf_dist["counts"],
            marker_color="lightblue",
        )
    ])
    confidence_chart.update_layout(
        title="Gemini Confidence Score Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        hovermode="x unified",
    )

    # Timeline chart
    timeline = analyzer.get_timeline_data(hours=24)
    if timeline:
        timeline_df = pd.DataFrame(timeline)
        timeline_chart = go.Figure()
        timeline_chart.add_trace(go.Scatter(
            x=timeline_df["timestamp"],
            y=timeline_df["auto_enrollments"],
            mode="lines+markers",
            name="Auto Enrollments",
            line=dict(color="green"),
        ))
        timeline_chart.add_trace(go.Scatter(
            x=timeline_df["timestamp"],
            y=timeline_df["approved"],
            mode="lines+markers",
            name="Approved",
            line=dict(color="blue"),
        ))
        timeline_chart.add_trace(go.Scatter(
            x=timeline_df["timestamp"],
            y=timeline_df["rejected"],
            mode="lines+markers",
            name="Rejected",
            line=dict(color="red"),
        ))
        timeline_chart.update_layout(
            title="Auto-Training Activity (Last 24 Hours)",
            xaxis_title="Time",
            yaxis_title="Count",
            hovermode="x unified",
        )
        timeline_chart_html = timeline_chart.to_html(full_html=False)
    else:
        timeline_chart_html = "<p>No data available yet</p>"

    confidence_chart_html = confidence_chart.to_html(full_html=False)

    # Identity stats table
    identity_stats = analyzer.get_identity_stats()
    identity_rows = []
    for label, data in sorted(identity_stats.items(), key=lambda x: x[1]["enrolled_faces"], reverse=True):
        identity_rows.append({
            "label": label,
            "enrolled": data["enrolled_faces"],
            "approved": data["approvals"],
            "rejected": data["rejections"],
            "avg_conf": f"{data['avg_confidence']:.2f}",
        })

    # Recent decisions
    recent = analyzer.get_recent_decisions(limit=10)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gemini Accuracy Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #666;
                margin-top: 30px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-card.success {{
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            }}
            .stat-card.warning {{
                background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
            }}
            .stat-card.danger {{
                background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
            }}
            .stat-card h3 {{
                margin: 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .stat-card .value {{
                font-size: 32px;
                font-weight: bold;
                margin: 10px 0 0 0;
            }}
            .chart-container {{
                margin-bottom: 30px;
                padding: 15px;
                background-color: #fafafa;
                border-radius: 8px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .recent-decisions {{
                margin-top: 30px;
            }}
            .decision-item {{
                padding: 10px;
                border-left: 4px solid #4CAF50;
                background-color: #f9f9f9;
                margin-bottom: 10px;
                border-radius: 4px;
            }}
            .decision-item.rejected {{
                border-left-color: #f44336;
            }}
            .refresh-info {{
                text-align: right;
                color: #999;
                font-size: 12px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Gemini Facial Recognition Auto-Training Dashboard</h1>
            
            <div class="stats-grid">
                <div class="stat-card success">
                    <h3>Total Entries</h3>
                    <div class="value">{stats['total_entries']}</div>
                </div>
                <div class="stat-card success">
                    <h3>Auto Enrollments</h3>
                    <div class="value">{stats['auto_enrollments']}</div>
                </div>
                <div class="stat-card success">
                    <h3>Approved</h3>
                    <div class="value">{stats['enrollments_approved']}</div>
                </div>
                <div class="stat-card warning">
                    <h3>Pending Review</h3>
                    <div class="value">{stats['pending_confirmations']}</div>
                </div>
                <div class="stat-card danger">
                    <h3>Rejected</h3>
                    <div class="value">{stats['enrollments_rejected']}</div>
                </div>
                <div class="stat-card danger">
                    <h3>Retraining Flags</h3>
                    <div class="value">{stats['retraining_flags']}</div>
                </div>
                <div class="stat-card">
                    <h3>Avg Confidence</h3>
                    <div class="value">{stats['avg_gemini_confidence']:.2%}</div>
                </div>
                <div class="stat-card">
                    <h3>New Identities</h3>
                    <div class="value">{stats['new_identities']}</div>
                </div>
            </div>

            <div class="chart-container">
                {confidence_chart_html}
            </div>

            <div class="chart-container">
                {timeline_chart_html}
            </div>

            <h2>Performance by Identity</h2>
            <table>
                <tr>
                    <th>Identity</th>
                    <th>Enrolled Faces</th>
                    <th>Approvals</th>
                    <th>Rejections</th>
                    <th>Avg Confidence</th>
                </tr>
    """

    for row in identity_rows:
        html += f"""
                <tr>
                    <td><strong>{row['label']}</strong></td>
                    <td>{row['enrolled']}</td>
                    <td>{row['approved']}</td>
                    <td>{row['rejected']}</td>
                    <td>{row['avg_conf']}</td>
                </tr>
        """

    html += """
            </table>

            <div class="recent-decisions">
                <h2>Recent Auto-Training Decisions</h2>
    """

    for decision in recent:
        class_name = "rejected" if decision["action"] == "enrollment_rejected" else ""
        html += f"""
                <div class="decision-item {class_name}">
                    <strong>{decision['action']}</strong>: {decision['label']} 
                    (confidence: {decision['confidence']:.2f}) 
                    <br><small>{decision['timestamp']}</small>
                </div>
        """

    html += """
            </div>

            <div class="refresh-info">
                Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
                <br>Auto-refresh: <a href="javascript:location.reload()">Refresh</a> 
                or <a href="javascript:setInterval('location.reload()', 5000)">Auto (5s)</a>
            </div>
        </div>
    </body>
    </html>
    """

    return html


@app.route("/api/summary")
def api_summary():
    """API endpoint for summary stats."""
    analyzer = GeminiAccuracyAnalyzer(AUDIT_LOG_PATH)
    return flask.jsonify(analyzer.get_summary_stats())


@app.route("/api/confidence-dist")
def api_confidence_dist():
    """API endpoint for confidence distribution."""
    analyzer = GeminiAccuracyAnalyzer(AUDIT_LOG_PATH)
    return flask.jsonify(analyzer.get_confidence_distribution())


@app.route("/api/timeline")
def api_timeline():
    """API endpoint for timeline data."""
    analyzer = GeminiAccuracyAnalyzer(AUDIT_LOG_PATH)
    return flask.jsonify(analyzer.get_timeline_data(hours=24))


@app.route("/api/identities")
def api_identities():
    """API endpoint for identity stats."""
    analyzer = GeminiAccuracyAnalyzer(AUDIT_LOG_PATH)
    return flask.jsonify(analyzer.get_identity_stats())


if __name__ == "__main__":
    print(f"Starting Gemini Accuracy Dashboard...")
    print(f"Audit log: {AUDIT_LOG_PATH}")
    print(f"\nDashboard available at: http://localhost:5000")
    print(f"API endpoints:")
    print(f"  - /api/summary - Overall statistics")
    print(f"  - /api/confidence-dist - Confidence distribution")
    print(f"  - /api/timeline - Timeline data (24h)")
    print(f"  - /api/identities - Per-identity stats")
    print(f"\nPress Ctrl+C to stop\n")

    app.run(debug=False, host="0.0.0.0", port=5000)
