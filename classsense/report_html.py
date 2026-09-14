# classsense/report_html.py
# Renders a session report as one self-contained HTML page.
#
# No CDN, no external stylesheet, no charting library: a Pi in a classroom may
# have no route to the internet, and the report has to be readable from a phone
# on the local network and from a file copied onto a laptop months later. Every
# chart is inline SVG built from the numbers.
#
# Colour follows the project's status palette (good / warning / critical /
# neutral) because Attentive, Sleepy and Distracted are states, not series.
# Status colour never carries meaning alone here - every state is labelled in
# the legend, in the table, and in each tooltip - which is also what makes the
# palette's sub-3:1 warning hue safe on a light surface.

import html
import json

# ── palette ────────────────────────────────────
# Status roles, fixed. Distinct from any categorical series colour so a state
# can never impersonate one.
STATE_COLOR = {
    "Attentive":   "var(--good)",
    "Distracted":  "var(--warning)",
    "Sleepy":      "var(--critical)",
    "Unknown":     "var(--muted-fill)",
    "Unmonitored": "var(--muted-fill)",
}
# Sleepy takes the strongest colour rather than Distracted: a sleeping student
# is the finding a teacher most wants pulled out of a lesson.
STACK_ORDER = ["Attentive", "Distracted", "Sleepy", "Unknown", "Unmonitored"]

CSS = """
:root{
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --muted-fill:#c3c2b7;
  --accent:#2a78d6;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19;
    --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --muted-fill:#4a4a46; --accent:#3987e5;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19;
  --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --muted-fill:#4a4a46; --accent:#3987e5;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:24px 20px 64px}
header{margin-bottom:22px}
h1{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:14px;margin:0 0 2px;font-weight:600}
.sub{color:var(--ink-2);font-size:13px}
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:16px 18px;margin-bottom:16px}
.hero{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.hero .fig{font-size:52px;font-weight:650;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;line-height:1}
.hero .of{color:var(--ink-2);font-size:13px;max-width:46ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:13px 15px}
.kpi .v{font-size:25px;font-weight:620;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;line-height:1.15}
.kpi .k{color:var(--ink-2);font-size:12px;margin-top:3px}
.kpi .n{color:var(--muted);font-size:11px;margin-top:5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 10px;
  font-size:12px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block;
  outline:1px solid var(--border)}
.note{color:var(--ink-2);font-size:12.5px;margin-top:10px;max-width:78ch}
table{width:100%;border-collapse:collapse;font-size:13px;
  font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;color:var(--ink-2);font-size:12px;
  padding:7px 8px;border-bottom:1px solid var(--axis);white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid var(--grid)}
td.num,th.num{text-align:right}
tr:last-child td{border-bottom:none}
.tag{display:inline-flex;align-items:center;gap:5px;font-size:12px}
.bar{display:inline-block;width:110px;height:8px;border-radius:4px;
  background:var(--grid);position:relative;vertical-align:middle;
  outline:1px solid var(--border)}
.bar>i{position:absolute;inset:0 auto 0 0;border-radius:4px;display:block;
  min-width:2px}
.scroll{overflow-x:auto}
svg{display:block;max-width:100%;height:auto}
svg text{font:11px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  fill:var(--muted)}
svg .lbl{fill:var(--ink-2)}
.hit{fill:transparent;cursor:crosshair}
.hit:hover~.mk,.mk:hover{opacity:1}
.warn{border-left:3px solid var(--warning);padding-left:13px}
footer{color:var(--muted);font-size:12px;margin-top:26px}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--grid);padding:1px 5px;border-radius:4px}
"""

TOOLTIP_JS = """
(function(){
  var tip=document.createElement('div');
  tip.style.cssText='position:fixed;pointer-events:none;z-index:99;'+
    'background:var(--surface);color:var(--ink);border:1px solid var(--border);'+
    'border-radius:7px;padding:7px 10px;font:12px system-ui,sans-serif;'+
    'box-shadow:0 4px 14px rgba(0,0,0,.16);display:none;max-width:260px';
  document.body.appendChild(tip);
  document.addEventListener('mouseover',function(e){
    var t=e.target.closest('[data-tip]'); if(!t) return;
    tip.innerHTML=t.getAttribute('data-tip'); tip.style.display='block';
  });
  document.addEventListener('mousemove',function(e){
    if(tip.style.display==='none') return;
    var x=e.clientX+13,y=e.clientY+13;
    var r=tip.getBoundingClientRect();
    if(x+r.width>innerWidth) x=e.clientX-r.width-13;
    if(y+r.height>innerHeight) y=e.clientY-r.height-13;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('[data-tip]')) tip.style.display='none';
  });
})();
"""


# ── helpers ────────────────────────────────────
def esc(text):
    return html.escape(str(text), quote=True)


def fmt_duration(seconds):
    seconds = int(round(seconds or 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def pct(value, digits=0):
    """A fraction as a percentage, or an em-dash when there is nothing to show."""
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _swatch(state):
    return (f'<span class="sw" style="background:{STATE_COLOR[state]}"></span>'
            f'{esc(state)}')


def legend(states):
    return ('<div class="legend">'
            + "".join(f"<span>{_swatch(s)}</span>" for s in states)
            + "</div>")


def states_in_timeline(timeline):
    """
    Only the states a column actually shows.

    A legend must not advertise something the chart does not draw. This is not
    hypothetical: a state can hold real time across the session and still never
    land on a sample - in one run 58 seconds of Unknown existed while no
    timeline point contained any, so a fixed legend claimed a colour the reader
    would never find.
    """
    present = set()
    for point in timeline:
        for state, count in point.get("counts", {}).items():
            if count:
                present.add(state)
    return [s for s in STACK_ORDER if s in present]


# ── charts ─────────────────────────────────────
def chart_engagement(timeline, width=1000, height=190):
    """
    Engagement over the session: one series, so no legend - the title names it.

    Drawn as a line with a soft fill rather than bars because the reader's job
    here is trend, not comparison of individual samples.
    """
    points = [p for p in timeline if p.get("engagement") is not None]
    if len(points) < 2:
        return ('<p class="note">Not enough samples yet to plot a trend - '
                'the session needs to run a little longer.</p>')

    pad_l, pad_r, pad_t, pad_b = 40, 12, 12, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    t_max = max(p["t"] for p in points) or 1.0

    def sx(t):
        return pad_l + (t / t_max) * plot_w

    def sy(v):
        return pad_t + (1 - v) * plot_h

    coords = [(sx(p["t"]), sy(p["engagement"])) for p in points]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = (f"{coords[0][0]:.1f},{pad_t + plot_h:.1f} " + line
            + f" {coords[-1][0]:.1f},{pad_t + plot_h:.1f}")

    grid = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = sy(frac)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                    f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        grid.append(f'<text x="{pad_l - 7}" y="{y + 3.5:.1f}" '
                    f'text-anchor="end">{int(frac * 100)}%</text>')

    marks = []
    for (x, y), p in zip(coords, points):
        tip = (f"{fmt_duration(p['t'])} into the session<br>"
               f"<b>{pct(p['engagement'])}</b> attentive<br>"
               f"{p['present']} present")
        marks.append(
            f'<circle class="mk" cx="{x:.1f}" cy="{y:.1f}" r="9" '
            f'fill="transparent" data-tip="{esc(tip)}"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="var(--accent)" '
            f'stroke="var(--surface)" stroke-width="2"/>'
        )

    return f"""<svg viewBox="0 0 {width} {height}" role="img"
   aria-label="Attentive share of readable students over the session">
  {''.join(grid)}
  <polygon points="{area}" fill="var(--accent)" opacity=".10"/>
  <polyline points="{line}" fill="none" stroke="var(--accent)"
     stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
  {''.join(marks)}
  <text x="{pad_l}" y="{height - 7}" class="lbl">start</text>
  <text x="{width - pad_r}" y="{height - 7}" text-anchor="end"
     class="lbl">{esc(fmt_duration(t_max))}</text>
</svg>"""


def chart_states_over_time(timeline, width=1000, height=210):
    """
    Who was in which state, sampled through the session.

    Stacked columns rather than a stacked area: the samples are discrete, and
    an area would imply the pipeline knew what happened between them.
    """
    if not timeline:
        return '<p class="note">No samples recorded yet.</p>'

    pad_l, pad_r, pad_t, pad_b = 40, 12, 10, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    peak = max((p["present"] for p in timeline), default=0) or 1

    n = len(timeline)
    slot = plot_w / n
    bar_w = max(2.0, min(26.0, slot * 0.72))

    bars = []
    for i, point in enumerate(timeline):
        x = pad_l + slot * (i + 0.5) - bar_w / 2
        y = pad_t + plot_h
        rows = []
        for state in STACK_ORDER:
            count = point["counts"].get(state, 0)
            if not count:
                continue
            seg_h = (count / peak) * plot_h
            y -= seg_h
            # 2px surface gap between segments so adjacent fills never merge.
            draw_h = max(1.0, seg_h - 2)
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{draw_h:.1f}" rx="2" fill="{STATE_COLOR[state]}"/>'
            )
            rows.append(f"{esc(state)}: <b>{count}</b>")
        tip = (f"{fmt_duration(point['t'])} into the session<br>"
               + ("<br>".join(rows) if rows else "nobody present"))
        bars.append(
            f'<rect class="hit" x="{pad_l + slot * i:.1f}" y="{pad_t}" '
            f'width="{slot:.1f}" height="{plot_h:.1f}" '
            f'data-tip="{esc(tip)}"/>'
        )

    grid = []
    for frac in (0, 0.5, 1.0):
        y = pad_t + plot_h * (1 - frac)
        grid.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                    f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        grid.append(f'<text x="{pad_l - 7}" y="{y + 3.5:.1f}" '
                    f'text-anchor="end">{int(round(peak * frac))}</text>')

    return f"""<svg viewBox="0 0 {width} {height}" role="img"
   aria-label="Students in each state, sampled through the session">
  {''.join(grid)}
  {''.join(bars)}
  <text x="{pad_l}" y="{height - 7}" class="lbl">start</text>
  <text x="{width - pad_r}" y="{height - 7}" text-anchor="end" class="lbl"
     >{esc(fmt_duration(timeline[-1]['t']))}</text>
</svg>"""


def per_student_rows(students):
    """
    Per-student detail, lowest attentiveness first.

    Ordered so whoever needs attention is at the top rather than buried. Also
    serves as the table view the charts are obliged to have.
    """
    if not students:
        return '<p class="note">No students were recorded in this session.</p>'

    rows = []
    for s in students:
        score = s["attentiveness"]
        if score is None:
            bar = ('<span class="tag" style="color:var(--muted)">'
                   'not readable</span>')
            score_cell = "—"
        else:
            bar = (f'<span class="bar"><i style="width:{score * 100:.0f}%;'
                   f'background:var(--good)"></i></span>')
            score_cell = pct(score)

        cov = s["coverage"]
        cov_note = ""
        if cov < 0.5:
            # A score over a small slice of the lesson is a different claim
            # from the same score over all of it, and the number cannot say so.
            cov_note = (' <span style="color:var(--warning)">▲</span>')

        sleep = ("—" if not s["slept"] else
                 f"{s['sleep_episodes']}×, {fmt_duration(s['sleep_s'])}")

        rows.append(f"""<tr>
  <td>#{esc(s['student'])}</td>
  <td class="num">{score_cell}</td>
  <td style="width:130px">{bar}</td>
  <td class="num">{pct(cov)}{cov_note}</td>
  <td class="num">{esc(fmt_duration(s['present_s']))}</td>
  <td class="num">{esc(sleep)}</td>
</tr>""")

    return f"""<div class="scroll"><table>
<thead><tr>
  <th>Student</th><th class="num">Attentiveness</th><th></th>
  <th class="num">Coverage</th><th class="num">Present</th>
  <th class="num">Sleep</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
<p class="note">Attentiveness is the share of <em>monitored</em> time spent
Attentive. Coverage is how much of a student's time in the room was readable at
all &mdash; a high score over low coverage is a confident claim about a small
slice of the lesson, and <span style="color:var(--warning)">&#9650;</span> marks
those below 50%.</p>"""


def kpi(value, label, note=""):
    note_html = f'<div class="n">{note}</div>' if note else ""
    return (f'<div class="kpi"><div class="v">{value}</div>'
            f'<div class="k">{esc(label)}</div>{note_html}</div>')


# ── page ───────────────────────────────────────
def render(report, live=False, title="ClassSense session report"):
    """Build the whole page from a SessionRecorder.report() dict."""
    head = report["headline"]
    cov = report["coverage"]
    sess = report["session"]
    states = report["state_seconds"]

    mean = head["mean_attentiveness"]
    hero = pct(mean) if mean is not None else "—"

    unreadable_share = 0.0
    total = cov["monitored_s"] + cov["unreadable_s"]
    if total > 0:
        unreadable_share = cov["unreadable_s"] / total

    kpis = "".join([
        kpi(head["students_seen"], "Students seen",
            f"peak {head['peak_present']} at once"),
        kpi(f"{head['pct_never_slept']:.0f}%"
            if head["pct_never_slept"] is not None else "—",
            "Never fell asleep",
            f"{head['students_slept']} did"),
        kpi(f"{head['pct_time_not_sleeping']:.1f}%"
            if head["pct_time_not_sleeping"] is not None else "—",
            "Of monitored time awake",
            f"{fmt_duration(head['total_sleep_s'])} asleep in total"),
        kpi(head["sleep_episodes"], "Sleep episodes",
            f"longest {fmt_duration(head['longest_sleep_s'])}"),
        kpi(pct(cov["mean_coverage"]), "Mean coverage",
            "share of time readable"),
        kpi(fmt_duration(sess["duration_s"]), "Session length",
            f"{sess['samples']} samples"),
    ])

    # The weighted mean is shown beside the hero rather than instead of it:
    # a large gap between them means one or two students dominated the
    # monitored time, which changes how the headline should be read.
    weighted = head["weighted_attentiveness"]
    weighted_note = ""
    if weighted is not None and mean is not None:
        gap = abs(weighted - mean)
        weighted_note = (
            f"Time-weighted across all monitored seconds: "
            f"<b>{pct(weighted)}</b>."
        )
        if gap > 0.10:
            weighted_note += (
                " The two differ by more than 10 points, which means a few "
                "students account for most of the monitored time — read "
                "the per-student table rather than the headline."
            )

    coverage_warning = ""
    if unreadable_share > 0.25:
        coverage_warning = f"""<div class="card warn">
<h2>Read these numbers with the coverage in mind</h2>
<p class="note">{pct(unreadable_share)} of recorded student-time was
<b>Unknown</b> or <b>Unmonitored</b> &mdash; too far from the camera to read, or
beyond what this machine can watch at a trustworthy rate. Those seconds are
excluded from every score rather than counted as inattention, so the scores are
honest about what was seen; they simply describe less of the lesson than the
session length suggests.</p></div>"""

    # How far apart the timeline samples are, so the chart says what it is.
    timeline = report["timeline"]
    if len(timeline) >= 2:
        gap = timeline[1]["t"] - timeline[0]["t"]
        sample_note = f"{fmt_duration(gap)}"
    else:
        sample_note = "few seconds"

    live_bits = ""
    if live:
        live_bits = """<meta http-equiv="refresh" content="20">"""

    shown = states_in_timeline(report["timeline"])
    state_legend = legend(shown) if shown else ""

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
{live_bits}
<style>{CSS}</style>
</head><body>
<div class="wrap">

<header>
  <h1>{esc(title)}</h1>
  <div class="sub">{esc(sess['started'])} &rarr; {esc(sess['ended'])}
    &nbsp;&middot;&nbsp; {esc(fmt_duration(sess['duration_s']))}
    {'&nbsp;&middot;&nbsp; live, refreshing' if live else ''}</div>
</header>

<div class="card">
  <div class="hero">
    <div class="fig">{hero}</div>
    <div class="of"><b>Average attentiveness</b> &mdash; the mean across
      students of each student's share of monitored time spent Attentive.
      {weighted_note}</div>
  </div>
</div>

<div class="kpis">{kpis}</div>

<div class="card" style="margin-top:16px">
  <h2>Attentive share over the session</h2>
  <div class="sub" style="margin-bottom:8px">Of students readable at each
    sample. Hover any point.</div>
  {chart_engagement(report['timeline'])}
</div>

<div class="card">
  <h2>Students in each state</h2>
  <div class="sub" style="margin-bottom:6px">Sampled every
    {sample_note}; column height is how many people were present. Brief states
    can fall between samples &mdash; the totals below are the complete
    figures.</div>
  {state_legend}
  {chart_states_over_time(report['timeline'])}
</div>

{coverage_warning}

<div class="card">
  <h2>Per student</h2>
  <div class="sub" style="margin-bottom:10px">Lowest attentiveness first.</div>
  {per_student_rows(report['students'])}
</div>

<div class="card">
  <h2>Where the time went</h2>
  <div class="scroll"><table>
  <thead><tr><th>State</th><th class="num">Total time</th>
    <th class="num">Share of all student-time</th></tr></thead>
  <tbody>{''.join(
      f'<tr><td><span class="tag">{_swatch(s)}</span></td>'
      f'<td class="num">{fmt_duration(v)}</td>'
      f'<td class="num">{pct(v / sum(states.values()) if sum(states.values()) else None)}</td></tr>'
      for s, v in states.items() if v > 0
  )}</tbody></table></div>
</div>

<footer>
  Generated by ClassSense. Attentiveness and the awake percentage are computed
  over monitored time only; Unknown and Unmonitored seconds are excluded rather
  than counted against a student. Machine-readable copy at
  <code>/report.json</code>.
</footer>

</div>
<script>{TOOLTIP_JS}</script>
</body></html>"""


def render_json(report):
    return json.dumps(report, indent=2)
