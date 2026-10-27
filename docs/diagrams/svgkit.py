"""Write light/dark SVG diagrams in the style of 01's architecture diagram."""
THEMES = {
    "light": dict(bg="#ffffff", arrow="#52514e", violet="#6250d6", text="#0b0b0b", box="#ffffff", stroke="#c3c2b7",
                  learn="#2a78d6", frozen="#6b6a65", s="#52514e", c="#6b6a65", red="#e34948", green="#1baf7a", amber="#b07a00"),
    "dark": dict(bg="#0d1117", arrow="#c3c2b7", violet="#9085e9", text="#f0efec", box="#161b22", stroke="#3d444d",
                 learn="#3987e5", frozen="#9a9890", s="#c3c2b7", c="#9a9890", red="#ff6b6a", green="#3fd29b", amber="#f0b429"),
}

def svg(w, h, body, t):
    k = THEMES[t]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <rect x="0" y="0" width="{w}" height="{h}" rx="12" fill="{k['bg']}"/>
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{k['arrow']}"/></marker>
    <marker id="ahv" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{k['violet']}"/></marker>
    <marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{k['red']}"/></marker>
  </defs>
  <style>
    text {{ fill: {k['text']}; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .box {{ fill: {k['box']}; stroke: {k['stroke']}; stroke-width: 1.5; }}
    .learn {{ stroke: {k['learn']}; }} .frozen {{ stroke: {k['frozen']}; stroke-dasharray: 4 3; }} .loss {{ stroke: #e0622f; }}
    .bad {{ stroke: {k['red']}; }} .good {{ stroke: {k['green']}; }}
    .m {{ font-size: 13px; font-weight: 600; }}
    .h {{ font-size: 14px; font-weight: 700; }}
    .s {{ font-size: 11.5px; fill: {k['s']}; font-family: ui-monospace, Menlo, Consolas, monospace; }}
    .c {{ font-size: 11px; fill: {k['c']}; font-family: ui-monospace, Menlo, Consolas, monospace; }}
    .v {{ fill: {k['violet']}; }} .r {{ fill: {k['red']}; }} .g {{ fill: {k['green']}; }} .y {{ fill: {k['amber']}; }}
    .a {{ stroke: {k['arrow']}; stroke-width: 1.5; fill: none; marker-end: url(#ah); }}
    .ar {{ stroke: {k['red']}; stroke-width: 1.5; fill: none; marker-end: url(#ahr); }}
    .ema {{ stroke: {k['violet']}; stroke-width: 1.5; fill: none; stroke-dasharray: 5 4; marker-end: url(#ahv); }}
  </style>
  <g transform="translate(6,8)">
{body}
  </g>
</svg>
'''

def box(x, y, w, h, title, sub="", note="", cls=""):
    cx = x + w / 2
    out = [f'  <rect class="box {cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>',
           f'  <text class="m" x="{cx}" y="{y + 25}" text-anchor="middle">{title}</text>']
    if sub: out.append(f'  <text class="s" x="{cx}" y="{y + 42}" text-anchor="middle">{sub}</text>')
    if note: out.append(f'  <text class="c" x="{cx}" y="{y + 58}" text-anchor="middle">{note}</text>')
    return "\n".join(out)

def text(x, y, s, cls="c", anchor="start"):
    return f'  <text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{s}</text>'

def path(d, cls="a"):
    return f'  <path class="{cls}" d="{d}"/>'

def write(dirpath, name, w, h, body):
    from pathlib import Path
    for t in THEMES:
        Path(dirpath, f"{name}-{t}.svg").write_text(svg(w, h, body, t))
