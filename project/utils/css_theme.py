"""Institutional dark theme — Bloomberg aesthetic"""

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:      #0A0D14; --bg2: #12151F; --bg3: #161926;
    --border:  #1E2440; --border2: #252A45;
    --esg:     #00D4AA; --cls: #FF4560; --mkt: #4B9EFF;
    --gold:    #FFB800; --purple: #A855F7; --orange: #F97316;
    --txt:     #E8EAF0; --txt2: #7B8098; --txt3: #4A4E65;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--txt)!important;font-family:'Inter',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--txt)!important;}
.main .block-container{padding:1.2rem 2rem!important;max-width:1440px!important;}
h1,h2,h3,h4{font-family:'Inter',sans-serif!important;color:var(--txt)!important;font-weight:600!important;letter-spacing:-0.02em!important;}
[data-testid="stTabs"] [role="tablist"]{background:var(--bg2)!important;border-bottom:1px solid var(--border)!important;padding:0 0.5rem!important;}
[data-testid="stTabs"] [role="tab"]{background:transparent!important;color:var(--txt2)!important;border:none!important;border-bottom:2px solid transparent!important;padding:0.6rem 0.9rem!important;font-size:0.68rem!important;font-weight:600!important;letter-spacing:0.06em!important;text-transform:uppercase!important;border-radius:0!important;transition:color .15s;}
[data-testid="stTabs"] [role="tab"]:hover{color:var(--txt)!important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--esg)!important;border-bottom:2px solid var(--esg)!important;}
[data-testid="stMetric"]{background:var(--bg3)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:0.9rem 1.1rem!important;}
[data-testid="stMetricLabel"]{font-size:0.62rem!important;font-weight:600!important;letter-spacing:0.09em!important;text-transform:uppercase!important;color:var(--txt2)!important;}
[data-testid="stMetricValue"]{font-size:1.3rem!important;font-weight:700!important;color:var(--txt)!important;font-family:'JetBrains Mono',monospace!important;}
[data-testid="stMetricDelta"]{font-size:0.72rem!important;font-family:'JetBrains Mono',monospace!important;}
[data-testid="stButton"]>button{background:var(--esg)!important;color:#0A0D14!important;border:none!important;border-radius:6px!important;font-weight:700!important;font-size:0.8rem!important;padding:0.5rem 1.4rem!important;transition:all .2s;}
[data-testid="stButton"]>button:hover{background:#00B894!important;box-shadow:0 4px 18px rgba(0,212,170,.28)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px!important;overflow:hidden!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}
.sec{font-size:0.62rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--esg);padding:.35rem 0;border-bottom:1px solid var(--border);margin:1.1rem 0 .7rem 0;}
.memo{background:var(--bg3);border:1px solid var(--border);border-left:3px solid var(--esg);border-radius:8px;padding:1.4rem;font-size:.87rem;line-height:1.72;}
.pill-crisis{display:inline-block;background:rgba(255,69,96,.1);border:1px solid rgba(255,69,96,.3);color:#FF4560;border-radius:4px;padding:.1rem .42rem;font-size:.6rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;}
.badge-esg{display:inline-block;background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.3);color:#00D4AA;border-radius:4px;padding:.1rem .42rem;font-size:.6rem;font-weight:700;letter-spacing:.07em;}
.mono{font-family:'JetBrains Mono',monospace;}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#12151F", plot_bgcolor="#12151F",
    font=dict(family="Inter,sans-serif", color="#7B8098", size=11),
    xaxis=dict(gridcolor="#1E2440", linecolor="#1E2440",
               tickfont=dict(color="#7B8098", size=10), zerolinecolor="#1E2440"),
    yaxis=dict(gridcolor="#1E2440", linecolor="#1E2440",
               tickfont=dict(color="#7B8098", size=10),
               zerolinecolor="#252A45", zerolinewidth=1),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E2440",
                borderwidth=1, font=dict(color="#7B8098", size=10)),
    hoverlabel=dict(bgcolor="#1C2035", bordercolor="#252A45",
                    font=dict(family="JetBrains Mono,monospace", size=11, color="#E8EAF0")),
    margin=dict(l=55, r=30, t=45, b=40),
    colorway=["#00D4AA","#FF4560","#4B9EFF","#FFB800","#A855F7","#F97316"],
    title_font=dict(color="#E8EAF0", size=13, family="Inter,sans-serif"),
    title_x=0.02,
)

C_ESG="#00D4AA"; C_CLS="#FF4560"; C_MKT="#4B9EFF"
C_GOLD="#FFB800"; C_PUR="#A855F7"; C_ORG="#F97316"
C_BG="#12151F"; C_BDR="#1E2440"; C_TXT="#E8EAF0"; C_MUT="#7B8098"
