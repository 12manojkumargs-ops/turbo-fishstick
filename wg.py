import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Accessify_mk2",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# THEME / CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

/* HERO */

.hero {
    padding: 25px 28px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    margin-bottom: 20px;
}

.hero h1 {
    margin: 0;
    font-size: 2.35rem;
}

.hero p {
    margin: 7px 0 0;
    color: #cbd5e1;
}


/* DATASET CARDS */

.card {
    padding: 18px;
    border-radius: 16px;
    height: 100%;
    border: 1px solid rgba(128,128,128,0.35);
    background: var(--card-bg);
    color: var(--card-text);
}

.metric {
    font-size: 2rem;
    font-weight: 750;
    color: var(--card-text);
}

.muted {
    color: var(--card-muted);
    font-size: 0.9rem;
}


/* FINDINGS */

.finding {
    padding: 15px;
    border-radius: 13px;
    margin: 9px 0;
    border: 1px solid rgba(128,128,128,0.35);
    background: var(--finding-bg);
    color: var(--finding-text);
}

.finding p {
    margin-top: 10px;
}

.small {
    font-size: 0.85rem;
    color: var(--small-text);
}


/* PILLS */

.pill {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 999px;
    background: var(--pill-bg);
    color: var(--pill-text);
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 5px;
}


/* LIGHT THEME */

@media (prefers-color-scheme: light) {

    .card {
        --card-bg: #ffffff;
        --card-text: #111827;
        --card-muted: #64748b;
    }

    .finding {
        --finding-bg: #fafafa;
        --finding-text: #111827;
    }

    .small {
        --small-text: #475569;
    }

    .pill {
        --pill-bg: #e2e8f0;
        --pill-text: #111827;
    }
}


/* DARK THEME */

@media (prefers-color-scheme: dark) {

    .card {
        --card-bg: #1e293b;
        --card-text: #f8fafc;
        --card-muted: #cbd5e1;
    }

    .finding {
        --finding-bg: #1e293b;
        --finding-text: #f8fafc;
    }

    .small {
        --small-text: #cbd5e1;
    }

    .pill {
        --pill-bg: #334155;
        --pill-text: #f1f5f9;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Accessify mk2 AI</h1>
        <p>
            Explainable, dataset-backed accessibility risk analysis
            for real websites
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data
def load_data(uploaded_file=None):

    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    return pd.read_csv("Original_full_dataN.csv")


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(df):

    data = df.copy()

    data["violation_score"] = pd.to_numeric(
        data["violation_score"],
        errors="coerce"
    ).fillna(0)

    data["violation_count"] = pd.to_numeric(
        data["violation_count"],
        errors="coerce"
    ).fillna(0)

    data["text"] = (
        data["violation_name"].fillna("").astype(str)
        + " "
        + data["violation_category"].fillna("").astype(str)
        + " "
        + data["affected_html_elements"].fillna("").astype(str)
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=1800,
                ngram_range=(1, 2),
                min_df=2
            )
        ),

        (
            "rf",
            RandomForestClassifier(
                n_estimators=180,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    model.fit(
        data["text"],
        data["violation_impact"].fillna("moderate")
    )

    return model


# ============================================================
# IMPACT WEIGHTS
# ============================================================

IMPACT_SCORE = {
    "minor": 8,
    "moderate": 20,
    "serious": 35,
    "critical": 50
}

IMPACT_ORDER = {
    "minor": 0,
    "moderate": 1,
    "serious": 2,
    "critical": 3
}


# ============================================================
# WEBSITE SCANNING RULES
# ============================================================

def heading_order_issues(soup):

    levels = []

    for heading in soup.find_all(re.compile("^h[1-6]$")):
        levels.append(int(heading.name[1]))

    bad = 0

    for i in range(1, len(levels)):

        if levels[i] > levels[i - 1] + 1:
            bad += 1

    return bad


def duplicate_id_issues(soup):

    ids = [
        x.get("id")
        for x in soup.find_all(attrs={"id": True})
    ]

    return max(
        0,
        len(ids) - len(set(ids))
    )


def landmark_duplicate_issues(soup):

    total = 0

    for tag in [
        "main",
        "nav",
        "header",
        "footer",
        "aside"
    ]:

        n = len(soup.find_all(tag))

        if n > 1:
            total += n - 1

    return total


def form_label_issues(soup):

    labels = 0

    for inp in soup.find_all(
        ["input", "select", "textarea"]
    ):

        if inp.get("type") in [
            "hidden",
            "submit",
            "button",
            "reset"
        ]:
            continue

        iid = inp.get("id")

        labelled = bool(
            iid and
            soup.find(
                "label",
                attrs={"for": iid}
            )
        )

        labelled = labelled or bool(
            inp.get("aria-label")
            or inp.get("aria-labelledby")
        )

        if not labelled:
            labels += 1

    return labels


RULES = {

    "image-alt":
        lambda s:
        len(
            s.find_all(
                "img",
                alt=lambda x: x is None
            )
        ),

    "button-name":
        lambda s:
        sum(
            1
            for x in s.find_all("button")
            if
            not x.get_text(" ", strip=True)
            and not x.get("aria-label")
            and not x.get("title")
        ),

    "link-name":
        lambda s:
        sum(
            1
            for x in s.find_all("a")
            if
            not x.get_text(" ", strip=True)
            and not x.get("aria-label")
            and not x.get("title")
        ),

    "page-has-heading-one":
        lambda s:
        0
        if len(s.find_all("h1")) == 1
        else 1,

    "heading-order":
        lambda s:
        heading_order_issues(s),

    "landmark-one-main":
        lambda s:
        0
        if len(s.find_all("main")) == 1
        else 1,

    "landmark-unique":
        lambda s:
        landmark_duplicate_issues(s),

    "region":
        lambda s:
        0
        if s.find("main")
        else 1,

    "empty-heading":
        lambda s:
        sum(
            1
            for h in s.find_all(
                re.compile("^h[1-6]$")
            )
            if not h.get_text(
                " ",
                strip=True
            )
        ),

    "empty-table-header":
        lambda s:
        sum(
            1
            for h in s.find_all("th")
            if not h.get_text(
                " ",
                strip=True
            )
        ),

    "form-field-multiple-labels":
        lambda s:
        form_label_issues(s),

    "duplicate-id":
        lambda s:
        duplicate_id_issues(s),

    "html-has-lang":
        lambda s:
        0
        if s.html and s.html.get("lang")
        else 1
}


# ============================================================
# FETCH WEBSITE
# ============================================================

def get_html(url):

    headers = {
        "User-Agent":
        "Mozilla/5.0 "
        "( AI accessibility research)"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15,
        allow_redirects=True
    )

    response.raise_for_status()

    return (
        response.text,
        response.url
    )


# ============================================================
# SCAN HTML
# ============================================================

def scan_html(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    findings = []

    for name, function in RULES.items():

        try:
            count = int(function(soup))

        except Exception:
            count = 0

        if count > 0:

            findings.append(
                (name, count)
            )


    features = {

        "html_size":
            len(html),

        "dom_elements":
            len(soup.find_all(True)),

        "images":
            len(soup.find_all("img")),

        "links":
            len(soup.find_all("a")),

        "buttons":
            len(soup.find_all("button")),

        "forms":
            len(soup.find_all("form")),

        "inputs":
            len(
                soup.find_all(
                    ["input", "select", "textarea"]
                )
            ),

        "headings":
            len(
                soup.find_all(
                    re.compile("^h[1-6]$")
                )
            ),

        "h1_count":
            len(soup.find_all("h1")),

        "main_count":
            len(soup.find_all("main")),

        "nav_count":
            len(soup.find_all("nav")),

        "aria_count":
            len(
                soup.find_all(
                    attrs={"aria-label": True}
                )
            )
            +
            len(
                soup.find_all(
                    attrs={
                        "aria-labelledby": True
                    }
                )
            ),

        "lang_present":
            int(
                bool(
                    soup.html
                    and soup.html.get("lang")
                )
            )
    }

    return (
        findings,
        features,
        soup
    )


# ============================================================
# DATASET LOOKUP
# ============================================================

def dataset_lookup(
    df,
    violation
):

    x = df[
        df["violation_name"]
        .astype(str)
        .str.lower()
        ==
        violation.lower()
    ]

    if len(x) == 0:

        x = df[
            df["violation_name"]
            .astype(str)
            .str.contains(
                re.escape(violation),
                case=False,
                na=False
            )
        ]

    return x


# ============================================================
# ENRICH LIVE FINDING WITH DATASET + ML
# ============================================================

def nearest_dataset_finding(
    df,
    violation,
    count,
    model
):

    x = dataset_lookup(
        df,
        violation
    )

    if len(x) == 0:
        return None

    row = x.iloc[0]

    text = (
        str(
            row.get(
                "violation_name",
                ""
            )
        )
        + " "
        +
        str(
            row.get(
                "violation_category",
                ""
            )
        )
        + " "
        +
        str(
            row.get(
                "affected_html_elements",
                ""
            )
        )
    )

    try:

        predicted = model.predict(
            [text]
        )[0]

        probabilities = (
            model.predict_proba(
                [text]
            )[0]
        )

        confidence = float(
            np.max(probabilities)
        )

    except Exception:

        predicted = str(
            row.get(
                "violation_impact",
                "moderate"
            )
        )

        confidence = 0.0


    score_value = pd.to_numeric(
        row.get(
            "violation_score",
            3
        ),
        errors="coerce"
    )

    if pd.isna(score_value):
        score_value = 3


    return {

        "name":
            str(
                row.get(
                    "violation_name",
                    violation
                )
            ),

        "count":
            count,

        "impact":
            str(
                row.get(
                    "violation_impact",
                    predicted
                )
            ).lower(),

        "predicted_impact":
            str(predicted).lower(),

        "confidence":
            confidence,

        "score":
            float(score_value),

        "category":
            str(
                row.get(
                    "violation_category",
                    "Accessibility"
                )
            ),

        "description":
            str(
                row.get(
                    "violation_description",
                    "No description available."
                )
            ),

        "wcag":
            str(
                row.get(
                    "wcag_reference",
                    "Not specified"
                )
            ),

        "elements":
            str(
                row.get(
                    "affected_html_elements",
                    "Not specified"
                )
            )
    }


# ============================================================
# NEW IMPROVED RISK SCORE
# ============================================================

def calculate_risk(
    findings,
    df,
    features
):

    if not findings:
        return 3.0


    # --------------------------------------------------------
    # 1. SEVERITY
    # --------------------------------------------------------

    severity_total = 0

    for finding in findings:

        impact = str(
            finding["impact"]
        ).lower()

        count = int(
            finding["count"]
        )

        # Prevent one huge repeated problem
        # from completely dominating the score
        effective_count = min(
            count,
            10
        )

        severity_total += (
            IMPACT_SCORE.get(
                impact,
                20
            )
            *
            effective_count
        )


    # --------------------------------------------------------
    # 2. VIOLATION DIVERSITY
    # --------------------------------------------------------

    # More different violation types
    # = broader accessibility risk

    diversity_score = min(
        len(findings) * 4,
        20
    )


    # --------------------------------------------------------
    # 3. DATASET SCORE
    # --------------------------------------------------------

    dataset_total = 0

    for finding in findings:

        dataset_score = float(
            finding.get(
                "score",
                0
            )
        )

        if dataset_score > 0:

            dataset_total += min(
                dataset_score,
                10
            )

    dataset_score = min(
        dataset_total,
        20
    )


    # --------------------------------------------------------
    # 4. VIOLATION DENSITY
    # --------------------------------------------------------

    dom_elements = max(
        int(
            features.get(
                "dom_elements",
                1
            )
        ),
        1
    )

    total_violations = sum(
        min(
            int(
                finding["count"]
            ),
            10
        )
        for finding in findings
    )

    density = (
        total_violations
        /
        dom_elements
    )

    density_score = min(
        density * 100,
        15
    )


    # --------------------------------------------------------
    # 5. PAGE COMPLEXITY FACTOR
    # --------------------------------------------------------

    elements = max(
        dom_elements,
        1
    )

    images = int(
        features.get(
            "images",
            0
        )
    )

    forms = int(
        features.get(
            "forms",
            0
        )
    )

    inputs = int(
        features.get(
            "inputs",
            0
        )
    )

    # A complex page with accessibility
    # problems deserves slightly more attention
    complexity_factor = min(
        (
            elements / 1000
            +
            images / 100
            +
            forms / 20
            +
            inputs / 30
        ),
        10
    )


    # --------------------------------------------------------
    # 6. FINAL SCORE
    # --------------------------------------------------------

    raw_score = (

        severity_total * 0.50

        +

        diversity_score

        +

        dataset_score * 0.50

        +

        density_score

        +

        complexity_factor

    )


    # Clamp to 0–100

    final_score = min(
        100,
        max(
            0,
            raw_score
        )
    )

    return round(
        final_score,
        1
    )


# ============================================================
# SCORE LABEL
# ============================================================

def risk_label(score):

    if score < 20:
        return "LOW RISK"

    if score < 45:
        return "MODERATE RISK"

    if score < 70:
        return "HIGH RISK"

    return "CRITICAL RISK"


# ============================================================
# LOAD DATASET
# ============================================================

with st.sidebar:

    st.header("📁 Dataset")

    uploaded = st.file_uploader(
        "Use a different CSV",
        type=["csv"]
    )

    st.caption(
        "Default: Original_full_dataN.csv"
    )


try:

    df = load_data(
        uploaded
    )

except Exception:

    st.error(
        "CSV not found. Put "
        "Original_full_dataN.csv "
        "beside this app, or upload "
        "it from the sidebar."
    )

    st.stop()


# ============================================================
# CHECK DATASET
# ============================================================

required_columns = [

    "violation_name",
    "violation_score",
    "violation_description",
    "violation_category",
    "violation_impact",
    "wcag_reference",
    "affected_html_elements"

]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    st.error(
        "Dataset is missing: "
        +
        ", ".join(missing)
    )

    st.stop()


# ============================================================
# TRAIN MODEL
# ============================================================

model = train_model(
    df
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric">
                {len(df):,}
            </div>
            <div class="muted">
                Dataset records
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="card">
            <div class="metric">
                {df["violation_name"].nunique()}
            </div>
            <div class="muted">
                Violation types
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    domain_count = (
        df["domain_category"].nunique()
        if "domain_category" in df.columns
        else 0
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="metric">
                {domain_count}
            </div>
            <div class="muted">
                Domain categories
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        """
        <div class="card">
            <div class="metric">
                ML
            </div>
            <div class="muted">
                Explainable prediction
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WEBSITE INPUT
# ============================================================

st.subheader(
    "🌐 Analyze a Website"
)

url = st.text_input(
    "Website URL",
    placeholder="https://example.com"
)


colA, colB = st.columns(
    [1, 4]
)

with colA:

    analyze = st.button(
        "🔎 Analyze Website",
        type="primary",
        use_container_width=True
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze:

    if not url.strip():

        st.warning(
            "Enter a website URL first."
        )

        st.stop()


    if not re.match(
        r"^https?://",
        url.strip(),
        re.I
    ):

        url = (
            "https://"
            +
            url.strip()
        )


    try:

        parsed = urlparse(
            url
        )

        if not parsed.netloc:

            raise ValueError(
                "Invalid URL"
            )


        with st.spinner(
            "Fetching and analyzing the website..."
        ):

            html, final_url = get_html(
                url
            )

            raw_findings, features, soup = scan_html(
                html
            )


            enriched = []

            for violation, count in raw_findings:

                finding = nearest_dataset_finding(
                    df,
                    violation,
                    count,
                    model
                )

                if finding:

                    enriched.append(
                        finding
                    )


            risk = calculate_risk(
                enriched,
                df,
                features
            )


        st.session_state[
            "_result"
        ] = (
            final_url,
            features,
            enriched,
            risk
        )


    except Exception as e:

        st.error(
            f"Could not analyze this website: {e}"
        )

        st.info(
            "Some websites block automated "
            "requests. Try another public "
            "HTTPS website."
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "_result" in st.session_state:

    (
        final_url,
        features,
        findings,
        risk
    ) = st.session_state[
        "_result"
    ]


    st.divider()

    st.subheader(
        "🎯 Website Accessibility Risk Score"
    )


    left, right = st.columns(
        [1.1, 1.4]
    )


    # ========================================================
    # SPEEDOMETER
    # ========================================================

    with left:

        fig = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=risk,

                number={
                    "suffix": " / 100",
                    "font": {
                        "size": 34
                    }
                },

                title={
                    "text": risk_label(risk),
                    "font": {
                        "size": 18
                    }
                },

                gauge={

                    "axis": {
                        "range": [
                            0,
                            100
                        ],
                        "tickwidth": 1,
                        "dtick": 20
                    },

                    "bar": {
                        "color": "#94a3b8",
                        "thickness": 0.25
                    },

                    "bgcolor":
                        "rgba(0,0,0,0)",

                    "borderwidth": 1,

                    "steps": [

                        {
                            "range": [
                                0,
                                20
                            ],
                            "color":
                                "#22c55e"
                        },

                        {
                            "range": [
                                20,
                                45
                            ],
                            "color":
                                "#eab308"
                        },

                        {
                            "range": [
                                45,
                                70
                            ],
                            "color":
                                "#f97316"
                        },

                        {
                            "range": [
                                70,
                                100
                            ],
                            "color":
                                "#ef4444"
                        }

                    ],

                    "threshold": {

                        "line": {
                            "color":
                                "#94a3b8",
                            "width": 5
                        },

                        "thickness":
                            0.8,

                        "value":
                            risk
                    }
                }
            )
        )


        fig.update_layout(

            height=330,

            margin=dict(
                l=20,
                r=20,
                t=60,
                b=10
            ),

            transition_duration=1200
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    # ========================================================
    # SCORE EXPLANATION
    # ========================================================

    with right:

        st.markdown(
            "### 📊 What this score means"
        )

        st.write(
            "This is an **Accessibility Risk Score**. "
            "It is not a percentage of WCAG compliance. "
            "Higher values mean greater accessibility risk."
        )


        st.markdown(
            f"""
            **Score:** `{risk}/100`

            **Classification:** `{risk_label(risk)}`
            """
        )


        st.markdown(
            "The score considers:"
        )

        st.markdown(
            """
            - 🔴 Violation severity
            - 🔢 Number of detected violations
            - 🧩 Different violation types
            - 📚 Dataset violation scores
            - 📐 Violation density
            - 🏗️ Website structural complexity
            """
        )


        if findings:

            impact_counts = pd.Series(
                [
                    x["impact"]
                    for x in findings
                ]
            ).value_counts()


            st.markdown(
                "### Detected severity"
            )


            for level in [
                "critical",
                "serious",
                "moderate",
                "minor"
            ]:

                if level in impact_counts:

                    st.write(
                        f"**{level.title()}** — "
                        f"{int(impact_counts[level])}"
                    )

        else:

            st.success(
                "No rule-matched accessibility "
                "issues were detected."
            )


    # ========================================================
    # FINDINGS
    # ========================================================

    st.subheader(
        "🔍 Explainable Findings"
    )


    if not findings:

        st.info(
            "No dataset-matched findings "
            "were detected."
        )

    else:

        sorted_findings = sorted(
            findings,
            key=lambda x:
                IMPACT_ORDER.get(
                    x["impact"],
                    0
                ),
            reverse=True
        )


        for f in sorted_findings:

            confidence_text = ""

            if f["confidence"]:

                confidence_text = (
                    f" "
                    f"(confidence "
                    f"{f['confidence'] * 100:.0f}%)"
                )


            st.markdown(
                f"""
                <div class="finding">

                    <b>{f['name']}</b>

                    <br><br>

                    <span class="pill">
                        {f['impact'].upper()}
                    </span>

                    <span class="pill">
                        {f['category']}
                    </span>

                    <span class="pill">
                        Detected: {f['count']}
                    </span>

                    <span class="pill">
                        Dataset score:
                        {f['score']:.0f}
                    </span>

                    <p>
                        {f['description']}
                    </p>

                    <div class="small">
                        <b>WCAG:</b>
                        {f['wcag']}
                    </div>

                    <div class="small">
                        <b>Affected elements:</b>
                        {f['elements'][:600]}
                    </div>

                    <div class="small">
                        <b>ML predicted impact:</b>
                        {f['predicted_impact']}
                        {confidence_text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # WEBSITE STRUCTURE
    # ========================================================

    st.subheader(
        "🏗️ Website Structure"
    )


    a, b, c, d = st.columns(4)


    with a:
        st.metric(
            "DOM elements",
            features["dom_elements"]
        )


    with b:
        st.metric(
            "Images",
            features["images"]
        )


    with c:
        st.metric(
            "Links",
            features["links"]
        )


    with d:
        st.metric(
            "Headings",
            features["headings"]
        )


    a, b, c, d = st.columns(4)


    with a:
        st.metric(
            "Buttons",
            features["buttons"]
        )


    with b:
        st.metric(
            "Forms",
            features["forms"]
        )


    with c:
        st.metric(
            "Inputs",
            features["inputs"]
        )


    with d:
        st.metric(
            "ARIA labels",
            features["aria_count"]
        )


    # ========================================================
    # VIOLATION GRAPH
    # ========================================================

    if findings:

        st.subheader(
            "📈 Detected Violations"
        )


        chart_df = pd.DataFrame({

            "Violation":
                [
                    x["name"]
                    for x in findings
                ],

            "Count":
                [
                    x["count"]
                    for x in findings
                ]

        })


        chart_df = chart_df.sort_values(
            "Count",
            ascending=True
        )


        fig2 = go.Figure(
            go.Bar(

                x=chart_df["Count"],

                y=chart_df["Violation"],

                orientation="h"
            )
        )


        fig2.update_layout(

            height=max(
                300,
                len(chart_df) * 45
            ),

            margin=dict(
                l=10,
                r=20,
                t=20,
                b=20
            ),

            xaxis_title=
                "Detected count",

            yaxis_title=""
        )


        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={
                "displayModeBar":
                    False
            }
        )


    # ========================================================
    # REPORT
    # ========================================================

    st.subheader(
        "📄 Analysis Report"
    )


    report = [

        "Accessify_mk2",
        "WEBSITE ACCESSIBILITY RISK REPORT",

        "",

        f"URL: {final_url}",

        f"Risk Score: "
        f"{risk}/100",

        f"Classification: "
        f"{risk_label(risk)}",

        "",

        "FINDINGS"

    ]


    for f in findings:

        report.append(

            f"- {f['name']} | "
            f"count={f['count']} | "
            f"impact={f['impact']} | "
            f"category={f['category']} | "
            f"WCAG={f['wcag']}"

        )

        report.append(
            f"  {f['description']}"
        )


    report.extend([

        "",

        "WEBSITE STRUCTURE",

        f"DOM elements: "
        f"{features['dom_elements']}",

        f"Images: "
        f"{features['images']}",

        f"Links: "
        f"{features['links']}",

        f"Buttons: "
        f"{features['buttons']}",

        f"Forms: "
        f"{features['forms']}",

        f"Inputs: "
        f"{features['inputs']}",

        f"Headings: "
        f"{features['headings']}",

        f"ARIA labels: "
        f"{features['aria_count']}",

        "",

        "Generated using "
        "Original_full_dataN.csv."
    ])


    st.download_button(

        "⬇️ Download Report",

        "\n".join(report),

        file_name=
            "webguard_accessibility_report.txt",

        mime=
            "text/plain"
    )


# ============================================================
# ABOUT ML
# ============================================================

with st.expander(
    "🧠 About the ML Pipeline"
):

    st.write(
        """
        Accessify_mk2 uses the supplied accessibility
        dataset as its knowledge base.

        The ML pipeline converts violation names,
        categories and affected HTML elements into
        TF-IDF features and trains a Random Forest
        classifier to predict violation impact.

        During analysis, the live website is scanned
        for structural accessibility problems.

        Detected problems are matched with records
        from the supplied dataset.

        The final Accessibility Risk Score combines:

        • live violation severity

        • number of violations

        • violation diversity

        • dataset violation scores

        • violation density

        • website structural complexity

        This produces a risk score from 0 to 100,
        where lower is better.
        """
    )
