
import streamlit as st
import pandas as pd

from triagelens_inference import TriageLensPredictor


# --------------------------------------------------
# STEP 1: Configure application
# --------------------------------------------------

st.set_page_config(
    page_title="TriageLens",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# STEP 2: Load saved models once
# --------------------------------------------------

@st.cache_resource
def load_predictor():
    return TriageLensPredictor()


try:
    predictor = load_predictor()
except Exception as error:
    st.error(f"Unable to load TriageLens models: {error}")
    st.stop()


# --------------------------------------------------
# STEP 3: Application header
# --------------------------------------------------

st.title("🛡️ TriageLens")

st.subheader(
    "Confidence-Aware Human-in-the-Loop "
    "Intent Classification"
)

st.caption(
    "A master's research prototype using seven "
    "reliability signals to support automation decisions."
)

st.divider()


# --------------------------------------------------
# STEP 4: Sidebar — Research information
# --------------------------------------------------

with st.sidebar:

    st.header("About TriageLens")

    st.write(
        "TriageLens predicts user intents and estimates "
        "whether a prediction is reliable enough "
        "for automatic classification."
    )

    st.subheader("Frozen Policy Thresholds")

    st.metric(
        "Confidence threshold",
        f"{predictor.confidence_threshold:.2%}"
    )

    st.metric(
        "Trust threshold",
        f"{predictor.trust_threshold:.2%}"
    )

    st.divider()

    st.subheader("Research Configuration")

    st.write("**Dataset:** CLINC150")
    st.write("**Known intents:** 150")
    st.write("**Reliability signals:** 7")
    st.write("**Assumed OOS prevalence:** 18%")
    st.write("**Wrong automation / review cost:** 10:1")

    st.info(
        "This is a research demonstration. "
        "AUTOMATE means accepting the predicted intent, "
        "not executing a real banking or other action."
    )


# --------------------------------------------------
# STEP 5: User query input
# --------------------------------------------------

st.header("Analyze a User Query")

with st.form("query_form"):

    user_query = st.text_area(
        "Enter your query",
        value="I need to report my stolen credit card",
        height=100,
        placeholder="Type a sentence here..."
    )

    analyze_button = st.form_submit_button(
        "Analyze Query",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# STEP 6: Run inference
# --------------------------------------------------

if analyze_button:

    if not user_query.strip():

        st.warning("Please enter a query.")

    else:

        with st.spinner("Analyzing query..."):

            try:
                result = predictor.predict(user_query)
            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.stop()

        st.divider()

        # ------------------------------------------
        # Predicted intent
        # ------------------------------------------

        st.header("Prediction Results")

        st.write("**Predicted intent:**")
        st.code(result["predicted_intent"])

        # ------------------------------------------
        # Confidence and trust scores
        # ------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Classifier Confidence",
                f"{result['confidence']:.2%}"
            )

            st.progress(
                float(result["confidence"])
            )

        with col2:
            st.metric(
                "Learned Trust Score",
                f"{result['trust_score']:.2%}"
            )

            st.progress(
                float(result["trust_score"])
            )

        # ------------------------------------------
        # Decision comparison
        # ------------------------------------------

        st.subheader("Automation Decisions")

        col3, col4 = st.columns(2)

        with col3:
            st.write("**Confidence-Only Policy**")

            st.info(
                result["confidence_decision"]
            )

        with col4:
            st.write("**Seven-Signal TriageLens**")

            if result["triagelens_decision"] == "AUTOMATE":

                st.success(
                    "AUTOMATE — Accept intent classification"
                )

            else:

                st.warning(
                    "HUMAN REVIEW — Escalate query"
                )

        # ------------------------------------------
        # Threshold margin
        # ------------------------------------------

        trust_margin = (
            result["trust_score"]
            - predictor.trust_threshold
        )

        st.write(
            f"**Trust threshold margin:** "
            f"{trust_margin:+.4f}"
        )

        if abs(trust_margin) < 0.02:

            st.warning(
                "This decision is close to the "
                "automation threshold."
            )

        # ------------------------------------------
        # Seven reliability signals
        # ------------------------------------------

        st.divider()

        st.subheader("Seven Reliability Signals")

        signals_df = pd.DataFrame([
            {
                "Signal": signal,
                "Value": round(float(value), 4)
            }
            for signal, value in result["signals"].items()
        ])

        st.dataframe(
            signals_df,
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Understanding the Signals"):

            st.write(
                "**Confidence:** Primary classifier's "
                "maximum class probability."
            )

            st.write(
                "**TF-IDF similarity:** Similarity to "
                "the closest training query."
            )

            st.write(
                "**TF-IDF kNN agreement:** Weighted "
                "consensus among five lexical neighbors."
            )

            st.write(
                "**LSA similarity:** Similarity to the "
                "closest training query in LSA space."
            )

            st.write(
                "**LSA kNN agreement:** Weighted "
                "consensus among five LSA neighbors."
            )

            st.write(
                "**Model agreement:** Negative "
                "Jensen-Shannon distance. Values "
                "closer to zero mean greater agreement."
            )

            st.write(
                "**Classifier-LSA kNN match:** "
                "1 means both select the same intent; "
                "0 means they disagree."
            )

        st.caption(
            "Trust scores are statistical estimates, "
            "not guarantees. Human review does not "
            "necessarily imply that a query is OOS."
        )


# --------------------------------------------------
# STEP 7: Footer
# --------------------------------------------------

st.divider()

st.caption(
    "TriageLens | Master's Research Prototype | "
    "CLINC150 | Frozen Seven-Signal Trust Model"
)
