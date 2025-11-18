"""
📧 Intelligent Email QA Agent for Promotional Content
Analyzes email HTML, compares with reference images, validates against brand guidelines
and provides bonus features:
- Readability
- Spam risk
- Link validation
"""

import os
import base64
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
import json
import streamlit as st
import textstat
import re

# ======================================================================
# 1. LLM CLIENT INITIALIZATION
# ======================================================================

class LLMClient:
    """Initialize and manage OpenAI client for GPT-4o"""

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"

    def call(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Call GPT-4o model"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content

# ======================================================================
# 2. COPY QA AGENT
# ======================================================================

@dataclass
class QAReport:
    tone_analysis: Dict
    grammar_issues: List[Dict]
    message_deviations: List[Dict]
    visual_consistency: Dict
    overall_score: float
    recommendations: List[str]

class CopyQAAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    # -------------------
    # Core Methods
    # -------------------
    def extract_text_from_html(self, html_content: str) -> Dict[str, str]:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return {
            "subject": soup.find('title').get_text() if soup.find('title') else "",
            "headings": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])],
            "body_text": soup.get_text(separator='\n', strip=True),
            "cta_buttons": [a.get_text().strip() for a in soup.find_all('a', href=True)
                            if 'button' in str(a.get('class', [])).lower()],
            "links": [a.get_text().strip() for a in soup.find_all('a', href=True)]
        }

    def encode_image(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_copy_quality(self, text_content: Dict, brand_guidelines: str) -> Dict:
        prompt = f"""You are an expert email QA analyst. Analyze the following email content against the brand guidelines.

EMAIL CONTENT:
- Subject: {text_content['subject']}
- Headings: {', '.join(text_content['headings'])}
- Body: {text_content['body_text'][:500]}...
- CTAs: {', '.join(text_content['cta_buttons'])}

BRAND GUIDELINES:
{brand_guidelines}

Provide a detailed analysis in JSON format with:
1. tone_analysis: {{consistency: score(0-100), issues: [list], brand_voice_match: score}}
2. grammar_issues: [{{location: str, issue: str, severity: str, suggestion: str}}]
3. message_deviations: [{{element: str, deviation: str, impact: str}}]
4. recommendations: [list of actionable improvements]
"""

        messages = [
            {"role": "system", "content": "You are an expert QA analyst for marketing emails."},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.call(messages)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except json.JSONDecodeError:
            return {"tone_analysis": {}, "grammar_issues": [], "message_deviations": [], "recommendations": [response]}

    def compare_with_reference_image(self, html_content: str, reference_image_bytes: bytes, text_content: Dict) -> Dict:
        base64_image = self.encode_image(reference_image_bytes)

        prompt = f"""Compare this reference email design image with the actual content below.

ACTUAL EMAIL CONTENT:
- Subject: {text_content['subject']}
- Headings: {', '.join(text_content['headings'])}
- CTAs: {', '.join(text_content['cta_buttons'])}

Analyze:
1. Visual consistency
2. Element placement
3. Content accuracy
4. Missing elements

Provide JSON output:
{{
    "visual_match_score": score(0-100),
    "discrepancies": [{{element: str, issue: str, severity: str}}],
    "missing_elements": [list],
    "layout_consistency": score(0-100)
}}"""

        messages = [
            {"role": "system", "content": "You are an expert at comparing email designs."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]

        response = self.llm.call(messages)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            return json.loads(response)
        except json.JSONDecodeError:
            return {"visual_match_score": 0, "discrepancies": [], "missing_elements": [], "layout_consistency": 0}

    # -------------------
    # Bonus Features
    # -------------------
    def analyze_readability(self, text_content: Dict) -> Dict:
        readability_score = textstat.flesch_reading_ease(text_content['body_text'])
        grade_level = textstat.text_standard(text_content['body_text'], float_output=True)
        return {"flesch_reading_ease": readability_score, "grade_level": grade_level}

    def analyze_spam_risk(self, text_content: Dict) -> Dict:
        spam_trigger_words = ["free", "buy now", "urgent", "act now", "guarantee"]
        found = [w for w in spam_trigger_words if w.lower() in text_content['body_text'].lower()]
        score = min(len(found) * 20, 100)
        return {"spam_trigger_words": found, "spam_risk_score": score}

    def validate_links(self, text_content: Dict) -> Dict:
        url_pattern = re.compile(r'https?://\S+')
        valid_links = []
        invalid_links = []
        for link in text_content['links']:
            if re.match(url_pattern, link):
                valid_links.append(link)
            else:
                invalid_links.append(link)
        return {"valid_links": valid_links, "invalid_links": invalid_links}

    # -------------------
    # Generate Report
    # -------------------
    def generate_report(self, html_content: str, reference_image_bytes: Optional[bytes], brand_guidelines: str) -> QAReport:
        text_content = self.extract_text_from_html(html_content)
        copy_analysis = self.analyze_copy_quality(text_content, brand_guidelines)

        visual_analysis = {}
        if reference_image_bytes:
            visual_analysis = self.compare_with_reference_image(html_content, reference_image_bytes, text_content)
        else:
            visual_analysis = {"visual_match_score": 0, "discrepancies": [], "missing_elements": [], "layout_consistency": 0}

        # Bonus features
        readability = self.analyze_readability(text_content)
        spam_risk = self.analyze_spam_risk(text_content)
        link_validation = self.validate_links(text_content)

        # Update recommendations
        recommendations = copy_analysis.get('recommendations', [])
        if readability["flesch_reading_ease"] < 50:
            recommendations.append("Email content is hard to read. Simplify sentences for better readability.")
        if spam_risk["spam_risk_score"] > 40:
            recommendations.append("High spam risk detected. Consider rephrasing flagged words.")
        if link_validation["invalid_links"]:
            recommendations.append(f"Found invalid links: {link_validation['invalid_links']}")

        # Merge bonus features into visual_consistency for display
        visual_analysis.update({
            "readability": readability,
            "spam_risk": spam_risk,
            "link_validation": link_validation
        })

        tone_score = copy_analysis.get('tone_analysis', {}).get('consistency', 0)
        visual_score = visual_analysis.get('visual_match_score', 0)
        overall_score = tone_score if visual_score == 0 else (tone_score * 0.6 + visual_score * 0.4)

        return QAReport(
            tone_analysis=copy_analysis.get('tone_analysis', {}),
            grammar_issues=copy_analysis.get('grammar_issues', []),
            message_deviations=copy_analysis.get('message_deviations', []),
            visual_consistency=visual_analysis,
            overall_score=overall_score,
            recommendations=recommendations
        )

# ======================================================================
# 3. STREAMLIT FRONTEND
# ======================================================================

def main():
    import base64

    # Configure Streamlit page
    st.set_page_config(page_title="📧 Email QA Agent", layout="wide")

    # ============================
    # Load Logo
    # ============================
    # Helper to load and encode the brand logo for the header bar.
    def load_logo(path: str):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # Load logo from the local project directory
    logo_base64 = load_logo("DGS_logo.png")

    # ============================
    # Header Bar (branding + title)
    # ============================
    # This creates a clean branded header so the tool feels polished and official.
    st.markdown("""
        <style>
            .header-bar {
                display: flex;
                align-items: center;
                gap: 20px;
                padding: 15px 20px;
                background: #f5f7fa;
                border-bottom: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-bottom: 25px;
            }
            .header-title {
                font-size: 32px;
                font-weight: 700;
                color: #333;
                margin: 0;
            }
            .header-logo {
                height: 55px;
                width: auto;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="header-bar">
            <img src="data:image/png;base64,{logo_base64}" class="header-logo">
            <h1 class="header-title">  Campaign Custodian  </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Simple instruction for users
    st.write("Upload your HTML, reference image, and brand checklist to run the QA analysis.")

    # ============================
    # Inputs Section
    # ============================
    # This section collects all user inputs required to run QA:
    # - Email HTML
    # - Optional reference design image
    # - Brand guideline checklist
    st.header("Inputs 🔧")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    col1, col2, col3 = st.columns(3)

    with col1:
        html_file = st.file_uploader("Email HTML File", type=["html", "htm", "eml"])
    with col2:
        reference_image = st.file_uploader("Reference Image (optional)", type=["png", "jpg", "jpeg"])
    with col3:
        checklist_file = st.file_uploader("Brand Guidelines (Excel)", type=["xls", "xlsx"])

    # Helper: color-coding boxes by score
    def score_color(score):
        # Green = strong, Yellow = moderate, Red = needs attention
        if score >= 80:
            return "#e6ffed"
        elif score >= 60:
            return "#fff7e6"
        return "#ffecec"

    # ============================
    # Run QA Button
    # ============================
    if st.button("Run QA Analysis"):
        if not html_file:
            st.error("Please upload an HTML file.")
            st.stop()

        # Initialize LLM client
        with st.spinner("Initializing LLM client..."):
            llm_client = LLMClient(api_key=api_key)
            qa_agent = CopyQAAgent(llm_client)

        html_content = html_file.read().decode("utf-8")

        # Load brand checklist (Excel → text)
        # This converts the checklist rows into readable text for the LLM.
        if checklist_file:
            try:
                import pandas as pd
                df = pd.read_excel(checklist_file)
                brand_guidelines = df.to_string(index=False)
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                brand_guidelines = "Error reading Excel checklist."
        else:
            brand_guidelines = "No checklist provided."

        # Load reference design image (if uploaded)
        image_bytes = reference_image.read() if reference_image else None

        # Run the full QA analysis pipeline
        with st.spinner("Running QA analysis... ⏳"):
            report = qa_agent.generate_report(html_content, image_bytes, brand_guidelines)

        st.success("✅ QA Analysis Complete!")

        # ============================
        # SUMMARY DASHBOARD
        # ============================
        # High-level overview of results:
        # - Overall score
        # - Readability
        # - Spam risk
        # - Layout consistency
        st.markdown("---")
        st.subheader("📊 Summary Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        box_style = """
            padding:18px; 
            border-radius:10px; 
            background:{bg_color}; 
            text-align:center; 
            display:flex; 
            flex-direction:column; 
            justify-content:center; 
            align-items:center;
        """

        header_style = "font-size:18px; font-weight:600; margin-bottom:8px;"
        score_style = "font-size:32px; font-weight:bold;"

        # Overall score = combined tone + visual + readability + spam + grammar
        with col1:
            st.markdown(f"""
                <div style="{box_style.format(bg_color=score_color(report.overall_score))}">
                    <div style="{header_style}">⭐ Overall Score</div>
                    <span style="{score_style}">{report.overall_score:.1f}</span>
                </div>
            """, unsafe_allow_html=True)

        # Readability score = Flesch Reading Ease
        with col2:
            readability = report.visual_consistency.get("readability", {})
            st.markdown(f"""
                <div style="{box_style.format(bg_color=score_color(readability.get('flesch_reading_ease', 0)))}">
                    <div style="{header_style}">📗 Readability</div>
                    <span style="{score_style}">{readability.get('flesch_reading_ease', 0):.1f}</span>
                </div>
            """, unsafe_allow_html=True)

        # Spam risk score = count of trigger words
        with col3:
            spam = report.visual_consistency.get("spam_risk", {})
            st.markdown(f"""
                <div style="{box_style.format(bg_color=score_color(100 - spam.get('spam_risk_score', 0)))}">
                    <div style="{header_style}">🚫 Spam Risk</div>
                    <span style="{score_style}">{spam.get('spam_risk_score', 0)}</span>
                </div>
            """, unsafe_allow_html=True)

        # Layout consistency = visual similarity to reference
        with col4:
            layout_score = report.visual_consistency.get("layout_consistency", 0)
            st.markdown(f"""
                <div style="{box_style.format(bg_color=score_color(layout_score))}">
                    <div style="{header_style}">🧩 Layout Consistency</div>
                    <span style="{score_style}">{layout_score}</span>
                </div>
            """, unsafe_allow_html=True)

        # ============================
        # DETAIL TABS
        # ============================
        # These tabs help QA reviewers dive deeper into each category.
        tabs = st.tabs([
            "📝 Tone Analysis", 
            "✏️ Grammar Issues", 
            "📋 Message Deviations", 
            "🖼 Visual & Bonus Checks", 
            "💡 Recommendations"
        ])

        # --- Tone Analysis Tab ---
        with tabs[0]:
            st.markdown("## 📝 Tone Analysis")
            st.markdown("*This section checks how well the email matches the expected brand voice.*")
            tone = report.tone_analysis

            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; background:#f7faff;">
                <b>Consistency Score:</b> {tone.get('consistency', 'N/A')} / 100 <br>
                <b>Brand Voice Match:</b> {tone.get('brand_voice_match', 'N/A')} / 100
            </div>
            """, unsafe_allow_html=True)

            if tone.get("issues"):
                st.markdown("### Tone Issues Identified")
                for issue in tone["issues"]:
                    st.markdown(f"- {issue}")
            else:
                st.success("No tone issues detected.")

        # --- Grammar Issues Tab ---
        with tabs[1]:
            st.markdown("## ✏️ Grammar Issues")
            st.markdown("*Grammar, punctuation, clarity, and structural issues.*")
            if report.grammar_issues:
                for issue in report.grammar_issues:
                    with st.expander(f"⚠️ {issue['location']} — {issue['severity']}", expanded=issue['severity']=="high"):
                        st.write(issue)
            else:
                st.success("No grammar issues detected.")

        # --- Message Deviations Tab ---
        with tabs[2]:
            st.markdown("## 📋 Message Deviations")
            st.markdown("*Checks for misalignment with brand guidelines or required content.*")
            if report.message_deviations:
                for item in report.message_deviations:
                    st.markdown(f"""
                    **Element:** {item['element']}  
                    **Issue:** {item['deviation']}  
                    **Impact:** :orange[{item['impact']}]  
                    """)
                    st.divider()
            else:
                st.success("No message deviations found.")

        # --- Visual + Bonus Tab ---
        with tabs[3]:
            st.markdown("## 🖼 Visual Consistency & Bonus Checks")
            st.markdown("*How well the email matches the reference layout + readability + spam + links.*")

            with st.expander("🔍 Visual Summary"):
                st.json({
                    "visual_match_score": report.visual_consistency.get("visual_match_score"),
                    "layout_consistency": report.visual_consistency.get("layout_consistency"),
                    "missing_elements": report.visual_consistency.get("missing_elements"),
                    "discrepancies": report.visual_consistency.get("discrepancies"),
                })

            with st.expander("📗 Readability"):
                brief = report.visual_consistency.get("readability", {})
                st.markdown(f"""
                **Flesch Ease:** {brief.get('flesch_reading_ease', 'N/A')}  
                **Grade Level:** {brief.get('grade_level', 'N/A')}  
                """)

            with st.expander("🚫 Spam Risk"):
                s = report.visual_consistency.get("spam_risk", {})
                st.markdown(f"""
                **Spam Score:** {s.get('spam_risk_score', 'N/A')}  
                **Trigger Words:** {', '.join(s.get('spam_trigger_words', [])) or 'None'}  
                """)

            with st.expander("🔗 Link Validation"):
                lv = report.visual_consistency.get("link_validation", {})
                st.markdown(f"""
                **Valid Links:**  
                {', '.join(lv.get('valid_links', [])) or 'None'}  

                **Invalid Links:**  
                {', '.join(lv.get('invalid_links', [])) or 'None'}  
                """)

        # --- Recommendations Tab ---
        with tabs[4]:
            st.markdown("## 💡 Recommendations")
            st.markdown("*Actionable improvements based on tone, grammar, layout, and clarity.*")
            for i, rec in enumerate(report.recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

        # --- Email Preview (raw HTML render) ---
        st.markdown("---")
        st.subheader("📧 Email Preview")
        st.components.v1.html(
            f"<div style='border:1px solid #ddd; padding:10px;'>{html_content}</div>",
            height=450,
            scrolling=True
        )

# ======================================================================
# 4. ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()