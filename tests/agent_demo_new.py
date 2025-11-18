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
    st.set_page_config(page_title="📧 Email QA Agent", layout="wide")
    st.title("📧 Intelligent Email QA Agent")
    st.write("""
    Analyze your email HTML against brand guidelines with bonus checks:
    readability, spam risk, link validation, and design consistency.
    """)

    # Sidebar Inputs
    st.sidebar.header("Inputs 🔧")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    html_file = st.sidebar.file_uploader("Email HTML File", type=["html", "htm"])
    reference_image = st.sidebar.file_uploader("Reference Image (optional)", type=["png", "jpg", "jpeg"])
    checklist_file = st.sidebar.file_uploader("Brand Guidelines (Excel)", type=["xls", "xlsx"])

    if st.sidebar.button("Run QA Analysis"):
        if not html_file:
            st.sidebar.error("Please upload an HTML file.")
            st.stop()

        with st.spinner("Initializing LLM client..."):
            llm_client = LLMClient(api_key=api_key)
            qa_agent = CopyQAAgent(llm_client)

        html_content = html_file.read().decode("utf-8")

        if checklist_file:
            import pandas as pd
            try:
                df = pd.read_excel(checklist_file)
                brand_guidelines = df.to_string(index=False)
            except Exception as e:
                st.sidebar.error(f"Error reading Excel file: {e}")
                brand_guidelines = "Error reading Excel checklist."
        else:
            brand_guidelines = "No checklist provided."

        image_bytes = reference_image.read() if reference_image else None

        with st.spinner("Running QA analysis... ⏳"):
            report = qa_agent.generate_report(html_content, image_bytes, brand_guidelines)

        st.success("✅ QA Analysis Complete!")

        # ===============================
        # Display Key Metrics (Cards)
        # ===============================
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Score", f"{report.overall_score:.1f}/100")
        col2.metric("Readability", f"{report.visual_consistency['readability']['flesch_reading_ease']:.1f}")
        col3.metric("Spam Risk", f"{report.visual_consistency['spam_risk']['spam_risk_score']}/100")
        col4.metric("Layout Consistency", f"{report.visual_consistency.get('layout_consistency', 0)}/100")

        # ===============================
        # Tabs for Detailed QA Results
        # ===============================
        tabs = st.tabs(["Tone & Brand", "Language Audit", "Content Accuracy", "Design & UX", "Insights & Actions"])

        with tabs[0]:
            st.header("📝 Tone & Brand")
            st.json(report.tone_analysis, expanded=False)

        with tabs[1]:
            st.header("✏️ Language Audit")
            if report.grammar_issues:
                for issue in report.grammar_issues:
                    with st.expander(f"{issue['location']} | {issue['severity'].capitalize()}"):
                        st.write(issue['issue'])
                        st.markdown(f"**Suggestion:** {issue['suggestion']}")
            else:
                st.success("No grammar issues detected ✅")

        with tabs[2]:
            st.header("📋 Content Accuracy")
            if report.message_deviations:
                for deviation in report.message_deviations:
                    with st.expander(f"{deviation['element']} | Impact: {deviation.get('impact','N/A')}"):
                        st.write(deviation['deviation'])
            else:
                st.success("No content deviations detected ✅")

        with tabs[3]:
            st.header("🖼️ Design & UX")
            st.subheader("Visual Consistency")
            st.json(report.visual_consistency, expanded=False)

        with tabs[4]:
            st.header("💡 Insights & Actions")
            for i, rec in enumerate(report.recommendations, 1):
                st.write(f"**{i}.** {rec}")

        # ===============================
        # Optional: Render HTML Preview
        # ===============================
        st.subheader("📧 Email Preview")
        st.components.v1.html(html_content, height=400, scrolling=True)

# ======================================================================
# 4. ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()