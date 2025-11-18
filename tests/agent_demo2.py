"""
📧 Intelligent Email QA Agent for Promotional Content
Analyzes email HTML, compares with reference images, validates against brand guidelines
and provides bonus features:
- Readability
- Spam risk
- Link validation
- Image alt text
- CTA effectiveness
- Mobile responsiveness
- Compliance
"""

import os
import base64
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import textstat

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
        self.model = "gpt-3.5-turbo"

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
    # Advanced Bonus Features
    # -------------------
    def analyze_image_alt_text(self, html_content: str) -> Dict:
        soup = BeautifulSoup(html_content, 'html.parser')
        images = soup.find_all('img')
        missing_alt = [img['src'] for img in images if not img.get('alt')]
        return {
            "total_images": len(images),
            "missing_alt_text": missing_alt,
            "score": max(0, 100 - len(missing_alt) * 10)
        }

    def analyze_cta_effectiveness(self, text_content: Dict) -> Dict:
        weak_ctas = ["click here", "learn more", "submit"]
        weak_found = [cta for cta in text_content['cta_buttons']
                      if any(w.lower() in cta.lower() for w in weak_ctas)]
        score = max(0, 100 - len(weak_found) * 20)
        return {
            "weak_ctas": weak_found,
            "cta_score": score
        }

    def check_mobile_responsiveness(self, html_content: str) -> Dict:
        soup = BeautifulSoup(html_content, 'html.parser')
        issues = []
        for table in soup.find_all('table'):
            width = table.get('width')
            if width and int(width) > 600:
                issues.append(f"Table width {width}px may break mobile layout")
        return {
            "issues": issues,
            "score": max(0, 100 - len(issues) * 20)
        }

    def check_compliance(self, html_content: str) -> Dict:
        soup = BeautifulSoup(html_content, 'html.parser')
        unsubscribe_links = [a for a in soup.find_all('a') if 'unsubscribe' in a.get_text().lower()]
        return {
            "unsubscribe_present": bool(unsubscribe_links),
            "score": 100 if unsubscribe_links else 0
        }

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

        # Advanced bonus features
        image_alt = self.analyze_image_alt_text(html_content)
        cta_eval = self.analyze_cta_effectiveness(text_content)
        mobile_resp = self.check_mobile_responsiveness(html_content)
        compliance = self.check_compliance(html_content)

        # Update recommendations
        recommendations = copy_analysis.get('recommendations', [])
        if readability["flesch_reading_ease"] < 50:
            recommendations.append("Email content is hard to read. Simplify sentences for better readability.")
        if spam_risk["spam_risk_score"] > 40:
            recommendations.append("High spam risk detected. Consider rephrasing flagged words.")
        if link_validation["invalid_links"]:
            recommendations.append(f"Found invalid links: {link_validation['invalid_links']}")
        if image_alt["missing_alt_text"]:
            recommendations.append(f"Images missing alt text: {image_alt['missing_alt_text']}")
        if cta_eval["weak_ctas"]:
            recommendations.append(f"Weak CTA phrases detected: {cta_eval['weak_ctas']}")
        if mobile_resp["issues"]:
            recommendations.append(f"Mobile responsiveness issues: {mobile_resp['issues']}")
        if compliance["score"] == 0:
            recommendations.append("Missing unsubscribe link. Add one to comply with regulations.")

        # Merge all bonus features
        visual_analysis.update({
            "readability": readability,
            "spam_risk": spam_risk,
            "link_validation": link_validation,
            "image_alt_text": image_alt,
            "cta_effectiveness": cta_eval,
            "mobile_responsiveness": mobile_resp,
            "compliance": compliance
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
    st.title("📧 Email QA Agent")
    st.write("Upload your HTML, reference image, and brand checklist to run the QA analysis.")

    # Sidebar Inputs
    st.sidebar.header("Inputs 🔧")
    load_dotenv()
    # api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or "")
    html_file = st.sidebar.file_uploader("Email HTML File", type=["html", "htm"])
    reference_image = st.sidebar.file_uploader("Reference Image (optional)", type=["png", "jpg", "jpeg"])
    checklist_file = st.sidebar.file_uploader("Brand Guidelines", type=["txt", "md"])

    if st.sidebar.button("Run QA Analysis"):
        if not html_file:
            st.sidebar.error("Please upload an HTML file.")
            st.stop()

        with st.spinner("Initializing LLM client..."):
            llm_client = LLMClient(api_key=api_key)
            qa_agent = CopyQAAgent(llm_client)

        html_content = html_file.read().decode("utf-8")
        brand_guidelines = checklist_file.read().decode("utf-8") if checklist_file else "No checklist provided."
        image_bytes = reference_image.read() if reference_image else None

        with st.spinner("Running QA analysis... ⏳"):
            report = qa_agent.generate_report(html_content, image_bytes, brand_guidelines)

        st.success("✅ QA Analysis Complete!")

        # -------------------
        # Key Metrics Cards
        # -------------------
        st.subheader("📊 Key Metrics")
        cols = st.columns(4)
        cols[0].metric("Overall Score", f"{report.overall_score:.1f}/100")
        cols[1].metric("Readability", f"{report.visual_consistency['readability']['flesch_reading_ease']:.1f}")
        cols[2].metric("Spam Risk", f"{report.visual_consistency['spam_risk']['spam_risk_score']}/100")
        cols[3].metric("Layout Consistency", f"{report.visual_consistency.get('layout_consistency', 0)}/100")

        cols2 = st.columns(4)
        cols2[0].metric("CTA Effectiveness", f"{report.visual_consistency['cta_effectiveness']['cta_score']}/100")
        cols2[1].metric("Image Alt Score", f"{report.visual_consistency['image_alt_text']['score']}/100")
        cols2[2].metric("Mobile Score", f"{report.visual_consistency['mobile_responsiveness']['score']}/100")
        cols2[3].metric("Compliance", f"{report.visual_consistency['compliance']['score']}/100")

        # -------------------
        # Detailed Tabs with Expanders
        # -------------------
        tabs = st.tabs(["Tone Analysis", "Grammar Issues", "Message Deviations", "Visual + Bonus", "Recommendations"])

        with tabs[0]:
            st.header("📝 Tone Analysis")
            st.table(report.tone_analysis.items())

        with tabs[1]:
            st.header("✏️ Grammar Issues")
            if report.grammar_issues:
                st.table(report.grammar_issues)
            else:
                st.success("No grammar issues detected ✅")

        with tabs[2]:
            st.header("📋 Message Deviations")
            if report.message_deviations:
                st.table(report.message_deviations)
            else:
                st.success("No deviations detected ✅")

        with tabs[3]:
            st.header("🖼️ Visual Consistency + Bonus Features")
            st.json(report.visual_consistency)

        with tabs[4]:
            st.header("💡 Recommendations")
            for i, rec in enumerate(report.recommendations, 1):
                st.write(f"**{i}.** {rec}")

        # Optional: Render HTML preview
        st.subheader("📧 Email Preview")
        st.components.v1.html(html_content, height=400, scrolling=True)

# ======================================================================
# 4. ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
