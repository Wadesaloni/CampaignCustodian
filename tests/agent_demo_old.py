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
from typing import Dict, List, Optional, Any, Tuple
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
import json
import streamlit as st
import textstat
import re
import pandas as pd

from urllib.parse import urlparse, urlunparse, urljoin, parse_qs
from email import policy
from email.parser import BytesParser

# CrewAI imports (agent orchestration)
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

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

# -----------------------
# Utility functions (deterministic)
# -----------------------

def deterministic_normalize_url(raw: str, base: str = None) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.startswith("data:"):
        return s
    if base and not bool(urlparse(s).netloc):
        s = urljoin(base, s)
    try:
        p = urlparse(s)
    except Exception:
        return s
    query = parse_qs(p.query, keep_blank_values=True)
    filtered_q = {k: v for k, v in query.items() if not re.match(r"^(utm_|gclid|fbclid)", k, flags=re.I)}
    qs_parts = []
    for k in sorted(filtered_q.keys()):
        vals = filtered_q[k]
        if not vals:
            qs_parts.append(k)
        else:
            qs_parts.append("{}={}".format(k, ",".join(vals)))
    new_query = "&".join(qs_parts)
    normalized = urlunparse((p.scheme, p.netloc.lower(), p.path or '/', p.params, new_query, ""))
    return normalized


# In[120]:


# Tools: wrap the core functions so AI Tasks can call them.
# Each tool is deterministic, except where we explicitly call the LLM via a separate tool.

# from typing import List
# import pandas as pd
# from ai.tools import tool

@tool
def read_excel_tool(xlsx_path: str, main_entity_col: str = 'MainEntityName') -> List[Dict[str, Any]]:
    """
    Read Excel and return list of dicts for rows where MainEntityName == 'All Other' (case-insensitive).
    Ensures keys: LogoURL, BannerImage, Mod2, Mod3, MainEntityName are present in each dict.
    """
    expected_cols = ["LogoURL", "BannerImage", "Mod2", "Mod3", main_entity_col]
    df = pd.read_excel(xlsx_path, dtype=str)

    # Map expected canonical columns to actual headers (case-insensitive, contains fallback)
    found_map = {}
    for expected in expected_cols:
        exact = [c for c in df.columns if c.strip().lower() == expected.strip().lower()]
        if exact:
            found_map[expected] = exact[0]
        else:
            contains = [c for c in df.columns if expected.strip().lower() in c.strip().lower()]
            found_map[expected] = contains[0] if contains else None

    # Build canonical DataFrame with all expected keys
    canon_df = pd.DataFrame()
    for k in expected_cols:
        if found_map.get(k):
            canon_df[k] = df[found_map[k]].astype(str).where(~df[found_map[k]].isna(), None)
        else:
            canon_df[k] = [None] * len(df)

    # Filter rows where MainEntityName == 'All Other' (case-insensitive trim)
    def is_all_other(x):
        if x is None:
            return False
        return str(x).strip().lower() == 'all other'

    mask = canon_df[main_entity_col].apply(is_all_other)
    filtered = canon_df.loc[mask].reset_index(drop=True)

    # Return list of dicts; each dict contains LogoURL,BannerImage,Mod2,Mod3,MainEntityName
    return filtered.to_dict(orient='records')


# In[124]:


@tool
def parse_eml_tool(eml_path: str) -> Dict[str, Any]:
    """
    Parse a .eml file and return structured artifacts useful for matching.
    Returns a dict with keys:
      - html: (str) the HTML body or empty string
      - images: (List[str]) src attributes from <img>
      - anchors: (List[str]) href values from <a>
      - raw_attrs: (List[Tuple[tag, attr_name, attr_value]]) for background/style/src/href
      - text_urls: (List[str]) absolute URLs found in plain text nodes

    The function reads the .eml file from `eml_path` and parses the HTML part using BeautifulSoup.
    """
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    html = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/html':
                html = part.get_content()
                break
    else:
        if msg.get_content_type() == 'text/html':
            html = msg.get_content()

    soup = BeautifulSoup(html or "", 'html.parser')
    images = []
    anchors = []
    raw_attrs = []

    # Extract <img src>
    for img in soup.find_all('img'):
        src = img.get('src')
        if src:
            images.append(src)
            raw_attrs.append(('img', 'src', src))

    # Extract <a href>
    for a in soup.find_all('a'):
        href = a.get('href')
        if href:
            anchors.append(href)
            raw_attrs.append(('a', 'href', href))

    # background attributes and url(...) in style attributes
    for tag in soup.find_all(True):
        bg = tag.get('background')
        if bg:
            raw_attrs.append((tag.name, 'background', bg))
        style = tag.get('style')
        if style and 'url(' in style:
            urls = re.findall(r'url\\(([^)]+)\\)', style)
            for u in urls:
                raw_attrs.append((tag.name, 'style:url', u.strip("\"'")))

    text = soup.get_text(separator=' ')
    text_urls = [m.group(0) for m in re.finditer(r'https?://[^\\s\"<>\\)]+', text)]

    return {
        'html': html or "",
        'images': images,
        'anchors': anchors,
        'raw_attrs': raw_attrs,
        'text_urls': text_urls
    }


# In[125]:


@tool
def compare_tool(excel_rows: List[Dict[str, Any]], eml_struct: Dict[str, Any], base_url: str = None) -> List[Dict[str, Any]]:
    """
    For each filtered excel row, evaluate each of the four columns (LogoURL,BannerImage,Mod2,Mod3).
    Emit one output record per non-empty column for the selected rows. Each record contains row_index
    (index within the filtered list), source_column, source_value, normalized_value, matched (bool),
    match_symbol, match_locations, comparison_method.
    """
    priority_cols = ["LogoURL","BannerImage","Mod2","Mod3"]

    # Build eml candidate normalization map
    eml_candidates = []
    for tag, attr, val in eml_struct.get('raw_attrs', []):
        eml_candidates.append((val,f"eml:{tag}[{attr}]") )
    for v in eml_struct.get('images',[]):
        eml_candidates.append((v,'eml:img[src]'))
    for v in eml_struct.get('anchors',[]):
        eml_candidates.append((v,'eml:a[href]'))
    for v in eml_struct.get('text_urls',[]):
        eml_candidates.append((v,'eml:text'))

    eml_norm_map = {}
    for raw_val, loc in eml_candidates:
        norm = deterministic_normalize_url(raw_val, base=base_url)
        eml_norm_map.setdefault(norm,set()).add(loc)
        eml_norm_map.setdefault(raw_val,set()).add(loc)

    records = []
    for row_idx, row in enumerate(excel_rows):
        for col in priority_cols:
            raw_val = row.get(col)
            if raw_val is None or str(raw_val).strip() == '':
                continue
            val_str = str(raw_val).strip()
            norm_val = deterministic_normalize_url(val_str, base=base_url)

            matched = False
            match_locations = []
            comparison_method = None

            if val_str in eml_norm_map:
                matched = True
                match_locations = list(eml_norm_map[val_str])
                comparison_method = 'exact'
            elif norm_val in eml_norm_map:
                matched = True
                match_locations = list(eml_norm_map[norm_val])
                comparison_method = 'normalized'
            else:
                for candidate, locs in eml_norm_map.items():
                    if candidate and val_str in candidate:
                        matched = True
                        match_locations = list(locs)
                        comparison_method = 'substring'
                        break

            records.append({
                'row_index': row_idx,
                'source_column': col,
                'source_value': val_str,
                'normalized_value': norm_val,
                'matched': bool(matched),
                'match_symbol': '✔️' if matched else '❌',
                'match_locations': match_locations,
                'comparison_method': comparison_method or 'none'
            })
    return records



# In[127]:


@tool
def llm_review_tool(ambiguous_items: List[Dict[str, Any]], llm_model: str = 'gpt-4o', max_items_per_call: int = 10) -> List[Dict[str, Any]]:
    """
    Inspect ambiguous url-like items with an LLM and return suggestions/explanations.

    Input:
      - ambiguous_items: list of dicts with keys: source_column, source_value, normalized_value (optional)
      - llm_model: model identifier used by AI LLM wrapper
      - max_items_per_call: batch size for LLM prompting

    Output:
      - returns a list of dicts (one per input item) augmented with:
          - llm_suggestion: (str|None) cleaned/normalized URL suggestion or None
          - llm_explanation: (str|None) short explanation or raw LLM text if parsing failed

    IMPORTANT: This tool DOES NOT change deterministic matched flags. It only provides suggestions for human review.
    """
    import json
    results: List[Dict[str, Any]] = []

    # instantiate client using AI wrapper (reads OPENAI_API_KEY from env)
    llm_client = LLM(model=llm_model, api_key=OPENAI_API_KEY)

    def _safe_call(prompt_text: str):
        """
        Try a couple of common invocation styles for the LLM wrapper.
        Return the raw response object or raise RuntimeError if none work.
        """
        # try common call styles defensively
        try:
            # common: call(prompt=...)
            return llm_client.call(prompt=prompt_text)
        except TypeError:
            pass
        except Exception as e:
            # network/auth errors should surface
            raise

        try:
            # some wrappers expect input as dict
            return llm_client.call({"input": prompt_text})
        except Exception:
            pass

        try:
            # some wrappers accept a single positional string
            return llm_client.call(prompt_text)
        except Exception:
            pass

        # if none worked, surface a helpful error so you can inspect dir(llm_client)
        raise RuntimeError(f"LLM client doesn't accept known call signatures. Inspect dir(llm_client).")

    def _extract_text(resp: Any) -> str:
        """Best-effort extraction of meaningful text from common response shapes."""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            # common keys
            for k in ("text", "output_text", "content"):
                if k in resp and isinstance(resp[k], str):
                    return resp[k]
            # openai-like choices
            if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                first = resp["choices"][0]
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict):
                        return first["message"].get("content") or first["message"].get("text") or str(first)
                    if "text" in first:
                        return first["text"]
                return str(first)
            if "generations" in resp and isinstance(resp["generations"], list) and resp["generations"]:
                gen = resp["generations"][0]
                if isinstance(gen, dict):
                    return gen.get("text") or gen.get("output_text") or str(gen)
                return str(gen)
            return str(resp)
        # fallback to attribute access
        if hasattr(resp, "text"):
            try:
                return resp.text
            except Exception:
                pass
        if hasattr(resp, "content"):
            try:
                return resp.content
            except Exception:
                pass
        return str(resp)

    # Batch items to reduce token usage
    for i in range(0, len(ambiguous_items), max_items_per_call):
        batch = ambiguous_items[i:i + max_items_per_call]

        prompt_lines = [
            "You are a careful assistant. For each item below (a possibly-messy URL-like string),",
            "return a JSON array where each element is an object with keys:",
            "  - llm_suggestion: cleaned URL or empty string if none",
            "  - explanation: 1-2 short sentences explaining why this may or may not match.",
            "Respond ONLY with valid JSON (an array).",
            "Items:"
        ]
        for it in batch:
            prompt_lines.append(f"- column: {it.get('source_column','')} | value: {it.get('source_value','')}")
        prompt = "\n".join(prompt_lines)

        # call the LLM defensively
        try:
            raw = _safe_call(prompt)
        except Exception as err:
            # If the LLM can't be called, attach the error message as explanation and continue
            for it in batch:
                out = it.copy()
                out["llm_suggestion"] = None
                out["llm_explanation"] = f"LLM call failed: {err}"
                results.append(out)
            continue

        text = _extract_text(raw)

        # parse JSON output
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                parsed = [parsed]
            for item, suggestion in zip(batch, parsed):
                out = item.copy()
                if isinstance(suggestion, dict):
                    out["llm_suggestion"] = suggestion.get("llm_suggestion")
                    out["llm_explanation"] = suggestion.get("explanation")
                else:
                    out["llm_suggestion"] = None
                    out["llm_explanation"] = str(suggestion)
                results.append(out)
        except Exception:
            # fallback: store raw text as explanation
            for it in batch:
                out = it.copy()
                out["llm_suggestion"] = None
                out["llm_explanation"] = text[:1000]
                results.append(out)

    return results


# In[129]:


@tool("llm_review_tool")
def llm_review_tool(ambiguous_items: List[Dict[str, Any]], llm_model: str = 'gpt-4o', max_items_per_call: int = 10) -> List[Dict[str, Any]]:
    """
    Use an LLM to produce conservative suggestions for ambiguous URL-like items.

    Inputs:
      - ambiguous_items: list of dicts with keys: source_column, source_value, normalized_value (optional)
      - llm_model: model id used by AI's LLM wrapper
      - max_items_per_call: number of items to batch per LLM call

    Output:
      - List[Dict]: each input dict augmented with:
          - llm_suggestion: (str|None) suggested normalized URL
          - llm_explanation: (str|None) brief explanation or raw LLM text
    Note: This tool does NOT change deterministic matched flags; it only returns suggestions/explanations.
    """
    import json
    results: List[Dict[str, Any]] = []

    # instantiate LLM client (reads OPENAI_API_KEY from env)
    llm_client = LLM(model=llm_model, api_key=OPENAI_API_KEY)

    def _safe_call(prompt_text: str):
        """
        Try a few common invocation styles for the AI LLM client and return the raw response.
        Raises RuntimeError if no known call signature works.
        """
        # 1) call(prompt=...)
        try:
            return llm_client.call(prompt=prompt_text)
        except TypeError:
            pass
        except Exception:
            raise

        # 2) call({'input': prompt}) or call(prompt_text) positional
        try:
            return llm_client.call({"input": prompt_text})
        except Exception:
            pass
        try:
            return llm_client.call(prompt_text)
        except Exception:
            pass

        # 3) give a helpful error so you can inspect dir(llm_client)
        raise RuntimeError("LLM client did not accept known call signatures. Inspect dir(llm_client).")

    def _extract_text(resp: Any) -> str:
        """Best-effort extraction of textual output from common response shapes."""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            for k in ("text", "output_text", "content"):
                if k in resp and isinstance(resp[k], str):
                    return resp[k]
            if "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                first = resp["choices"][0]
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict):
                        return first["message"].get("content") or first["message"].get("text") or str(first)
                    if "text" in first:
                        return first["text"]
                return str(first)
            if "generations" in resp and isinstance(resp["generations"], list) and resp["generations"]:
                gen = resp["generations"][0]
                if isinstance(gen, dict):
                    return gen.get("text") or gen.get("output_text") or str(gen)
                return str(gen)
            return str(resp)
        if hasattr(resp, "text"):
            try:
                return resp.text
            except Exception:
                pass
        if hasattr(resp, "content"):
            try:
                return resp.content
            except Exception:
                pass
        return str(resp)

    # Batch ambiguous items
    for i in range(0, len(ambiguous_items), max_items_per_call):
        batch = ambiguous_items[i:i + max_items_per_call]
        prompt_lines = [
            "You are a careful assistant. For each item below (a possibly-messy URL-like string),",
            "return a JSON array where each element is an object with keys:",
            "  - llm_suggestion: cleaned URL or empty string if none",
            "  - explanation: 1-2 short sentences explaining why this may or may not match.",
            "Respond ONLY with valid JSON (an array).",
            "Items:"
        ]
        for it in batch:
            prompt_lines.append(f"- column: {it.get('source_column','')} | value: {it.get('source_value','')}")
        prompt = "\n".join(prompt_lines)

        # call LLM defensively
        try:
            raw = _safe_call(prompt)
        except Exception as err:
            # If LLM isn't callable, attach error as explanation to each item and continue
            for it in batch:
                out = it.copy()
                out["llm_suggestion"] = None
                out["llm_explanation"] = f"LLM call failed: {err}"
                results.append(out)
            continue

        text = _extract_text(raw)

        # parse JSON from LLM
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                parsed = [parsed]
            for item, suggestion in zip(batch, parsed):
                out = item.copy()
                if isinstance(suggestion, dict):
                    out["llm_suggestion"] = suggestion.get("llm_suggestion")
                    out["llm_explanation"] = suggestion.get("explanation")
                else:
                    out["llm_suggestion"] = None
                    out["llm_explanation"] = str(suggestion)
                results.append(out)
        except Exception:
            # fallback: store raw text as explanation
            for it in batch:
                out = it.copy()
                out["llm_suggestion"] = None
                out["llm_explanation"] = text[:1000]
                results.append(out)

    return results



# In[130]:


# ParserAgent: orchestrates reading, parsing, comparing, and optional LLM review
class ParserAgent:
    def __init__(self, llm_enabled: bool = True, llm_model: str = 'gpt-4o'):
        self.llm_enabled = llm_enabled and bool(OPENAI_API_KEY)
        self.llm_model = llm_model
        self.priority_cols = ["LogoURL","BannerImage","Mod2","Mod3"]

    def run(self, xlsx_path: str, eml_path: str, base_url: str = None) -> pd.DataFrame:
        # 1) Read filtered excel rows (only rows where MainEntityName == 'All Other')
        excel_rows = read_excel_tool.func(xlsx_path)
        if not excel_rows:
            # return empty DataFrame with expected columns
            cols = ['row_index','source_column','source_value','normalized_value','matched','match_symbol','match_locations','comparison_method','llm_suggestion','llm_explanation']
            return pd.DataFrame(columns=cols)

        # 2) Parse EML
        eml_struct = parse_eml_tool.func(eml_path)

        # 3) Compare: evaluate each of the 4 columns and get records for every non-empty column
        compare_records = compare_tool.func(excel_rows, eml_struct, base_url=base_url)
        compare_df = pd.DataFrame(compare_records) if compare_records else pd.DataFrame(columns=['row_index','source_column','source_value','normalized_value','matched','match_symbol','match_locations','comparison_method'])

        # 4) Build final outputs: here we keep one record per non-empty column (no early break)
        outputs = []
        for row_idx, row in enumerate(excel_rows):
            for col in self.priority_cols:
                # find compare entries for this row_idx and column
                matches = compare_df[(compare_df['row_index'] == row_idx) & (compare_df['source_column'] == col)] if not compare_df.empty else pd.DataFrame()
                if matches.empty:
                    # If compare didn't produce any record (maybe column was missing), but the excel has value, create an unmatched entry
                    val = row.get(col)
                    if val is None or str(val).strip() == '':
                        continue
                    outputs.append({
                        'row_index': row_idx,
                        'source_column': col,
                        'source_value': str(val).strip(),
                        'normalized_value': deterministic_normalize_url(val, base=base_url),
                        'matched': False,
                        'match_symbol': '❌',
                        'match_locations': [],
                        'comparison_method': 'none'
                    })
                else:
                    # there may be multiple compare rows (but compare_tool emits single per column)
                    for _, rec in matches.iterrows():
                        outputs.append({
                            'row_index': int(rec['row_index']),
                            'source_column': rec['source_column'],
                            'source_value': rec['source_value'],
                            'normalized_value': rec['normalized_value'],
                            'matched': bool(rec['matched']),
                            'match_symbol': rec['match_symbol'],
                            'match_locations': rec.get('match_locations', []),
                            'comparison_method': rec.get('comparison_method', 'none')
                        })

        result_df = pd.DataFrame(outputs)

        # 5) Optional LLM review for unmatched entries
        if self.llm_enabled:
            ambiguous = []
            for _, r in result_df.iterrows():
                if not r['matched']:
                    ambiguous.append({'source_column': r['source_column'], 'source_value': r['source_value'], 'normalized_value': r['normalized_value']})
            if ambiguous:
                llm_results = llm_review_tool.func(ambiguous, llm_model=self.llm_model)
                lookup = {(it['source_column'], it['source_value']): it for it in llm_results}
                llm_sugg, llm_expl = [], []
                for _, r in result_df.iterrows():
                    key = (r['source_column'], r['source_value'])
                    item = lookup.get(key)
                    llm_sugg.append(item.get('llm_suggestion') if item else None)
                    llm_expl.append(item.get('llm_explanation') if item else None)
                result_df['llm_suggestion'] = llm_sugg
                result_df['llm_explanation'] = llm_expl

        # Ensure columns exist
        expected_cols = ['row_index','source_column','source_value','normalized_value','matched','match_symbol','match_locations','comparison_method','llm_suggestion','llm_explanation']
        for c in expected_cols:
            if c not in result_df.columns:
                result_df[c] = None

        return result_df


# In[133]:


# ======================================================================
# 3. STREAMLIT FRONTEND
# ======================================================================

# Inside main() replace the UI part with:

def main():
    st.set_page_config(page_title="📧 Email QA Agent", layout="wide")
    st.title("Campaign Custodian")
    st.write("Upload your files and brand checklist to run the QA analysis.")

    # ----------------------------------------------------------
# Horizontal Inputs Row Under Header
# ----------------------------------------------------------

        st.markdown("🔧 Input Files")

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")

        # 4 equal columns for nice horizontal layout
        col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 1])

        with col_a:
            html_file = st.file_uploader("📄 Email HTML File", type=["html", "htm"])

        with col_b:
            reference_image = st.file_uploader("🖼 Reference Image (optional)", type=["png", "jpg", "jpeg"])

        with col_c:
            checklist_file = st.file_uploader("📘 Brand Guidelines (Excel)", type=["xls", "xlsx"])

        with col_d:
            run_clicked = st.button("🚀 Run QA")

        # Stop execution unless button clicked
        if run_clicked:
            if not html_file:
                st.error("Please upload an HTML file before running the QA.")
                st.stop()


        # ----------------------------
        # Load LLM + Inputs
        # ----------------------------
        with st.spinner("Initializing LLM client..."):
            llm_client = LLMClient(api_key=api_key)
            qa_agent = CopyQAAgent(llm_client)

        html_content = html_file.read().decode("utf-8")

        # Read Excel brand guidelines
        if checklist_file:
            try:
                import pandas as pd
                df = pd.read_excel(checklist_file)
                brand_guidelines = df.to_string(index=False)
            except Exception as e:
                st.sidebar.error(f"Error reading Excel file: {e}")
                brand_guidelines = "Error reading Excel checklist."
        else:
            brand_guidelines = "No checklist provided."

        image_bytes = reference_image.read() if reference_image else None

        # ----------------------------
        # Run QA Analysis
        # ----------------------------
        with st.spinner("Running QA analysis... ⏳"):
            report = qa_agent.generate_report(html_content, image_bytes, brand_guidelines)

        st.success("✅ QA Analysis Complete!")

        # ======================================================
        # 📊 Summary Dashboard
        # ======================================================
        st.markdown("---")
        st.subheader("📊 Summary Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### ⭐ Overall Score")
            st.markdown(f"""
                <div style="padding:18px; border-radius:10px; background:#eef6ff; text-align:center;">
                    <span style="font-size:32px; font-weight:bold;">{report.overall_score:.1f}</span><br>

                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📗 Readability")
            st.markdown(f"""
                <div style="padding:18px; border-radius:10px; background:#f0fff4; text-align:center;">
                    <span style="font-size:32px; font-weight:bold;">{report.visual_consistency['readability']['flesch_reading_ease']:.1f}</span>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("### 🚫 Spam Risk")
            st.markdown(f"""
                <div style="padding:18px; border-radius:10px; background:#fff5f5; text-align:center;">
                    <span style="font-size:32px; font-weight:bold;">{report.visual_consistency['spam_risk']['spam_risk_score']}</span>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("### 🧩 Layout Consistency")
            st.markdown(f"""
                <div style="padding:18px; border-radius:10px; background:#fdf7e3; text-align:center;">
                    <span style="font-size:32px; font-weight:bold;">{report.visual_consistency.get('layout_consistency', 0)}</span>
                </div>
            """, unsafe_allow_html=True)

        # ======================================================
        # Tabs Section
        # ======================================================

        tabs = st.tabs([
            "📝 Tone Analysis", 
            "✏️ Grammar Issues", 
            "📋 Message Deviations", 
            "🖼 Visual & Bonus Checks", 
            "💡 Recommendations"
        ])

        # ---------------------------
        # 📝 Tone Analysis
        # ---------------------------
        with tabs[0]:
            st.markdown("## 📝 Tone Analysis")
            tone = report.tone_analysis

            st.markdown(f"""
            <div style="padding:15px; border-radius:8px; background:#f6f9ff;">
                <b>Consistency Score:</b> {tone.get('consistency', 'N/A')} / 100 <br>
                <b>Brand Voice Match:</b> {tone.get('brand_voice_match', 'N/A')} / 100
            </div>
            """, unsafe_allow_html=True)

            if tone.get("issues"):
                st.markdown("### Issues Identified")
                for issue in tone["issues"]:
                    st.markdown(f"- {issue}")
            else:
                st.success("No tone issues detected.")

        # ---------------------------
        # ✏️ Grammar Issues
        # ---------------------------
        with tabs[1]:
            st.markdown("## ✏️ Grammar Issues")

            if report.grammar_issues:
                st.markdown("### Issues Found")
                st.table(report.grammar_issues)
            else:
                st.success("No grammar issues detected.")

        # ---------------------------
        # 📋 Message Deviations
        # ---------------------------
        with tabs[2]:
            st.markdown("## 📋 Message Deviations")

            if report.message_deviations:
                st.table(report.message_deviations)
            else:
                st.success("No message deviations found.")

        # ---------------------------
        # 🖼 Visual + Bonus
        # ---------------------------
        with tabs[3]:
            st.markdown("## 🖼 Visual Consistency & Bonus Features")

            with st.expander("🔍 Visual Consistency Details"):
                if __name__ == '__main__':
                    # example_xlsx = 'ARS00118_DE__Global_Matrix_v36.xlsx'
                    # example_eml = '[ERS00925_Sep_MNL-MainEntity=All Other-M1=Young Saver-M2=Prospect_ProAccount-proof] Kristin, grow your retirement savings___.eml'
                #     agent = ParserAgent(llm_enabled=False)
                #     results = agent.run(checklist_file, html_content, base_url=None)
                #     pd.set_option('display.max_rows', None)
                #     pd.set_option('display.max_colwidth', 300)
                #     print("Final results:")
                #     print(results[['row_index','source_column','source_value','match_symbol']])
                st.json({
                    "visual_match_score": report.visual_consistency.get("visual_match_score"),
                    "layout_consistency": report.visual_consistency.get("layout_consistency"),
                    "missing_elements": report.visual_consistency.get("missing_elements"),
                    "discrepancies": report.visual_consistency.get("discrepancies"),
                })

            with st.expander("📗 Readability Metrics"):
                st.json(report.visual_consistency["readability"])

            with st.expander("🚫 Spam Risk"):
                st.json(report.visual_consistency["spam_risk"])

            with st.expander("🔗 Link Validation"):
                st.json(report.visual_consistency["link_validation"])

        # ---------------------------
        # 💡 Recommendations
        # ---------------------------
        with tabs[4]:
            st.markdown("## 💡 Recommendations")
            for i, rec in enumerate(report.recommendations, 1):
                st.markdown(f"**{i}.** {rec}")

        # ---------------------------
        # 📧 Email Preview
        # ---------------------------
        st.markdown("---")
        st.subheader("📧 Email Preview")
        st.components.v1.html(html_content, height=450, scrolling=True)



# ======================================================================
# 4. ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()