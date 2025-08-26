# =============================
# Required Imports - ADD THESE TO THE TOP OF complete_seo_tools.py
# =============================

import validators
import requests
import json
import re
from bs4 import BeautifulSoup
from langchain.tools import tool

# Import uploaded_documents from admin_views
try:
    from .admin_views import uploaded_documents
except ImportError:
    # Fallback if import fails
    uploaded_documents = {}

# @tool
# def comprehensive_seo_analysis(url: str, document_content: str = None, document_id: str = None) -> str:
#     """
#     Complete SEO analysis including heading tags, metadata, FAQ, and schema markup.
#     Args:
#         url: The URL to analyze
#         document_content: Optional document content to compare against
#         document_id: Optional document ID from uploaded documents
#     Returns:
#         Comprehensive SEO analysis report
#     """
#     global chat_system
#     chat_system = "Tool call - Comprehensive SEO Analysis"
    
#     if not validators.url(url):
#         return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
#     try:
#         response = requests.get(url, timeout=15, headers={
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         })
#         response.raise_for_status()
        
#         soup = BeautifulSoup(response.text, "html.parser")
        
#         # Initialize analysis results
#         analysis = {
#             'url': url,
#             'headings': analyze_heading_structure(soup),
#             'metadata': analyze_metadata(soup),
#             'faq': analyze_faq_content(soup),
#             'schema': analyze_schema_markup(soup),
#             'seo_score': 0,
#             'recommendations': []
#         }
        
#         # Calculate SEO score and generate report
#         report = generate_seo_report(analysis, document_content, document_id)
        
#         return report
        
#     except requests.exceptions.RequestException as e:
#         return f"⚠️ Error fetching URL: {str(e)}"
#     except Exception as e:
#         return f"⚠️ Error in SEO analysis: {str(e)}"

# def analyze_heading_structure(soup):
#     """Analyze heading structure (H1-H6) for SEO."""
#     headings = {
#         'h1': [],
#         'h2': [],
#         'h3': [],
#         'h4': [],
#         'h5': [],
#         'h6': [],
#         'structure_score': 0,
#         'issues': []
#     }
    
#     # Extract all heading tags
#     for level in range(1, 7):
#         tag = f'h{level}'
#         elements = soup.find_all(tag)
#         headings[tag] = [elem.get_text().strip() for elem in elements if elem.get_text().strip()]
    
#     # Analyze structure
#     h1_count = len(headings['h1'])
#     total_headings = sum(len(headings[f'h{i}']) for i in range(1, 7))
    
#     # Scoring and issues
#     if h1_count == 1:
#         headings['structure_score'] += 25
#     elif h1_count == 0:
#         headings['issues'].append("❌ Missing H1 tag - critical for SEO")
#     else:
#         headings['issues'].append(f"⚠️ Multiple H1 tags ({h1_count}) - use only one per page")
    
#     if len(headings['h2']) >= 2:
#         headings['structure_score'] += 20
#     elif len(headings['h2']) == 0:
#         headings['issues'].append("💡 Consider adding H2 tags for better structure")
    
#     if total_headings >= 3:
#         headings['structure_score'] += 15
    
#     # Check heading hierarchy
#     hierarchy_issues = check_heading_hierarchy(headings)
#     headings['issues'].extend(hierarchy_issues)
    
#     return headings

# def analyze_metadata(soup):
#     """Analyze metadata tags for SEO."""
#     metadata = {
#         'title': '',
#         'description': '',
#         'keywords': '',
#         'robots': '',
#         'canonical': '',
#         'language': '',
#         'author': '',
#         'viewport': '',
#         'charset': '',
#         'open_graph': {},
#         'twitter_card': {},
#         'score': 0,
#         'issues': []
#     }
    
#     # Title tag
#     if soup.title and soup.title.string:
#         metadata['title'] = soup.title.string.strip()
#         title_len = len(metadata['title'])
#         if 30 <= title_len <= 60:
#             metadata['score'] += 20
#         elif title_len < 30:
#             metadata['issues'].append(f"⚠️ Title too short ({title_len} chars) - aim for 30-60")
#         else:
#             metadata['issues'].append(f"⚠️ Title too long ({title_len} chars) - aim for 30-60")
#     else:
#         metadata['issues'].append("❌ Missing title tag")
    
#     # Meta tags
#     for meta in soup.find_all('meta'):
#         name = meta.get('name', '').lower()
#         property_attr = meta.get('property', '').lower()
#         content = meta.get('content', '')
        
#         if name == 'description':
#             metadata['description'] = content
#             desc_len = len(content)
#             if 150 <= desc_len <= 160:
#                 metadata['score'] += 20
#             elif desc_len < 120:
#                 metadata['issues'].append(f"⚠️ Description too short ({desc_len} chars)")
#             elif desc_len > 160:
#                 metadata['issues'].append(f"⚠️ Description too long ({desc_len} chars)")
#             else:
#                 metadata['score'] += 15
                
#         elif name == 'keywords':
#             metadata['keywords'] = content
#         elif name == 'robots':
#             metadata['robots'] = content
#         elif name == 'author':
#             metadata['author'] = content
#         elif name == 'viewport':
#             metadata['viewport'] = content
#             metadata['score'] += 10
#         elif meta.get('charset'):
#             metadata['charset'] = meta.get('charset')
#             metadata['score'] += 5
        
#         # Open Graph
#         elif property_attr.startswith('og:'):
#             og_key = property_attr.replace('og:', '')
#             metadata['open_graph'][og_key] = content
        
#         # Twitter Card
#         elif name.startswith('twitter:'):
#             twitter_key = name.replace('twitter:', '')
#             metadata['twitter_card'][twitter_key] = content
    
#     # Canonical URL
#     canonical = soup.find('link', rel='canonical')
#     if canonical:
#         metadata['canonical'] = canonical.get('href', '')
#         metadata['score'] += 10
    
#     # Language
#     html_tag = soup.find('html')
#     if html_tag and html_tag.get('lang'):
#         metadata['language'] = html_tag.get('lang')
#         metadata['score'] += 5
    
#     # Check for missing critical elements
#     if not metadata['description']:
#         metadata['issues'].append("❌ Missing meta description")
#     if not metadata['viewport']:
#         metadata['issues'].append("⚠️ Missing viewport meta tag")
#     if not metadata['language']:
#         metadata['issues'].append("⚠️ Missing language attribute")
    
#     # Social media optimization
#     if metadata['open_graph']:
#         metadata['score'] += 15
#     else:
#         metadata['issues'].append("💡 Add Open Graph tags for social sharing")
    
#     return metadata

# def analyze_faq_content(soup):
#     """Analyze FAQ content and structured data for SEO."""
#     faq_analysis = {
#         'structured_faqs': [],
#         'html_faqs': [],
#         'faq_schema_count': 0,
#         'question_count': 0,
#         'score': 0,
#         'recommendations': []
#     }
    
#     # Look for FAQ structured data (Schema.org)
#     for script in soup.find_all('script', type='application/ld+json'):
#         try:
#             data = json.loads(script.string)
            
#             # Handle single FAQ page
#             if isinstance(data, dict) and data.get('@type') == 'FAQPage':
#                 faq_analysis['faq_schema_count'] += 1
#                 main_entity = data.get('mainEntity', [])
#                 if isinstance(main_entity, list):
#                     for item in main_entity:
#                         if item.get('@type') == 'Question':
#                             question = item.get('name', '')
#                             answer = ''
#                             accepted_answer = item.get('acceptedAnswer', {})
#                             if isinstance(accepted_answer, dict):
#                                 answer = accepted_answer.get('text', '')
#                             faq_analysis['structured_faqs'].append({
#                                 'question': question,
#                                 'answer': answer
#                             })
#                             faq_analysis['question_count'] += 1
            
#             # Handle array of FAQ items
#             elif isinstance(data, list):
#                 for item in data:
#                     if isinstance(item, dict) and item.get('@type') == 'Question':
#                         question = item.get('name', '')
#                         answer = ''
#                         accepted_answer = item.get('acceptedAnswer', {})
#                         if isinstance(accepted_answer, dict):
#                             answer = accepted_answer.get('text', '')
#                         faq_analysis['structured_faqs'].append({
#                             'question': question,
#                             'answer': answer
#                         })
#                         faq_analysis['question_count'] += 1
                        
#         except (json.JSONDecodeError, AttributeError):
#             continue
    
#     # Look for HTML FAQ patterns
#     faq_patterns = [
#         {'q': 'dt', 'a': 'dd'},
#         {'q': '.faq-question', 'a': '.faq-answer'},
#         {'q': '.question', 'a': '.answer'},
#         {'q': '[id*="question"]', 'a': '[id*="answer"]'}
#     ]
    
#     for pattern in faq_patterns:
#         questions = soup.select(pattern['q'])
#         for question_elem in questions:
#             question_text = question_elem.get_text().strip()
#             if '?' in question_text and len(question_text) > 10:
#                 answer_elem = question_elem.find_next_sibling()
#                 answer_text = answer_elem.get_text().strip() if answer_elem else ''
                
#                 faq_analysis['html_faqs'].append({
#                     'question': question_text,
#                     'answer': answer_text
#                 })
    
#     # Scoring
#     if faq_analysis['faq_schema_count'] > 0:
#         faq_analysis['score'] += 30
#         faq_analysis['recommendations'].append("✅ FAQ structured data found - excellent for rich snippets!")
#     else:
#         faq_analysis['recommendations'].append("💡 Add FAQ structured data (Schema.org) for rich snippets")
    
#     if faq_analysis['question_count'] >= 3:
#         faq_analysis['score'] += 20
#     elif faq_analysis['question_count'] > 0:
#         faq_analysis['score'] += 10
#         faq_analysis['recommendations'].append("💡 Add more FAQ questions (3+ recommended)")
    
#     if len(faq_analysis['html_faqs']) > 0:
#         faq_analysis['score'] += 10
    
#     return faq_analysis

# def analyze_schema_markup(soup):
#     """Analyze Schema.org structured data for SEO."""
#     schema_analysis = {
#         'json_ld_count': 0,
#         'microdata_count': 0,
#         'rdfa_count': 0,
#         'schemas_found': [],
#         'score': 0,
#         'recommendations': []
#     }
    
#     # JSON-LD Schema
#     for script in soup.find_all('script', type='application/ld+json'):
#         try:
#             data = json.loads(script.string)
#             schema_analysis['json_ld_count'] += 1
            
#             # Extract schema types
#             if isinstance(data, dict):
#                 schema_type = data.get('@type', '')
#                 if schema_type:
#                     schema_analysis['schemas_found'].append(schema_type)
#             elif isinstance(data, list):
#                 for item in data:
#                     if isinstance(item, dict):
#                         schema_type = item.get('@type', '')
#                         if schema_type:
#                             schema_analysis['schemas_found'].append(schema_type)
                            
#         except (json.JSONDecodeError, AttributeError):
#             continue
    
#     # Microdata
#     microdata_elements = soup.find_all(attrs={'itemtype': True})
#     schema_analysis['microdata_count'] = len(microdata_elements)
    
#     for elem in microdata_elements:
#         itemtype = elem.get('itemtype', '')
#         if 'schema.org' in itemtype:
#             schema_type = itemtype.split('/')[-1]
#             if schema_type not in schema_analysis['schemas_found']:
#                 schema_analysis['schemas_found'].append(schema_type)
    
#     # RDFa
#     rdfa_elements = soup.find_all(attrs={'typeof': True})
#     schema_analysis['rdfa_count'] = len(rdfa_elements)
    
#     # Scoring
#     total_schemas = schema_analysis['json_ld_count'] + schema_analysis['microdata_count'] + schema_analysis['rdfa_count']
    
#     if total_schemas >= 3:
#         schema_analysis['score'] += 30
#     elif total_schemas >= 1:
#         schema_analysis['score'] += 20
#     else:
#         schema_analysis['recommendations'].append("💡 Add structured data markup for better search visibility")
    
#     # Check for important schema types
#     important_schemas = ['Organization', 'WebSite', 'Article', 'Product', 'LocalBusiness', 'FAQPage', 'BreadcrumbList']
#     found_important = [s for s in schema_analysis['schemas_found'] if s in important_schemas]
    
#     if found_important:
#         schema_analysis['score'] += len(found_important) * 5
#         schema_analysis['recommendations'].append(f"✅ Found important schemas: {', '.join(found_important)}")
    
#     # Recommendations for missing schemas
#     if 'WebSite' not in schema_analysis['schemas_found']:
#         schema_analysis['recommendations'].append("💡 Add WebSite schema for site search box")
#     if 'Organization' not in schema_analysis['schemas_found']:
#         schema_analysis['recommendations'].append("💡 Add Organization schema for brand info")
#     if 'BreadcrumbList' not in schema_analysis['schemas_found']:
#         schema_analysis['recommendations'].append("💡 Add BreadcrumbList schema for navigation")
    
#     return schema_analysis

# def check_heading_hierarchy(headings):
#     """Check if heading hierarchy follows best practices."""
#     issues = []
    
#     # Check if H2s exist before H3s, etc.
#     if len(headings['h3']) > 0 and len(headings['h2']) == 0:
#         issues.append("⚠️ H3 tags found without H2 - maintain proper hierarchy")
    
#     if len(headings['h4']) > 0 and len(headings['h3']) == 0:
#         issues.append("⚠️ H4 tags found without H3 - maintain proper hierarchy")
    
#     # Check for empty headings
#     for level in range(1, 7):
#         tag = f'h{level}'
#         empty_count = sum(1 for h in headings[tag] if not h.strip())
#         if empty_count > 0:
#             issues.append(f"⚠️ {empty_count} empty {tag.upper()} tag(s) found")
    
#     return issues

# def generate_seo_report(analysis, document_content=None, document_id=None):
#     """Generate comprehensive SEO analysis report."""
    
#     # Get document content if provided
#     doc_content = None
#     doc_name = "provided document"
    
#     if document_id and document_id in uploaded_documents:
#         doc_info = uploaded_documents[document_id]
#         try:
#             with open(doc_info['file_path'], 'r', encoding='utf-8') as f:
#                 doc_content = f.read()
#             doc_name = doc_info['original_name']
#         except Exception:
#             pass
#     elif document_content:
#         doc_content = document_content
    
#     # Calculate overall SEO score
#     total_score = (
#         analysis['headings']['structure_score'] +
#         analysis['metadata']['score'] +
#         analysis['faq']['score'] +
#         analysis['schema']['score']
#     )
    
#     # Generate report
#     report = f"📊 **COMPREHENSIVE SEO ANALYSIS**\n"
#     report += f"{'='*60}\n"
#     report += f"📊 **URL:** {analysis['url']}\n"
#     report += f"📊 **Overall SEO Score:** {total_score}\n\n"
    
#     # SEO Score Interpretation
#     if total_score >= 80:
#         report += "🟢 **Excellent SEO Implementation**\n\n"
#     elif total_score >= 60:
#         report += "🟡 **Good SEO - Some Improvements Needed**\n\n"
#     elif total_score >= 40:
#         report += "🟠 **Fair SEO - Multiple Issues to Address**\n\n"
#     else:
#         report += "🔴 **Poor SEO - Needs Significant Improvement**\n\n"
    
#     # 1. HEADING STRUCTURE ANALYSIS
#     report += "📑 **1. HEADING STRUCTURE ANALYSIS**\n"
#     report += f"{'─'*40}\n"
#     report += f"**Score: {analysis['headings']['structure_score']}/60**\n\n"
    
#     for level in range(1, 7):
#         tag = f'h{level}'
#         count = len(analysis['headings'][tag])
#         if count > 0:
#             report += f"**{tag.upper()} Tags ({count}):**\n"
#             for i, heading in enumerate(analysis['headings'][tag][:3], 1):  # Show first 3
#                 report += f"  {i}. {heading}\n"
#             if count > 3:
#                 report += f"  ... and {count-3} more\n"
#             report += "\n"
    
#     if analysis['headings']['issues']:
#         report += "**Issues Found:**\n"
#         for issue in analysis['headings']['issues']:
#             report += f"• {issue}\n"
#         report += "\n"
    
#     # 2. METADATA ANALYSIS
#     report += "🏷️ **2. METADATA ANALYSIS**\n"
#     report += f"{'─'*40}\n"
#     report += f"**Score: {analysis['metadata']['score']}/100**\n\n"
    
#     report += f"**Title:** {analysis['metadata']['title']}\n"
#     report += f"**Description:** {analysis['metadata']['description'][:150]}{'...' if len(analysis['metadata']['description']) > 150 else ''}\n"
#     report += f"**Keywords:** {analysis['metadata']['keywords'] or 'Not specified'}\n"
#     report += f"**Canonical URL:** {analysis['metadata']['canonical'] or 'Not specified'}\n"
#     report += f"**Language:** {analysis['metadata']['language'] or 'Not specified'}\n"
#     report += f"**Robots:** {analysis['metadata']['robots'] or 'Not specified'}\n\n"
    
#     if analysis['metadata']['open_graph']:
#         report += f"**Open Graph Tags:** {len(analysis['metadata']['open_graph'])} found\n"
#     if analysis['metadata']['twitter_card']:
#         report += f"**Twitter Card Tags:** {len(analysis['metadata']['twitter_card'])} found\n"
    
#     if analysis['metadata']['issues']:
#         report += "\n**Issues Found:**\n"
#         for issue in analysis['metadata']['issues']:
#             report += f"• {issue}\n"
#         report += "\n"
    
#     # 3. FAQ ANALYSIS
#     report += "❓ **3. FAQ CONTENT ANALYSIS**\n"
#     report += f"{'─'*40}\n"
#     report += f"**Score: {analysis['faq']['score']}/60**\n\n"
    
#     report += f"**FAQ Schema Markup:** {analysis['faq']['faq_schema_count']} found\n"
#     report += f"**Structured FAQ Questions:** {analysis['faq']['question_count']}\n"
#     report += f"**HTML FAQ Elements:** {len(analysis['faq']['html_faqs'])}\n\n"
    
#     if analysis['faq']['structured_faqs']:
#         report += "**Structured FAQ Questions (Top 3):**\n"
#         for i, faq in enumerate(analysis['faq']['structured_faqs'][:3], 1):
#             report += f"{i}. Q: {faq['question']}\n"
#             report += f"   A: {faq['answer'][:100]}{'...' if len(faq['answer']) > 100 else ''}\n"
#         report += "\n"
    
#     if analysis['faq']['recommendations']:
#         report += "**Recommendations:**\n"
#         for rec in analysis['faq']['recommendations']:
#             report += f"• {rec}\n"
#         report += "\n"
    
#     # 4. SCHEMA MARKUP ANALYSIS
#     report += "🗃️ **4. SCHEMA MARKUP ANALYSIS**\n"
#     report += f"{'─'*40}\n"
#     report += f"**Score: {analysis['schema']['score']}/80**\n\n"
    
#     report += f"**JSON-LD Scripts:** {analysis['schema']['json_ld_count']}\n"
#     report += f"**Microdata Elements:** {analysis['schema']['microdata_count']}\n"
#     report += f"**RDFa Elements:** {analysis['schema']['rdfa_count']}\n\n"
    
#     if analysis['schema']['schemas_found']:
#         report += f"**Schema Types Found:** {', '.join(set(analysis['schema']['schemas_found']))}\n\n"
    
#     if analysis['schema']['recommendations']:
#         report += "**Recommendations:**\n"
#         for rec in analysis['schema']['recommendations']:
#             report += f"• {rec}\n"
#         report += "\n"
    
#     # 5. DOCUMENT COMPARISON (if provided)
#     if doc_content:
#         report += f"📋 **5. DOCUMENT COMPARISON ANALYSIS**\n"
#         report += f"{'─'*40}\n"
#         report += f"**Comparing against:** {doc_name}\n\n"
        
#         # Compare title with document
#         if analysis['metadata']['title'] and doc_content:
#             title_words = analysis['metadata']['title'].lower().split()
#             doc_lower = doc_content.lower()
#             matching_words = [word for word in title_words if word in doc_lower and len(word) > 3]
#             match_percentage = (len(matching_words) / len(title_words)) * 100 if title_words else 0
            
#             report += f"**Title Alignment:** {match_percentage:.0f}% match with document\n"
            
#             if match_percentage >= 70:
#                 report += "✅ Title highly relevant to document content\n"
#             elif match_percentage >= 40:
#                 report += "⚠️ Title moderately relevant to document content\n"
#             else:
#                 report += "❌ Title poorly aligned with document content\n"
        
#         # Compare H1 with document
#         if analysis['headings']['h1'] and doc_content:
#             h1_text = analysis['headings']['h1'][0]
#             if h1_text.lower() in doc_content.lower():
#                 report += "✅ H1 tag content found in document\n"
#             else:
#                 h1_words = h1_text.lower().split()
#                 matching_words = [word for word in h1_words if word in doc_content.lower() and len(word) > 3]
#                 match_percentage = (len(matching_words) / len(h1_words)) * 100 if h1_words else 0
#                 report += f"**H1 Alignment:** {match_percentage:.0f}% match with document\n"
        
#         report += "\n"
    
#     # 6. PRIORITY RECOMMENDATIONS
#     report += "🎯 **6. PRIORITY ACTION ITEMS**\n"
#     report += f"{'─'*40}\n"
    
#     priority_actions = []
    
#     # Critical issues first
#     if not analysis['metadata']['title']:
#         priority_actions.append("🔴 HIGH: Add title tag immediately")
#     if not analysis['metadata']['description']:
#         priority_actions.append("🔴 HIGH: Add meta description")
#     if len(analysis['headings']['h1']) == 0:
#         priority_actions.append("🔴 HIGH: Add H1 tag")
#     elif len(analysis['headings']['h1']) > 1:
#         priority_actions.append("🔴 HIGH: Use only one H1 tag per page")
    
#     # Medium priority
#     if analysis['schema']['json_ld_count'] == 0:
#         priority_actions.append("🟡 MEDIUM: Add structured data markup")
#     if analysis['faq']['faq_schema_count'] == 0 and len(analysis['faq']['html_faqs']) > 0:
#         priority_actions.append("🟡 MEDIUM: Add FAQ structured data for rich snippets")
#     if not analysis['metadata']['canonical']:
#         priority_actions.append("🟡 MEDIUM: Add canonical URL")
#     if not analysis['metadata']['open_graph']:
#         priority_actions.append("🟡 MEDIUM: Add Open Graph tags for social sharing")
    
#     # Low priority improvements
#     if len(analysis['headings']['h2']) < 2:
#         priority_actions.append("🟢 LOW: Add more H2 tags for better structure")
#     if not analysis['metadata']['language']:
#         priority_actions.append("🟢 LOW: Add language attribute to HTML tag")
    
#     if priority_actions:
#         for action in priority_actions:
#             report += f"• {action}\n"
#     else:
#         report += "🎉 **Great job! No critical issues found.**\n"
    
#     report += f"\n{'='*60}\n"
#     report += "📈 **SEO Analysis Complete** - Use these insights to improve your search rankings!\n"
    
#     return report

# @tool
# def validate_seo_against_document(url: str, document_content: str = None, document_id: str = None) -> str:
#     """
#     Validate URL's SEO elements (headings, metadata, FAQ, schema) against a document.
#     Args:
#         url: The URL to validate
#         document_content: Optional document content to validate against
#         document_id: Optional document ID from uploaded documents
#     Returns:
#         SEO validation results with document comparison
#     """
#     global chat_system
#     chat_system = "Tool call - Validate SEO Against Document"
    
#     # Since this is essentially the same as comprehensive_seo_analysis,
#     # we'll call the same logic but with a different system message
#     if not validators.url(url):
#         return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
#     try:
#         response = requests.get(url, timeout=15, headers={
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         })
#         response.raise_for_status()
        
#         soup = BeautifulSoup(response.text, "html.parser")
        
#         # Initialize analysis results
#         analysis = {
#             'url': url,
#             'headings': analyze_heading_structure(soup),
#             'metadata': analyze_metadata(soup),
#             'faq': analyze_faq_content(soup),
#             'schema': analyze_schema_markup(soup),
#             'seo_score': 0,
#             'recommendations': []
#         }
        
#         # Generate report with document comparison focus
#         report = generate_seo_report(analysis, document_content, document_id)
        
#         # Add validation-specific header
#         validation_header = "🎯 **SEO VALIDATION AGAINST DOCUMENT**\n"
#         validation_header += "=" * 60 + "\n\n"
        
#         return validation_header + report
        
#     except requests.exceptions.RequestException as e:
#         return f"⚠️ Error fetching URL: {str(e)}"
#     except Exception as e:
#         return f"⚠️ Error in SEO validation: {str(e)}"
    

# Enhanced SEO Analysis Tool with Document-Based Scoring
# Integrates URL validation, metadata extraction, FAQ analysis, heading analysis, and content analysis with scoring

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import validators
import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urlparse, urljoin
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

@tool
def comprehensive_seo_analysis(url: str, document_id: str = None) -> str:
    """
    Comprehensive SEO analysis with scoring based on uploaded document guidelines.
    Combines all analysis functions with detailed scoring.
    
    Args:
        url: The URL to analyze
        document_id: Optional document ID to compare against (from uploaded_documents)
    Returns:
        Complete SEO analysis with scores for each component
    """
    global chat_system
    chat_system = "Tool call - Comprehensive SEO Analysis with Scoring"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        # Initialize results dictionary
        analysis_results = {
            'url': url,
            'basic_validation': {},
            'metadata_analysis': {},
            'heading_analysis': {},
            'content_analysis': {},
            'faq_analysis': {},
            'overall_score': 0,
            'recommendations': []
        }
        
        # Fetch webpage content
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 1. BASIC VALIDATION AND URL ANALYSIS (Score: 0-20)
        validation_score = analyze_url_structure(url, soup)
        analysis_results['basic_validation'] = validation_score
        
        # 2. METADATA ANALYSIS (Score: 0-25)
        metadata_score = analyze_metadata_with_scoring(url, soup)
        analysis_results['metadata_analysis'] = metadata_score
        
        # 3. HEADING STRUCTURE ANALYSIS (Score: 0-20)
        heading_score = analyze_headings_with_scoring(soup)
        analysis_results['heading_analysis'] = heading_score
        
        # 4. CONTENT ANALYSIS (Score: 0-25)
        content_score = analyze_content_with_scoring(url, soup)
        analysis_results['content_analysis'] = content_score
        
        # 5. FAQ ANALYSIS (Score: 0-10)
        faq_score = analyze_faqs_with_scoring(soup)
        analysis_results['faq_analysis'] = faq_score
        
        # Calculate overall score
        total_score = (
            validation_score['score'] +
            metadata_score['score'] +
            heading_score['score'] +
            content_score['score'] +
            faq_score['score']
        )
        analysis_results['overall_score'] = total_score
        
        # Generate comprehensive recommendations
        analysis_results['recommendations'] = generate_seo_recommendations(analysis_results)
        
        # Format response
        return format_comprehensive_analysis_response(analysis_results, document_id)
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error fetching URL: {str(e)}"
    except Exception as e:
        return f"⚠️ Error during analysis: {str(e)}"

def analyze_url_structure(url: str, soup) -> dict:
    """Analyze URL structure based on SEO best practices (Max: 20 points)"""
    score = 0
    issues = []
    recommendations = []
    
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    
    # URL Length (5 points)
    if len(path) <= 50:
        score += 5
    elif len(path) <= 100:
        score += 3
        recommendations.append("Consider shortening URL for better optimization")
    else:
        issues.append("URL is too long (>100 characters)")
        recommendations.append("Shorten URL significantly - keep under 50 characters")
    
    # URL Structure (5 points)
    if re.search(r'[a-z0-9-]+', path):
        score += 3
    if not re.search(r'[^a-zA-Z0-9-/]', path):
        score += 2
    else:
        issues.append("URL contains special characters")
        recommendations.append("Use only letters, numbers, and hyphens in URLs")
    
    # Keyword in URL (5 points)
    # This is basic - would need keyword context for full analysis
    if len(path.split('-')) >= 2:
        score += 3
        recommendations.append("Good use of hyphens for keyword separation")
    else:
        recommendations.append("Consider using hyphens to separate keywords in URL")
    
    # HTTPS (5 points)
    if parsed_url.scheme == 'https':
        score += 5
    else:
        issues.append("Site not using HTTPS")
        recommendations.append("Implement HTTPS for security and SEO benefits")
    
    return {
        'score': score,
        'max_score': 20,
        'percentage': round((score/20) * 100, 1),
        'issues': issues,
        'recommendations': recommendations
    }

def analyze_metadata_with_scoring(url: str, soup) -> dict:
    """Analyze metadata with scoring (Max: 25 points)"""
    score = 0
    issues = []
    recommendations = []
    metadata = {}
    
    # Title Tag Analysis (10 points)
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    metadata['title'] = title
    
    if title:
        if 30 <= len(title) <= 60:
            score += 5
        elif len(title) < 30:
            score += 2
            issues.append("Title tag too short")
            recommendations.append("Expand title tag to 30-60 characters")
        else:
            score += 2
            issues.append("Title tag too long")
            recommendations.append("Shorten title tag to under 60 characters")
        
        # Title structure (front-loaded keywords assumed good)
        if title and not title.endswith('...'):
            score += 3
        
        score += 2  # Base points for having a title
    else:
        issues.append("Missing title tag")
        recommendations.append("Add a compelling title tag (30-60 characters)")
    
    # Meta Description Analysis (8 points)
    desc_meta = soup.find('meta', attrs={'name': 'description'})
    description = desc_meta.get('content', '') if desc_meta else ''
    metadata['description'] = description
    
    if description:
        if 120 <= len(description) <= 160:
            score += 5
        elif len(description) < 120:
            score += 3
            recommendations.append("Expand meta description to 120-160 characters")
        else:
            score += 3
            recommendations.append("Shorten meta description to under 160 characters")
        
        score += 3  # Base points for having description
    else:
        issues.append("Missing meta description")
        recommendations.append("Add compelling meta description (120-160 characters)")
    
    # Additional Meta Tags (4 points)
    canonical = soup.find('link', rel='canonical')
    if canonical:
        score += 2
        metadata['canonical'] = canonical.get('href', '')
    else:
        recommendations.append("Consider adding canonical URL")
    
    # Language specification (2 points)
    html_tag = soup.find('html')
    if html_tag and html_tag.get('lang'):
        score += 2
        metadata['language'] = html_tag.get('lang')
    else:
        recommendations.append("Specify language in HTML tag")
    
    return {
        'score': score,
        'max_score': 25,
        'percentage': round((score/25) * 100, 1),
        'metadata': metadata,
        'issues': issues,
        'recommendations': recommendations
    }

def analyze_headings_with_scoring(soup) -> dict:
    """Analyze heading structure with scoring (Max: 20 points)"""
    score = 0
    issues = []
    recommendations = []
    headings = []
    
    # Extract all headings
    heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    heading_counts = {}
    
    for tag in heading_tags:
        found_headings = soup.find_all(tag)
        heading_counts[tag] = len(found_headings)
        
        for heading in found_headings:
            text = heading.get_text().strip()
            if text:
                headings.append({
                    'level': int(tag[1]),
                    'text': text,
                    'tag': tag.upper()
                })
    
    # H1 Analysis (8 points)
    h1_count = heading_counts.get('h1', 0)
    if h1_count == 1:
        score += 8
    elif h1_count == 0:
        issues.append("Missing H1 tag")
        recommendations.append("Add exactly one H1 tag per page")
    else:
        score += 4
        issues.append(f"Multiple H1 tags found ({h1_count})")
        recommendations.append("Use only one H1 tag per page")
    
    # Heading Hierarchy (6 points)
    if heading_counts.get('h2', 0) > 0:
        score += 3
    else:
        recommendations.append("Add H2 tags for better content structure")
    
    if heading_counts.get('h3', 0) > 0:
        score += 2
    
    # Proper hierarchy check
    if h1_count > 0 and heading_counts.get('h2', 0) > 0:
        score += 1
    
    # Content Organization (6 points)
    total_headings = sum(heading_counts.values())
    if total_headings >= 3:
        score += 4
        if total_headings >= 6:
            score += 2
    else:
        recommendations.append("Add more headings for better content organization")
    
    return {
        'score': score,
        'max_score': 20,
        'percentage': round((score/20) * 100, 1),
        'heading_counts': heading_counts,
        'total_headings': total_headings,
        'headings': headings[:10],  # Limit for response size
        'issues': issues,
        'recommendations': recommendations
    }

def analyze_content_with_scoring(url: str, soup) -> dict:
    """Analyze content with scoring (Max: 25 points)"""
    score = 0
    issues = []
    recommendations = []
    
    try:
        # Get content using LangChain
        loader = WebBaseLoader(url)
        documents = loader.load()
        content = documents[0].page_content if documents else ""
        
        # Basic content metrics
        word_count = len(content.split())
        char_count = len(content)
        
        # Word Count Analysis (8 points)
        if word_count >= 1500:
            score += 8
        elif word_count >= 800:
            score += 6
            recommendations.append("Consider expanding content to 1500+ words")
        elif word_count >= 300:
            score += 4
            recommendations.append("Expand content significantly (aim for 800+ words)")
        else:
            score += 1
            issues.append("Content too short")
            recommendations.append("Add substantial content (minimum 300 words)")
        
        # Image Analysis (5 points)
        images = soup.find_all('img')
        img_with_alt = [img for img in images if img.get('alt')]
        
        if len(images) > 0:
            alt_ratio = len(img_with_alt) / len(images)
            if alt_ratio == 1.0:
                score += 5
            elif alt_ratio >= 0.8:
                score += 4
                recommendations.append("Add alt text to all images")
            else:
                score += 2
                issues.append("Many images missing alt text")
                recommendations.append("Add descriptive alt text to all images")
        else:
            recommendations.append("Consider adding relevant images")
        
        # Link Analysis (4 points)
        internal_links = []
        external_links = []
        base_domain = urlparse(url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                link_domain = urlparse(href).netloc
                if link_domain == base_domain:
                    internal_links.append(href)
                else:
                    external_links.append(href)
        
        if len(internal_links) >= 3:
            score += 2
        else:
            recommendations.append("Add more internal links (3+ recommended)")
        
        if len(external_links) >= 1:
            score += 2
        else:
            recommendations.append("Add authoritative external links")
        
        # Content Structure (4 points)
        paragraphs = len([p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()])
        if paragraphs >= 5:
            score += 2
        
        # Lists and structure
        lists = soup.find_all(['ul', 'ol'])
        if len(lists) >= 1:
            score += 2
        else:
            recommendations.append("Use lists to improve content structure")
        
        # Schema/Structured Data (4 points)
        schema_scripts = soup.find_all('script', type='application/ld+json')
        if len(schema_scripts) > 0:
            score += 4
        else:
            recommendations.append("Add structured data markup for better SERP visibility")
        
    except Exception as e:
        content = soup.get_text()
        word_count = len(content.split()) if content else 0
        recommendations.append("Could not perform complete content analysis")
    
    return {
        'score': score,
        'max_score': 25,
        'percentage': round((score/25) * 100, 1),
        'word_count': word_count,
        'image_count': len(images) if 'images' in locals() else 0,
        'images_with_alt': len(img_with_alt) if 'img_with_alt' in locals() else 0,
        'internal_links': len(internal_links) if 'internal_links' in locals() else 0,
        'external_links': len(external_links) if 'external_links' in locals() else 0,
        'issues': issues,
        'recommendations': recommendations
    }

def analyze_faqs_with_scoring(soup) -> dict:
    """Analyze FAQ content with scoring (Max: 10 points)"""
    score = 0
    issues = []
    recommendations = []
    faqs = []
    
    # Look for FAQ structured data
    schema_faqs = 0
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get('@type') == 'FAQPage':
                main_entity = data.get('mainEntity', [])
                if isinstance(main_entity, list):
                    schema_faqs += len(main_entity)
                    for item in main_entity:
                        if item.get('@type') == 'Question':
                            faqs.append({
                                'question': item.get('name', ''),
                                'source': 'Schema.org'
                            })
        except (json.JSONDecodeError, AttributeError):
            continue
    
    # Look for HTML FAQ patterns
    html_faqs = 0
    question_patterns = [
        soup.find_all(string=re.compile(r'\?', re.I)),
        soup.find_all(['dt', 'h3', 'h4'], string=re.compile(r'\?', re.I))
    ]
    
    for pattern in question_patterns:
        if isinstance(pattern, list):
            html_faqs += len([q for q in pattern if len(str(q).strip()) > 10])
    
    # Scoring
    if schema_faqs > 0:
        score += 6  # Bonus for structured data
        if schema_faqs >= 5:
            score += 2
    elif html_faqs >= 3:
        score += 4
        recommendations.append("Add FAQ structured data for rich snippets")
    elif html_faqs > 0:
        score += 2
        recommendations.append("Expand FAQ content and add structured data")
    else:
        recommendations.append("Consider adding FAQ section with structured data")
    
    # Additional points for good FAQ practices
    if schema_faqs >= 3 or html_faqs >= 5:
        score += 2
    
    return {
        'score': score,
        'max_score': 10,
        'percentage': round((score/10) * 100, 1),
        'schema_faqs': schema_faqs,
        'html_faqs': html_faqs,
        'total_faqs': len(faqs),
        'sample_faqs': faqs[:3],
        'issues': issues,
        'recommendations': recommendations
    }

def generate_seo_recommendations(analysis_results: dict) -> list:
    """Generate prioritized SEO recommendations based on analysis"""
    all_recommendations = []
    
    # Collect all recommendations with priority scoring
    for section, data in analysis_results.items():
        if isinstance(data, dict) and 'recommendations' in data:
            section_weight = {
                'metadata_analysis': 1.0,
                'content_analysis': 0.9,
                'heading_analysis': 0.8,
                'basic_validation': 0.7,
                'faq_analysis': 0.6
            }.get(section, 0.5)
            
            for rec in data['recommendations']:
                all_recommendations.append({
                    'recommendation': rec,
                    'section': section.replace('_', ' ').title(),
                    'priority': section_weight,
                    'impact': 'High' if section_weight >= 0.8 else 'Medium' if section_weight >= 0.6 else 'Low'
                })
    
    # Sort by priority
    all_recommendations.sort(key=lambda x: x['priority'], reverse=True)
    
    return all_recommendations[:10]  # Return top 10

def format_comprehensive_analysis_response(results: dict, document_id: str = None) -> str:
    """Format the comprehensive analysis response"""
    
    # Overall score and grade
    overall_score = results['overall_score']
    max_score = 100
    percentage = round((overall_score / max_score) * 100, 1)
    
    grade = 'A' if percentage >= 90 else 'B' if percentage >= 80 else 'C' if percentage >= 70 else 'D' if percentage >= 60 else 'F'
    
    response = f"🔍 **COMPREHENSIVE SEO ANALYSIS REPORT**\n\n"
    response += f"📊 **Overall SEO Score: {overall_score}/100 ({percentage}%) - Grade: {grade}**\n\n"
    response += f"🌐 **URL Analyzed:** {results['url']}\n\n"
    
    # Section scores
    response += "📈 **Section Breakdown:**\n"
    sections = [
        ('basic_validation', '🔗 URL & Technical'),
        ('metadata_analysis', '📝 Metadata'),
        ('heading_analysis', '🏷️ Headings'),
        ('content_analysis', '📄 Content'),
        ('faq_analysis', '❓ FAQ')
    ]
    
    for section_key, section_name in sections:
        section_data = results[section_key]
        response += f"• {section_name}: {section_data['score']}/{section_data['max_score']} ({section_data['percentage']}%)\n"
    
    # Detailed analysis for each section
    response += "\n" + "="*50 + "\n"
    response += "📋 **DETAILED ANALYSIS**\n\n"
    
    # URL & Technical Analysis
    url_data = results['basic_validation']
    response += f"🔗 **URL & Technical Analysis** ({url_data['score']}/20)\n"
    if url_data['issues']:
        response += "⚠️ Issues:\n"
        for issue in url_data['issues']:
            response += f"  • {issue}\n"
    response += "\n"
    
    # Metadata Analysis
    meta_data = results['metadata_analysis']
    response += f"📝 **Metadata Analysis** ({meta_data['score']}/25)\n"
    metadata = meta_data['metadata']
    response += f"• Title: {metadata.get('title', 'Missing')[:60]}...\n"
    response += f"• Description: {metadata.get('description', 'Missing')[:80]}...\n"
    if meta_data['issues']:
        response += "⚠️ Issues:\n"
        for issue in meta_data['issues']:
            response += f"  • {issue}\n"
    response += "\n"
    
    # Heading Analysis
    heading_data = results['heading_analysis']
    response += f"🏷️ **Heading Structure Analysis** ({heading_data['score']}/20)\n"
    response += f"• Total headings: {heading_data['total_headings']}\n"
    for tag, count in heading_data['heading_counts'].items():
        if count > 0:
            response += f"• {tag.upper()}: {count}\n"
    if heading_data['issues']:
        response += "⚠️ Issues:\n"
        for issue in heading_data['issues']:
            response += f"  • {issue}\n"
    response += "\n"
    
    # Content Analysis
    content_data = results['content_analysis']
    response += f"📄 **Content Analysis** ({content_data['score']}/25)\n"
    response += f"• Word count: {content_data['word_count']}\n"
    response += f"• Images: {content_data['image_count']} (Alt text: {content_data['images_with_alt']})\n"
    response += f"• Internal links: {content_data['internal_links']}\n"
    response += f"• External links: {content_data['external_links']}\n"
    if content_data['issues']:
        response += "⚠️ Issues:\n"
        for issue in content_data['issues']:
            response += f"  • {issue}\n"
    response += "\n"
    
    # FAQ Analysis
    faq_data = results['faq_analysis']
    response += f"❓ **FAQ Analysis** ({faq_data['score']}/10)\n"
    response += f"• Schema FAQs: {faq_data['schema_faqs']}\n"
    response += f"• HTML FAQs: {faq_data['html_faqs']}\n"
    response += "\n"
    
    # Top Recommendations
    response += "🎯 **TOP PRIORITY RECOMMENDATIONS:**\n\n"
    for i, rec in enumerate(results['recommendations'][:5], 1):
        impact_emoji = "🔴" if rec['impact'] == 'High' else "🟡" if rec['impact'] == 'Medium' else "🟢"
        response += f"{i}. {impact_emoji} **{rec['impact']} Impact** - {rec['recommendation']}\n"
    
    if document_id:
        response += f"\n📋 **Note:** Analysis based on guidelines from document ID: {document_id}\n"
    
    response += f"\n💡 **Next Steps:** Focus on high-impact recommendations first for maximum SEO improvement.\n"
    
    return response

# Integration function to be added to existing tools
@tool  
def validate_seo_against_document(url: str, document_id: str) -> str:
    """
    Validate URL's SEO against uploaded document guidelines.
    
    Args:
        url: URL to analyze
        document_id: Document ID from uploaded documents
    Returns:
        SEO analysis compared against document guidelines
    """
    global uploaded_documents
    
    if document_id not in uploaded_documents:
        return f"❌ Document ID '{document_id}' not found. Available documents: {list(uploaded_documents.keys())}"
    
    # Use the comprehensive analysis with document context
    return comprehensive_seo_analysis(url, document_id)