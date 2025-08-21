# =============================
# Django & Library Imports
# =============================
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import validators
import requests, traceback
import json, time
import os,re
from dotenv import load_dotenv
import uuid
from django.http import StreamingHttpResponse
from urllib.parse import urljoin, urlparse

# Database models
from .models import ChatSession, ChatMessage

# Parsing HTML
from bs4 import BeautifulSoup

# LangChain & OpenAI integration
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage
from langchain_core.messages import ToolMessage,HumanMessage,AIMessageChunk
from langchain.tools import tool
from openai import OpenAIError, APITimeoutError
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =============================
# Global Variables & Config
# =============================

# Tracks current chat system state
chat_system = ""
messages = []

# Load environment variables for OpenAI API
load_dotenv()
OPENAIKEY = os.getenv('OPEN_AI_KEY')
MODEL = "gpt-3.5-turbo"

# Tracks name validation per session
user_name_status = {}  # session_id -> {'name_validated': bool, 'name': str}

# System message prompt for chatbot behavior
system_message = """You are an expert SEO assistant with name validation and advanced URL analysis capabilities.

    IMPORTANT INSTRUCTIONS:
    1. FIRST INTERACTION: Always start by asking the user for their name in a friendly way.
    2. WHEN USER PROVIDES NAME: Use the validate_name tool to validate and format their name properly.
    3. AFTER NAME VALIDATION: Proceed with your SEO expertise and other capabilities.
    4. IF USER SENDS MESSAGE WITHOUT VALID NAME: Politely ask them to provide their name first.

    Your expertise includes:
    - Keyword research and analysis
    - Content optimization strategies
    - Technical SEO recommendations
    - Meta tag optimization
    - Link building strategies
    - SEO auditing and reporting
    - Search ranking analysis
    - Competitor analysis
    - Document content analysis for SEO insights
    - Comprehensive URL analysis and parsing

    URL Analysis Tools Available:
    - validate_and_fetch_url: Basic URL validation with quick SEO insights
    - extract_metadata: Complete metadata extraction (title, description, Open Graph, etc.)
    - fetch_content: Content extraction and analysis using LangChain
    - extract_headings: Full heading structure analysis (H1-H6)
    - extract_faqs: FAQ and Q&A content extraction
    - extract_main_content: Comprehensive SEO content analysis

    Guidelines:
    1. Always start by collecting the user's name using a warm, friendly approach.
    2. Use the validate_name tool when they provide their name.
    3. After name validation, provide your full SEO services.
    4. Only answer SEO-related questions in detail AFTER name is validated.
    5. When users provide URLs, choose the most appropriate analysis tool based on their needs
    6. For basic validation: use validate_and_fetch_url
    7. For detailed analysis: use extract_metadata, fetch_content, extract_headings, extract_faqs, or extract_main_content
    8. If the user asks a question unrelated to SEO, politely say you specialize in SEO and cannot answer other topics.
    9. If the user clearly asks for a basic arithmetic calculation, call the multiply tool.
    10. You can analyze uploaded documents for SEO-related insights when asked.
    11. Always provide actionable, practical SEO recommendations with clear steps.
    12. If user sends any message before providing a valid name, remind them to share their name first.
    """

# Welcome message prompt for chatbot behavior
welcome_message = """Welcome to your Personal SEO Assistant!

I'm here to help you dominate search rankings and grow your online presence! 

I can assist you with:  
- Keyword Research & Analysis  
- Content Optimization Strategies  
- Technical SEO Recommendations  
- Meta Tag Optimization  
- Link Building Strategies  
- SEO Auditing & Reporting  
- Search Ranking Analysis  
- Competitor Analysis  
- Advanced URL Analysis & Parsing

NEW: Advanced URL Analysis Tools:
- Complete metadata extraction
- Content analysis with LangChain
- Heading structure analysis
- FAQ content extraction
- Comprehensive SEO audits

I can also help with:  
- Basic calculations (just ask me to multiply numbers)  
- URL validation and detailed analysis  
- Document analysis for SEO insights

But first, I'd love to know who I'm talking to. What's your name?  

Once I know, I'll be able to provide:  
- Personalized SEO Strategies  
- Targeted Recommendations  
- A clear roadmap to achieve your SEO goals  

So please tell me your name, and let's start your SEO journey together!"""

# This doesn't look like a name, ask for name first
name_request_message = """🙏 **Please provide your name first!**

I'd love to help you with your SEO needs, but I need to know who I'm talking to first. 

**What's your name?** 

Once you share your name, I'll be able to provide personalized SEO assistance, keyword research, content optimization strategies, and much more! 🚀✨"""


# =============================
# Custom Tool Functions
# =============================
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    Args:
        a: first number
        b: second number
    Returns:
        The product of a and b
    """
    global chat_system
    chat_system = "Tool call - Multiply"
    return a * b

@tool
def extract_metadata(url: str) -> str:
    """Extract comprehensive metadata from a URL including title, description, keywords, and Open Graph data.
    Args:
        url: The URL to extract metadata from
    Returns:
        JSON string containing extracted metadata
    """
    global chat_system
    chat_system = "Tool call - Extract Metadata"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract basic metadata
        metadata = {
            "url": url,
            "title": "",
            "description": "",
            "keywords": "",
            "author": "",
            "canonical_url": "",
            "language": "",
            "open_graph": {},
            "twitter_card": {},
            "schema_org": [],
            "meta_robots": "",
            "charset": ""
        }
        
        # Title
        if soup.title:
            metadata["title"] = soup.title.string.strip() if soup.title.string else ""
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if name == 'description':
                metadata["description"] = content
            elif name == 'keywords':
                metadata["keywords"] = content
            elif name == 'author':
                metadata["author"] = content
            elif name == 'robots':
                metadata["meta_robots"] = content
            elif name == 'language' or name == 'lang':
                metadata["language"] = content
            elif meta.get('charset'):
                metadata["charset"] = meta.get('charset')
            elif meta.get('http-equiv', '').lower() == 'content-type':
                metadata["charset"] = content
            
            # Open Graph
            elif property_attr.startswith('og:'):
                og_key = property_attr.replace('og:', '')
                metadata["open_graph"][og_key] = content
            
            # Twitter Card
            elif name.startswith('twitter:'):
                twitter_key = name.replace('twitter:', '')
                metadata["twitter_card"][twitter_key] = content
        
        # Canonical URL
        canonical = soup.find('link', rel='canonical')
        if canonical:
            metadata["canonical_url"] = canonical.get('href', '')
        
        # Language from html tag
        if not metadata["language"]:
            html_tag = soup.find('html')
            if html_tag:
                metadata["language"] = html_tag.get('lang', '')
        
        # Schema.org structured data
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                schema_data = json.loads(script.string)
                metadata["schema_org"].append(schema_data)
            except (json.JSONDecodeError, AttributeError):
                continue
        
        # Format response
        result = f"✅ **Metadata extracted from: {url}**\n\n"
        result += f"📄 **Title:** {metadata['title']}\n"
        result += f"📝 **Description:** {metadata['description'][:200]}{'...' if len(metadata['description']) > 200 else ''}\n"
        result += f"🏷️ **Keywords:** {metadata['keywords']}\n"
        result += f"👤 **Author:** {metadata['author']}\n"
        result += f"🌐 **Language:** {metadata['language']}\n"
        result += f"🔗 **Canonical URL:** {metadata['canonical_url']}\n"
        result += f"🤖 **Robots:** {metadata['meta_robots']}\n\n"
        
        if metadata["open_graph"]:
            result += "📱 **Open Graph Data:**\n"
            for key, value in metadata["open_graph"].items():
                result += f"  • {key}: {value}\n"
            result += "\n"
        
        if metadata["twitter_card"]:
            result += "🐦 **Twitter Card Data:**\n"
            for key, value in metadata["twitter_card"].items():
                result += f"  • {key}: {value}\n"
            result += "\n"
        
        if metadata["schema_org"]:
            result += f"📋 **Schema.org:** {len(metadata['schema_org'])} structured data blocks found\n"
        
        return result
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error fetching URL content: {str(e)}"
    except Exception as e:
        return f"⚠️ Error parsing metadata: {str(e)}"

@tool
def fetch_content(url: str) -> str:
    """Fetch and parse the main content from a URL using LangChain WebBaseLoader.
    Args:
        url: The URL to fetch content from
    Returns:
        Extracted and cleaned content from the webpage
    """
    global chat_system
    chat_system = "Tool call - Fetch Content"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        # Use LangChain WebBaseLoader for content extraction
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        if not documents:
            return "⚠️ No content could be extracted from the URL."
        
        # Get the main content
        content = documents[0].page_content
        
        # Split content for better handling
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Get first chunk as summary
        chunks = text_splitter.split_text(content)
        main_content = chunks[0] if chunks else content[:2000]
        
        # Clean up the content
        main_content = re.sub(r'\s+', ' ', main_content).strip()
        
        result = f"✅ **Content extracted from: {url}**\n\n"
        result += f"📊 **Content Length:** {len(content)} characters\n"
        result += f"📄 **Number of Chunks:** {len(chunks)}\n\n"
        result += f"**Main Content Preview:**\n{main_content}..."
        
        if len(chunks) > 1:
            result += f"\n\n💡 *This page has {len(chunks)} content sections. Use extract_main_content for full analysis.*"
        
        return result
        
    except Exception as e:
        return f"⚠️ Error fetching content: {str(e)}"

@tool
def extract_headings(url: str) -> str:
    """Extract all headings (H1-H6) from a URL for SEO analysis.
    Args:
        url: The URL to extract headings from
    Returns:
        Structured list of all headings with their hierarchy
    """
    global chat_system
    chat_system = "Tool call - Extract Headings"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract all headings
        headings = []
        heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        
        for tag in heading_tags:
            for heading in soup.find_all(tag):
                text = heading.get_text().strip()
                if text:  # Only include non-empty headings
                    headings.append({
                        'level': int(tag[1]),  # Extract number from h1, h2, etc.
                        'text': text,
                        'tag': tag.upper()
                    })
        
        if not headings:
            return f"⚠️ No headings found on {url}"
        
        result = f"✅ **Headings extracted from: {url}**\n\n"
        result += f"📊 **Total Headings:** {len(headings)}\n\n"
        
        # Group by heading level for SEO analysis
        heading_counts = {}
        for heading in headings:
            level = heading['level']
            heading_counts[level] = heading_counts.get(level, 0) + 1
        
        result += "**Heading Structure:**\n"
        for level in sorted(heading_counts.keys()):
            result += f"• H{level}: {heading_counts[level]} headings\n"
        
        result += "\n**All Headings:**\n"
        for heading in headings:
            indent = "  " * (heading['level'] - 1)
            result += f"{indent}• **{heading['tag']}:** {heading['text']}\n"
        
        # SEO recommendations
        result += "\n**SEO Analysis:**\n"
        h1_count = heading_counts.get(1, 0)
        if h1_count == 0:
            result += "⚠️ No H1 tag found - add one for better SEO\n"
        elif h1_count > 1:
            result += f"⚠️ Multiple H1 tags ({h1_count}) found - consider using only one\n"
        else:
            result += "✅ Good H1 structure (1 H1 tag found)\n"
        
        if heading_counts.get(2, 0) == 0:
            result += "💡 Consider adding H2 tags for better content structure\n"
        
        return result
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error fetching URL: {str(e)}"
    except Exception as e:
        return f"⚠️ Error extracting headings: {str(e)}"

@tool
def extract_faqs(url: str) -> str:
    """Extract FAQ sections and Q&A content from a URL for SEO analysis.
    Args:
        url: The URL to extract FAQs from
    Returns:
        Extracted FAQ content and structured data
    """
    global chat_system
    chat_system = "Tool call - Extract FAQs"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        faqs = []
        
        # Look for FAQ structured data (Schema.org)
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if data.get('@type') == 'FAQPage':
                        main_entity = data.get('mainEntity', [])
                        if isinstance(main_entity, list):
                            for item in main_entity:
                                if item.get('@type') == 'Question':
                                    question = item.get('name', '')
                                    answer = ''
                                    accepted_answer = item.get('acceptedAnswer', {})
                                    if isinstance(accepted_answer, dict):
                                        answer = accepted_answer.get('text', '')
                                    faqs.append({
                                        'question': question,
                                        'answer': answer,
                                        'source': 'Schema.org FAQ'
                                    })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') == 'Question':
                            question = item.get('name', '')
                            answer = ''
                            accepted_answer = item.get('acceptedAnswer', {})
                            if isinstance(accepted_answer, dict):
                                answer = accepted_answer.get('text', '')
                            faqs.append({
                                'question': question,
                                'answer': answer,
                                'source': 'Schema.org FAQ'
                            })
            except (json.JSONDecodeError, AttributeError):
                continue
        
        # Look for common FAQ patterns in HTML
        faq_patterns = [
            {'question_selector': 'dt', 'answer_selector': 'dd'},
            {'question_selector': '.faq-question', 'answer_selector': '.faq-answer'},
            {'question_selector': '.question', 'answer_selector': '.answer'},
            {'question_selector': 'h3', 'answer_selector': 'p'}
        ]
        
        for pattern in faq_patterns:
            questions = soup.find_all(class_=re.compile(pattern['question_selector'].replace('.', '').replace('class_', ''), re.I))
            if not questions:
                questions = soup.find_all(pattern['question_selector'].replace('.', ''))
            
            for question_elem in questions:
                question_text = question_elem.get_text().strip()
                if '?' in question_text and len(question_text) > 10:
                    # Find corresponding answer
                    answer_elem = question_elem.find_next_sibling()
                    answer_text = ''
                    if answer_elem:
                        answer_text = answer_elem.get_text().strip()
                    
                    # Avoid duplicates
                    if not any(faq['question'] == question_text for faq in faqs):
                        faqs.append({
                            'question': question_text,
                            'answer': answer_text,
                            'source': 'HTML Pattern'
                        })
        
        if not faqs:
            return f"⚠️ No FAQ content found on {url}"
        
        result = f"✅ **FAQ content extracted from: {url}**\n\n"
        result += f"📊 **Total FAQs Found:** {len(faqs)}\n\n"
        
        # Group by source
        schema_faqs = [faq for faq in faqs if faq['source'] == 'Schema.org FAQ']
        html_faqs = [faq for faq in faqs if faq['source'] == 'HTML Pattern']
        
        if schema_faqs:
            result += f"🏷️ **Structured Data FAQs ({len(schema_faqs)}):**\n"
            for i, faq in enumerate(schema_faqs, 1):
                result += f"\n**Q{i}:** {faq['question']}\n"
                result += f"**A:** {faq['answer'][:200]}{'...' if len(faq['answer']) > 200 else ''}\n"
        
        if html_faqs:
            result += f"\n📄 **HTML Pattern FAQs ({len(html_faqs)}):**\n"
            for i, faq in enumerate(html_faqs, 1):
                result += f"\n**Q{i}:** {faq['question']}\n"
                if faq['answer']:
                    result += f"**A:** {faq['answer'][:200]}{'...' if len(faq['answer']) > 200 else ''}\n"
        
        # SEO recommendations
        result += "\n**SEO Analysis:**\n"
        if schema_faqs:
            result += "✅ FAQ structured data found - excellent for rich snippets!\n"
        else:
            result += "💡 Consider adding FAQ structured data (Schema.org) for rich snippets\n"
        
        if len(faqs) >= 5:
            result += "✅ Good FAQ content volume for SEO\n"
        else:
            result += "💡 Consider adding more FAQ content for better SEO coverage\n"
        
        return result
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error fetching URL: {str(e)}"
    except Exception as e:
        return f"⚠️ Error extracting FAQs: {str(e)}"

@tool
def extract_main_content(url: str) -> str:
    """Extract and analyze the main content from a URL for comprehensive SEO insights.
    Args:
        url: The URL to analyze
    Returns:
        Comprehensive content analysis including word count, readability, and SEO metrics
    """
    global chat_system
    chat_system = "Tool call - Extract Main Content"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        # Use LangChain WebBaseLoader
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        if not documents:
            return "⚠️ No content could be extracted from the URL."
        
        content = documents[0].page_content
        
        # Also get raw HTML for additional analysis
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Content analysis
        word_count = len(content.split())
        char_count = len(content)
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Extract images
        images = soup.find_all('img')
        img_with_alt = [img for img in images if img.get('alt')]
        img_without_alt = [img for img in images if not img.get('alt')]
        
        # Extract links
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
            elif href.startswith('/'):
                internal_links.append(urljoin(url, href))
        
        # Basic readability (simple metric)
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        
        result = f"✅ **Comprehensive content analysis for: {url}**\n\n"
        result += "📊 **Content Metrics:**\n"
        result += f"• Word count: {word_count}\n"
        result += f"• Character count: {char_count}\n"
        result += f"• Sentences: {sentence_count}\n"
        result += f"• Paragraphs: {paragraph_count}\n"
        result += f"• Average words per sentence: {avg_words_per_sentence:.1f}\n\n"
        
        result += "🖼️ **Image Analysis:**\n"
        result += f"• Total images: {len(images)}\n"
        result += f"• Images with alt text: {len(img_with_alt)}\n"
        result += f"• Images missing alt text: {len(img_without_alt)}\n\n"
        
        result += "🔗 **Link Analysis:**\n"
        result += f"• Internal links: {len(internal_links)}\n"
        result += f"• External links: {len(external_links)}\n\n"
        
        # SEO recommendations
        result += "🎯 **SEO Recommendations:**\n"
        if word_count < 300:
            result += "⚠️ Content is quite short - consider expanding (300+ words recommended)\n"
        elif word_count > 1500:
            result += "✅ Good content length for SEO\n"
        else:
            result += "✅ Decent content length\n"
        
        if len(img_without_alt) > 0:
            result += f"⚠️ {len(img_without_alt)} images missing alt text\n"
        else:
            result += "✅ All images have alt text\n"
        
        if avg_words_per_sentence > 20:
            result += "💡 Consider shorter sentences for better readability\n"
        else:
            result += "✅ Good sentence length for readability\n"
        
        if len(internal_links) < 3:
            result += "💡 Consider adding more internal links for better site structure\n"
        else:
            result += "✅ Good internal linking structure\n"
        
        # Content preview
        result += f"\n📄 **Content Preview:**\n{content[:500]}..."
        
        return result
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error fetching URL: {str(e)}"
    except Exception as e:
        return f"⚠️ Error analyzing content: {str(e)}"

        
@tool
def validate_and_fetch_url(url: str) -> str:
    """Enhanced URL validation with basic SEO information.
    Args:
        url: The URL to validate and fetch basic info from
    Returns:
        Validation result with title, description, and quick SEO insights
    """
    global chat_system
    chat_system = "Tool call - URL Validation"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Basic info
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
        
        # Meta description
        description = ""
        desc_meta = soup.find('meta', attrs={'name': 'description'})
        if desc_meta:
            description = desc_meta.get('content', '')
        
        # Quick SEO checks
        h1_tags = soup.find_all('h1')
        meta_robots = soup.find('meta', attrs={'name': 'robots'})
        robots_content = meta_robots.get('content', '') if meta_robots else ''
        
        result = f"✅ **URL is valid and accessible**\n\n"
        result += f"🌐 **URL:** {url}\n"
        result += f"📄 **Title:** {title}\n"
        result += f"📝 **Description:** {description[:200]}{'...' if len(description) > 200 else ''}\n"
        result += f"🏷️ **H1 Tags:** {len(h1_tags)} found\n"
        result += f"🤖 **Robots:** {robots_content if robots_content else 'Not specified'}\n\n"
        
        # Quick SEO insights
        result += "⚡ **Quick SEO Insights:**\n"
        if len(title) < 30:
            result += "• Title might be too short\n"
        elif len(title) > 60:
            result += "• Title might be too long for search results\n"
        else:
            result += "• Title length looks good\n"
        
        if not description:
            result += "• Meta description is missing\n"
        elif len(description) > 160:
            result += "• Meta description might be too long\n"
        else:
            result += "• Meta description present\n"
        
        return result
        
    except requests.exceptions.RequestException as e:
        return f"⚠️ URL validation passed, but content fetch failed.\nError: {str(e)}"

@tool
def validate_name(name: str) -> str:
    """Validate a user's name and provide feedback.
    Args:
        name: The name to validate
    Returns:
        Validation result with personalized greeting
    """
    global chat_system
    chat_system = "Tool call - Name Validation"
    
    # Clean the name
    cleaned_name = name.strip()
    
    # Basic validation rules
    if not cleaned_name or len(cleaned_name.split()) == 0 or len(cleaned_name.split()[0]) < 2:
        return "❌ Please enter your name to continue."

    if not cleaned_name:
        return "❌ Name cannot be empty. Please provide your name."
    
    if len(cleaned_name) < 2:
        return "❌ Name must be at least 2 characters long. Please provide your full name."
    
    if len(cleaned_name) > 50:
        return "❌ Name is too long (max 50 characters). Please provide a shorter version."
    
    # Check for valid characters (letters, spaces, hyphens, apostrophes)
    if not re.match(r"^[a-zA-Z\s\-'\.]+$", cleaned_name):
        return "❌ Name can only contain letters, spaces, hyphens, and apostrophes. Please provide a valid name."
    
    # Check for reasonable format
    if re.match(r"^[^a-zA-Z]*$", cleaned_name):
        return "❌ Name must contain at least one letter. Please provide a valid name."
    
    # Check for excessive repetition
    if re.search(r"(.)\1{4,}", cleaned_name):
        return "❌ Name contains too many repeated characters. Please provide a valid name."
    
    # Split name parts
    name_parts = cleaned_name.split()
    
    # Capitalize each part properly
    formatted_name = " ".join([
        part.capitalize() if not any(c in part for c in ["-", "'"]) 
        else "-".join([p.capitalize() for p in part.split("-")]) if "-" in part
        else "'".join([p.capitalize() for p in part.split("'")]) if "'" in part
        else part.capitalize()
        for part in name_parts
    ])
    
    # Generate personalized greeting
    first_name = name_parts[0].capitalize()
    
    if len(name_parts) == 1:
        greeting = f"✅ Nice to meet you, {formatted_name}! Welcome to your SEO Assistant."
    elif len(name_parts) == 2:
        greeting = f"✅ Hello {formatted_name}! Great to have you here, {first_name}."
    else:
        greeting = f"✅ Welcome {formatted_name}! I'll call you {first_name} if that's okay."
    
    # Store the validated name in session (you might want to save this to database)
    return f"{greeting}\n\n🎯 Now I'm ready to help you with all your SEO needs!"

@tool
def check_name_requirement(session_id: str, user_message: str) -> str:
    """Check if user needs to provide name before proceeding.
    Args:
        session_id: Current session ID
        user_message: User's message content
    Returns:
        Message requiring name if needed, empty string if name validated
    """
    global user_name_status
    
    # Check if this session has validated name
    if session_id not in user_name_status or not user_name_status[session_id].get('name_validated', False):
        return """🙏 **Please provide your name first!**

I'd love to help you with your SEO needs, but I need to know who I'm talking to first. 

**What's your name?** 

Once you share your name, I'll be able to provide personalized SEO assistance, keyword research, content optimization strategies, and much more! 🚀✨"""
    
    return ""  # Name is validated, proceed normally

# =============================
# Simple Views (Frontend Pages)
# =============================

def chatbot_view(request):
    return render(request, 'chatbot.html')

def dashboard_view(request):
    return render(request, 'dashboard.html')

# =============================
# Global State for Chat Sessions
# =============================
chat_history = {}  # Store chat history by session
tokenresponse = {}  # Store chat history by session
messagedata = {}  # Store chat history by session
count = 0

# =============================
# Token Usage View
# =============================
def get_token_usage_view(request, run_id):
    token_usage = tokenresponse[run_id]
    return JsonResponse({
        'response_type': token_usage['response_type'],
        'tools_used': token_usage['tools_used'],
        'source': token_usage['source'],
        'session': run_id,
        'run_id': token_usage['run_id'],
        'input_tokens': token_usage['input_tokens'],
        'output_tokens': token_usage['output_tokens'],
        'total_tokens': token_usage['total_tokens']
    })

# =============================
# Streaming Welcome Message
# =============================
@csrf_exempt
def streaming_welcome_message(request):
    """Stream the welcome message with typing effect"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests allowed'})
    
    try:
        data = json.loads(request.body.decode('utf-8'))
        session_id = data.get("session_id", "default")
        
        return StreamingHttpResponse(
            generate_name_request_stream(session_id),
            content_type='text/plain'
        )
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def generate_name_request_stream(session_id):
    """Generate streaming name request message"""
    global chat_history, tokenresponse, messagedata, count, user_name_status
    
    # Initialize name status for new session
    if session_id not in user_name_status:
        user_name_status[session_id] = {'name_validated': False, 'name': ''}
    
    try:
        run_id = generate_run_id()
        count = 0
        
        # Initialize session with name collection system message
        if session_id not in chat_history:
            chat_history[session_id] = [SystemMessage(content=system_message)]
            tokenresponse[session_id] = {}
            messagedata[session_id] = []
            
            # Add system message
            messagedata[session_id].append({
                "message_type": "system",
                "content": system_message,
                "message_id": generate_message_id(),
                "tool_calls": '',
                "tool_call_id": '',
                "count": count,
                "response_type": 'name_collection',
                "tools_used": '',
                "source": 'system',
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            })
        
        # Stream the welcome message with typing effect
        words = welcome_message.split(' ')
        streamed_content = ""
        
        for i, word in enumerate(words):
            streamed_content += word + " "
            
            # Add slight delay for typing effect
            time.sleep(0.04)  # Slightly faster for welcome message
            
            # Yield the current word
            yield word + " "
            
            # Add pauses at line breaks for better effect
            if word.endswith('\n') or word.endswith('**') or word.endswith('!'):
                time.sleep(0.1)
        
        # Store the complete welcome message in chat history
        chat_history[session_id].append(AIMessage(content=welcome_message))
        
        # Calculate token usage
        token_count = len(welcome_message.split())
        
        # Store message data
        messagedata[session_id].append({
            "message_type": "ai",
            "content": welcome_message,
            "message_id": generate_message_id(),
            "tool_calls": '',
            "tool_call_id": '',
            "count": count,
            "response_type": 'welcome_message',
            "tools_used": False,
            "source": 'system',
            "input_tokens": 0,
            "output_tokens": token_count,
            "total_tokens": token_count
        })
        
        # Save to database
        tokenresponse[session_id].update({
            'total_tokens': token_count,
            'input_tokens': 0,
            'output_tokens': token_count,
            'response_type': 'welcome_message',
            'tools_used': False,
            'source': 'system',
            'run_id': run_id
        })
        
        session, created = get_or_create_chat_session(session_id, run_id, None, False, 'welcome_message')
        save_message_to_db(session, messagedata, run_id, count)
        
    except Exception as e:
        yield f"\n[Error occurred: {str(e)}]"

# =============================
# Chatbot Main Endpoint
# =============================
@csrf_exempt
def chatbot_input(request):
    """SEO-focused chatbot with name validation and chat history"""
    global chat_system, chat_history, tokenresponse, count, user_name_status

    if not hasattr(request, 'POST') or request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})
    
    data = json.loads(request.body.decode('utf-8'))
    usermessage = data.get("message", "")
    session_id = data.get("session_id", "default")

    runid = generate_run_id()

    if not usermessage:
        return JsonResponse({'success': False, 'error': 'No message provided'})

    try:
        # Initialize name status if not exists
        if session_id not in user_name_status:
            user_name_status[session_id] = {'name_validated': False, 'name': ''}

        # Check if this looks like a name (simple heuristic)
        is_potential_name = (
            len(usermessage.split()) <= 4 and  # Names are usually 1-4 words
            not any(word in usermessage.lower() for word in ['help', 'seo', 'keyword', 'website', 'search', 'rank', 'optimize', 'what', 'how', 'can', 'you']) and
            re.match(r'^[a-zA-Z\s\-\'\.]+$', usermessage.strip())  # Only contains name-like characters
        )

        # Tools available
        tools = [multiply, validate_and_fetch_url, validate_name, check_name_requirement, extract_metadata, fetch_content, extract_headings, extract_faqs, extract_main_content]

        llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=OPENAIKEY,
            model=MODEL
        )
        llm_with_tools = llm.bind_tools(tools)
        chat_system = "LLM Call"
        
        # Initialize or get existing chat history for this session
        if session_id not in chat_history:
            count = 0
            chat_history[session_id] = [SystemMessage(content=system_message)]
            tokenresponse[session_id] = {}
            messagedata[session_id] = []
            messagedata[session_id].append({
                "message_type": "system",
                "content": system_message,
                "message_id": generate_message_id(),
                "tool_calls": '',
                "tool_call_id": '',
                "count": count,
                "response_type": '',
                "tools_used": '',
                "source": '',
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            })
        else:
            count = count + 1

        # Get current conversation messages
        messages = chat_history[session_id].copy()
        
        # Check if user needs to provide name first
        if not user_name_status[session_id]['name_validated']:
            if is_potential_name:
                # This looks like a name, try to validate it
                # Add user message and let the LLM process with validate_name tool
                messages.append(HumanMessage(content=usermessage))
                
                # Add instruction to validate the name
                validation_instruction = f"The user provided: '{usermessage}'. Please use the validate_name tool to validate this as their name."
                messages.append(HumanMessage(content=validation_instruction))
                
            else:
                
                # Add user message to history
                chat_history[session_id].append(HumanMessage(content=usermessage))
                chat_history[session_id].append(AIMessage(content=name_request_message))
                
                # Store message data
                messagedata[session_id].append({
                    "message_type": "human",
                    "content": usermessage,
                    "message_id": generate_message_id(),
                    "tool_calls": '',
                    "tool_call_id": '',
                    "count": count,
                    "response_type": '',
                    "tools_used": '',
                    "source": '',
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                })

                messagedata[session_id].append({
                    "message_type": "ai",
                    "content": name_request_message,
                    "message_id": generate_message_id(),
                    "tool_calls": '',
                    "tool_call_id": '',
                    "count": count,
                    "response_type": 'name_request',
                    "tools_used": False,
                    "source": 'system',
                    "input_tokens": 0,
                    "output_tokens": len(name_request_message.split()),
                    "total_tokens": len(name_request_message.split())
                })

                # Save to database
                session, created = get_or_create_chat_session(session_id, runid, None, False, chat_system)
                save_message_to_db(session, messagedata, runid, count)
                
                return StreamingHttpResponse(
                    stream_static_message(name_request_message),
                    content_type='text/plain'
                )
        else:
            # Name is validated, proceed normally
            messages.append(HumanMessage(content=usermessage))

        # Store user message
        messagedata[session_id].append({
            "message_type": "human",
            "content": usermessage,
            "message_id": generate_message_id(),
            "tool_calls": '',
            "tool_call_id": '',
            "count": count,
            "response_type": '',
            "tools_used": '',
            "source": '',
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        })

        # Get initial AI response
        ai_msg = llm_with_tools.invoke(messages)
        
        # Process tool calls if any
        if ai_msg.tool_calls:
            messages.append(ai_msg)

            messagedata[session_id].append({
                "message_type": "ai",
                "content": ai_msg.content,
                "message_id": generate_message_id(),
                "tool_calls": ai_msg.tool_calls,
                "tool_call_id": '',
                "count": count,
                "response_type": '',
                "tools_used": '',
                "source": '',
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            })
            
            tool_mapping = {
                "multiply": multiply,
                "validate_and_fetch_url": validate_and_fetch_url,
                "validate_name": validate_name,
                "check_name_requirement": check_name_requirement,
                "extract_metadata": extract_metadata,
                "fetch_content": fetch_content,
                "extract_headings": extract_headings,
                "extract_faqs": extract_faqs,
                "extract_main_content": extract_main_content
            }

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                if tool_name in tool_mapping:
                    selected_tool = tool_mapping[tool_name]
                    try:
                        # Run the tool with args
                        tool_result = selected_tool.invoke(tool_args)
                        
                        # Special handling for name validation
                        if tool_name == "validate_name" and "✅" in tool_result:
                            # Name validation successful
                            user_name_status[session_id]['name_validated'] = True
                            user_name_status[session_id]['name'] = tool_args.get('name', '')
                        
                        # Append ToolMessage with matching tool_call_id
                        messages.append(
                            ToolMessage(content=str(tool_result), tool_call_id=tool_call_id)
                        )

                        messagedata[session_id].append({
                            "message_type": "tool",
                            "content": str(tool_result),
                            "message_id": generate_message_id(),
                            "tool_calls": ai_msg.tool_calls,
                            "tool_call_id": tool_call_id,
                            "count": count,
                            "response_type": '',
                            "tools_used": '',
                            "source": '',
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0
                        })
                        
                    except Exception as e:
                        messages.append(
                            ToolMessage(content=f"⚠️ Error running tool {tool_name}: {str(e)}", tool_call_id=tool_call_id)
                        )

                        messagedata[session_id].append({
                            "message_type": "tool",
                            "content": f"⚠️ Error running tool {tool_name}: {str(e)}",
                            "message_id": generate_message_id(),
                            "tool_calls": "",
                            "tool_call_id": tool_call_id,
                            "count": count,
                            "response_type": '',
                            "tools_used": '',
                            "source": '',
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0
                        })
                        
                else:
                    messages.append(
                        ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_call_id)
                    )
                    messagedata[session_id].append({
                        "message_type": "tool",
                        "content": f"Unknown tool: {tool_name}",
                        "message_id": generate_message_id(),
                        "tool_calls": "",
                        "tool_call_id": tool_call_id,
                        "count": count,
                        "response_type": '',
                        "tools_used": '',
                        "source": '',
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0
                    })
            
            # Store the conversation history (user message + AI response + tool messages)
            chat_history[session_id].append(HumanMessage(content=usermessage))
            chat_history[session_id].extend(messages[len(chat_history[session_id]):])
            
            return StreamingHttpResponse(
                build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system, count,usermessage),
                content_type='text/plain'
            )

        else:
            # No tools called, store conversation and stream response
            chat_history[session_id].append(HumanMessage(content=usermessage))
            
            return StreamingHttpResponse(
                build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system, count,usermessage),
                content_type='text/plain'
            )

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Processing error: {str(e)}'
        })

# =============================
# Streaming Helpers
# =============================
def stream_static_message(message):
    """Stream a static message with typing effect"""
    words = message.split(' ')
    
    for word in words:
        time.sleep(0.05)  # Typing effect
        yield word + " "
        
        # Add pauses at line breaks for better effect
        if word.endswith('\n') or word.endswith('**') or word.endswith('!'):
            time.sleep(0.1)
    
def build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system, count,usermessage):
    """Unified streaming response generator with token usage tracking"""
    try:
        final_response_content = ""
        usage_metadata = None
        run_id = None

        for chunk in llm_with_tools.stream(messages):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                final_response_content += chunk.content
                run_id = chunk.id
                yield chunk.content

            # Capture usage metadata (Anthropic supports this in stream)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata

        # Store the final AI response in chat history
        if final_response_content:
            if usage_metadata is None:  # Fallback for OpenAI
                resp = llm_with_tools.invoke(messages, config={"include_usage_metadata": True})
                usage_metadata = resp.usage_metadata

            chat_history[session_id].append(AIMessage(content=final_response_content))

            # Save token usage
            response_type = 'Tool Usage' if ai_msg.tool_calls else 'General Conversation'
            tools_used = len(ai_msg.tool_calls) > 0 if ai_msg.tool_calls else False
            
            tokenresponse[session_id].update({
                'total_tokens': usage_metadata['total_tokens'],
                'input_tokens': usage_metadata['input_tokens'],
                'output_tokens': usage_metadata['output_tokens'],
                'response_type': response_type,
                'tools_used': tools_used,
                'source': chat_system,
                'run_id': run_id
            })

            if chat_system == "Tool call - Name Validation": 

                first_name = extract_name(final_response_content)

                
                session, create = get_or_create_chat_session(session_id, run_id, first_name, True,chat_system)
            else:
                session, create = get_or_create_chat_session(session_id, run_id,None,False,chat_system)

            messagedata[session_id].append({
                "message_type": "ai",
                "content": final_response_content,
                "message_id": generate_message_id(),
                "tool_calls": '',
                "tool_call_id": '',
                "count": count,
                "response_type": response_type,
                "tools_used": tools_used,
                "source": chat_system,
                "input_tokens": usage_metadata['input_tokens'],
                "output_tokens": usage_metadata['output_tokens'],
                "total_tokens": usage_metadata['total_tokens']
            })

            save_message_to_db(session, messagedata, run_id, count)

    except (OpenAIError, APITimeoutError) as e:
        yield f"\n[Error occurred: {str(e)}]"


# =============================
# Database Helpers
# =============================
def get_or_create_chat_session(session_id, run_id=None, user_name=None, name_validated=False , chat_system='welcome_message'):
    """Get existing chat session or create new one"""
    try:

        session, created = ChatSession.objects.get_or_create(
            session_id=session_id,
            defaults={
                'run_id': run_id,
                'user_name': user_name,
                'name_validated': name_validated,
                'is_active': True
            }
        )
        
        # Update run_id if provided and different
        if run_id and session.run_id != run_id:
            session.run_id = run_id
            if chat_system == "Tool call - Name Validation": 
            
                session.user_name = user_name
                session.name_validated=name_validated
                session.save(update_fields=['run_id','user_name','updated_at','name_validated',])
            else:
                session.save(update_fields=['run_id','updated_at'])
            
        return session, created
    except Exception as e:
        print(f"Error creating/getting chat session: {str(e)}")
        raise

def save_message_to_db(session, messagedata, run_id=None, count=0):
    """Save a single message to database"""
    try:
        
        # Get the next order number for this session
        last_message = ChatMessage.objects.filter(session=session).order_by('-order').first()
        next_order = (last_message.order + 1) if last_message else 1
        
        # Generate IDs if not provided
        if not run_id:
            run_id = session.run_id or generate_run_id()
        

        savemessagedata = messagedata[session.session_id] 

        for msg in savemessagedata:
            messagecount = msg["count"]
            if messagecount == count:
                message_type = msg["message_type"]
                content = msg["content"]
                message_id = msg["message_id"]
                tool_calls = msg["tool_calls"]
                tool_call_id = msg["tool_call_id"]
                response_type = msg["response_type"]
                tools_used = msg["tools_used"]
                source = msg["source"]
                input_tokens   = msg["input_tokens"]
                output_tokens   = msg["output_tokens"]
                total_tokens   = msg["total_tokens"]
                # Create the message
                ChatMessage.objects.create(
                    session=session,
                    message_id=message_id,
                    run_id=run_id,
                    message_type=message_type,
                    content=content,
                    tool_call_id=tool_call_id or '',
                    tool_calls=tool_calls or {},
                    tools_used= tools_used,
                    response_type= response_type,
                    source= source,
                    input_tokens= input_tokens,
                    output_tokens= output_tokens,
                    total_tokens= total_tokens,
                    order=next_order
                )
        
        # # Update session's updated_at timestamp and run_id
        session.run_id = run_id
        session.save(update_fields=['updated_at', 'run_id'])
        
        return count
    except Exception as e:
        print(f"Error saving message to DB: {str(e)}")
        raise

# =============================
# Utility Helpers
# =============================
def generate_message_id():
    """Generate a unique message ID"""
    return f"msg--{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:12]}"

def extract_name(message: str) -> str:
    # Case 1: after comma and before ! or ?
    m = re.search(r",\s*([^!?]+)[!?]", message)
    if m:
        return m.group(1).strip(" '")

    # Case 2: after greeting (Hello, Hi, etc.) before ! or ?
    m = re.search(
        r"\b(?:Hello|Hi|Hey|Welcome|Great to meet you|Nice to meet you)[,\s]+([^!?]+)[!?]", 
        message, re.IGNORECASE
    )
    if m:
        return m.group(1).strip(" '")

    # Case 3: fallback → first word
    return message.strip().split()[0].strip(" '")

def generate_run_id():
    """Generate a unique run ID for conversation tracking"""
    return f"run--{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:12]}"

