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
from .models import ChatSession, ChatMessage, Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import mimetypes
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
# Parsing HTML
from bs4 import BeautifulSoup
import ast
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from typing import Literal

# LangChain & OpenAI integration
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage
from langchain_core.messages import ToolMessage,HumanMessage,AIMessageChunk
from langchain.tools import tool
from openai import OpenAIError, APITimeoutError
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from .admin_views import load_chat_history_from_db


# =============================
# Global Variables & Config
# =============================

# Tracks current chat system state
chat_system = ""
messages = []

excluded_extensions = {
    '.php', '.doc', '.docx', '.mp3', '.wav', '.jpg', '.png', '.pdf', '.csv', '.html',
    '.mp4', '.avi', '.mov', '.mkv',  # Video files
    '.zip', '.rar', '.tar', '.gz',  # Compressed files
    '.xls', '.xlsx', '.xlsm'  # Excel files
}

# Load environment variables for OpenAI API
load_dotenv()
OPENAIKEY = os.getenv('OPEN_AI_KEY')
MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"
APP_URL = "http://127.0.0.1:8000/"

llm_with_tools = None
ai_msg = None
# =============================
# Global State for Chat Sessions
# =============================
chat_history = {}  # Store chat history by session
tokenresponse = {}  # Store chat history by session
messagedata = {}  # Store chat history by session
count = 0

# -------------------------------
# Chroma DB setup
# -------------------------------
CHROMA_DB_PATH = os.path.join(settings.BASE_DIR, "chroma_db")
if not os.path.exists(CHROMA_DB_PATH):
    os.makedirs(CHROMA_DB_PATH)

embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAIKEY)

vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embedding_function,
    persist_directory=CHROMA_DB_PATH,
)


# Tracks name validation per session
user_name_status = {}  # session_id -> {'name_validated': bool, 'name': str}

# System message prompt for chatbot behavior
system_message = """You are an expert SEO assistant with name validation capabilities.

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

    Guidelines:
    1. Always start by collecting the user's name using a warm, friendly approach.
    2. Use the validate_name tool when they provide their name.
    3. After name validation, provide your full SEO services.
    4. Only answer SEO-related questions in detail AFTER name is validated.
    5. If the user asks a question unrelated to SEO, politely say you specialize in SEO and cannot answer other topics.
    6. If the user clearly asks for a basic arithmetic calculation (e.g., multiplication, addition, subtraction, division), call the appropriate calculation tool.
    7. If the user provides or asks to validate a URL, call the `validate_and_fetch_url` tool to check its validity and fetch its title.
    8. If the user provides a URL, call the `seo_url_parser` tool to parse the content and extract metadata, headings, FAQs, etc.
    9. If the user provides One or more URL + One or more Document IDs (one for SEO guidelines and another for semantic guidelines), call the `validate_url_with_document` tool to check compliance against both guidelines.
    12. With Output Parsers: Ensure the result from `validate_url_with_document` is structured using output parsers (so compliance, non-compliance, and analysis data are cleanly separated).
    13. You can analyze uploaded documents for SEO-related insights when asked.
    14. Always provide actionable, practical SEO recommendations with clear steps.
    15. If user sends any message before providing a valid name, remind them to share their name first.
    """

welcome_message = """👋 **Welcome to your Personal SEO Assistant!**

    I'm here to help you dominate search rankings and grow your online presence! 🚀  

    I can assist you with:  
    🔍 **Keyword Research & Analysis**  
    📊 **Content Optimization Strategies**  
    🔧 **Technical SEO Recommendations**  
    🏷️ **Meta Tag Optimization**  
    🔗 **Link Building Strategies**  
    📈 **SEO Auditing & Reporting**  
    🎯 **Search Ranking Analysis**  
    🕵️ **Competitor Analysis**  

    I can also help with:  
    - Basic calculations (just ask me to multiply numbers)  
    - URL validation and title fetching  
    - Document analysis for SEO insights  
    - One or More URLs + One or More Documents → Full SEO + Semantic Guideline Validation for all pages

    But first, I'd love to know who I'm talking to. **What's your name?**  

    Once I know, I'll be able to provide:  
    ✨ **Personalized SEO Strategies**  
    ✨ **Targeted Recommendations**  
    ✨ **A clear roadmap to achieve your SEO goals**  

    So please tell me your name, and let's start your SEO journey together! 🚀✨"""


# This doesn't look like a name, ask for name first
name_request_message = """🙏 **Please provide your name first!**
    I'd love to help you with your SEO needs, but I need to know who I'm talking to first. 
    **What's your name?** 
    Once you share your name, I'll be able to provide personalized SEO assistance, keyword research, content optimization strategies, and much more! 🚀✨"""


class SeoAuditMetadata(BaseModel):
    """Metadata schema for the SEO Strategic Auditor knowledge base."""

    primary_topic: Literal[
        "Core SEO Concept", "Silo Architecture", "On-Page SEO", 
        "Keyword Research", "Link Building", "Technical SEO", "Internal Process"
    ] = Field(description="The primary, high-level topic of the text chunk.")

    specific_element: str = Field(description="The specific SEO element being discussed, e.g., 'Title Tag', 'Anchor Text', 'Physical Silo', 'Client Onboarding'.")
    
    content_type: Literal[
        "Strategic Principle", "Actionable Tactic", "Explanatory Model", "Internal Guideline"
    ] = Field(description="The nature of the information in the chunk.")

    source_brand: Literal["Google", "Ahrefs", "Moz", "Bruce Clay", "Brihaspati Tech"] = Field(description="The brand or source of the document.")

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
def validate_and_fetch_url(url: str) -> str:
    """Validate a URL and fetch its title if valid.
    Args:
        url: The URL to validate and fetch title from
    Returns:
        Validation result and title if successful
    """
    global chat_system
    chat_system = "Tool call - URL Validation"
    
    if not validators.url(url):
        return "❌ Invalid URL. Please enter a valid one (e.g., https://example.com)."
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
        
        return f"✅ URL is valid.\nTitle: {title}"
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

@tool
def seo_url_parser(url: str) -> str:
    """
    SEO URL parser that validates URL and extracts headings, metadata, and FAQ data.
    
    Args:
        url: The URL to parse and extract data from
        
    Returns:
        JSON string containing validation status, headings, metadata, and FAQ data
    """
    global chat_system
    chat_system = "Tool call - SEO URL Parser"
    
    result = {
        "url": url,
        "validation": {
            "is_valid": False,
            "status_code": None,
            "error": None
        },
        "metadata": {},
        "headings": {},
        "faq_data": [],
        "additional_data": {}
    }
    
    # Step 1: Validate URL
    if not validators.url(url):
        result["validation"]["error"] = "Invalid URL format"
        return json.dumps(result, indent=2)
    
    try:
        # Step 2: Fetch the URL content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, timeout=15, headers=headers)
        result["validation"]["status_code"] = response.status_code
        response.raise_for_status()
        
        result["validation"]["is_valid"] = True
        
        # Step 3: Parse HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Step 4: Extract Metadata
        result["metadata"] = extract_metadata(soup, url)
        
        # Step 5: Extract Headings
        result["headings"] = extract_headings(soup)
   
        # Step 6: Extract FAQ Data
        result["faq_data"] = extract_faq_data(soup)
        
        # Step 7: Extract Additional SEO Data
        result["additional_data"] = extract_additional_seo_data(soup, url)
        
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    except requests.exceptions.RequestException as e:
        result["validation"]["error"] = f"Request failed: {str(e)}"
        return json.dumps(result, indent=2)
    except Exception as e:
        result["validation"]["error"] = f"Parsing error: {str(e)}"
        return json.dumps(result, indent=2)


@tool
def validate_url_with_document(url: list, document_ids: list) -> str:
    """Validate a URL and fetch its title if valid.
    Args:
        url: List of URL to validate and fetch title from
        document_ids: List of document IDs to validate against
    Returns:
        Validation result and title if successful
    """
    global chat_system
    chat_system = "Tool call - URL Validation with documents"
    
    try:        
        comparison_result = {
            "success": True,
            "url": url,
            "document_ids": document_ids,
        }
        return json.dumps(comparison_result, indent=2, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
       return json.dumps({
            "success": False,
            "error": f"Validation failed: {str(e)}"
        }, indent=2)

# =============================
# Simple Views (Frontend Pages)
# =============================

def chatbot_view(request):
    return render(request, 'chatbot.html')

def dashboard_view(request):
    return render(request, 'dashboard.html')



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
        'total_tokens': token_usage['total_tokens'],
        'log_input': token_usage['log_input'],
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
            'run_id': run_id,
            'log_input': ''
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
        tools = [multiply, validate_and_fetch_url, validate_name, seo_url_parser,validate_url_with_document]

        llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=OPENAIKEY,
            model=MODEL
        )
        llm_with_tools = llm.bind_tools(tools)
        chat_system = "LLM Call"
        
        all_docs = vectorstore._collection.get(include=["metadatas"])
        

        url_pattern = re.compile(r'https?://[^\s]+')
        match = url_pattern.search(usermessage)
        ids_str = ''
        documentid = ""
        reg_result = {}
        if match:
            # Extract just the document_id values
            document_ids = set()
            for metadata in all_docs.get("metadatas", []):
                if metadata and "document_id" in metadata:
                    document_ids.add(metadata["document_id"])
            ids = list(document_ids)

            ids_str = ", ".join(ids)
            if ids_str:
                documentid = f"against the document id {ids_str}"


        # Auto-generate structured metadata with LLM
        llm_metadata = generate_chunk_label(usermessage)

        reg_result = get_relevant_documents(usermessage,llm_metadata,session_id)

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
                usermessage = usermessage + " " + documentid              
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

            usermessage = usermessage + " " + documentid
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
        tool_result = None
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
                "validate_and_fetch_url":validate_and_fetch_url,
                "validate_name": validate_name,
                "seo_url_parser":seo_url_parser,
                "validate_url_with_document":validate_url_with_document
            }

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                if tool_name in tool_mapping:
                    tool_args["session_id"] = session_id
                    tool_args["tool_call_id"] = tool_call_id
                    tool_args["count"] = count 
                    selected_tool = tool_mapping[tool_name]
                    try:
                        # Run the tool with args
                        tool_result = selected_tool.invoke(tool_args)
                        
                        # Special handling for name validation
                        if tool_name == "validate_name" and "✅" in tool_result:
                            # Name validation successful
                            user_name_status[session_id]['name_validated'] = True
                            user_name_status[session_id]['name'] = tool_args.get('name', '')

                        # Sitemap Tool One by One URL Response
                        if chat_system == "Tool call - URL Validation with documents":
                            return StreamingHttpResponse(
                                sitemap_stream_static_message(llm_with_tools, tool_result, ai_msg, session_id, chat_system, count,tool_call_id,messages,reg_result),
                                content_type='text/plain'
                            )    
                        
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
                build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system, count,tool_result),
                content_type='text/plain'
            )

        else:
            # No tools called, store conversation and stream response
            chat_history[session_id].append(HumanMessage(content=usermessage))
            
            return StreamingHttpResponse(
                build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system, count,tool_result=None),
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

def sitemap_stream_static_message(llm_with_tools, tool_result, ai_msg, session_id, chat_system, count,tool_call_id,messages,reg_result):
    
    """Stream a static message with typing effect"""
    
    sitemap_response = ''
    xcount = 0
    lenusername = 0
    linkurls = []
	
    # Define the main dictionary
    comparison_result = {}
    tool_result_dict = json.loads(tool_result)
    urls = tool_result_dict['url']
    document_ids = tool_result_dict['document_ids']

    yield yield_runtime_status("🌐 Validating URL accessibility...")
    for i, url in enumerate(urls, 1):
        yield yield_runtime_status(f"🌐 Validating URL {i}/{len(urls)}: {url}")
        valid_url = validate_and_fetch_url.invoke({
            "url": url,
        })

        if url.endswith(".xml"):
            yield yield_runtime_status("🗺️ Fetching sitemap contents...")
            loader = SitemapLoader(web_path=url, continue_on_failure=True)
            documents = loader.load()
            yield yield_runtime_status("🔗 Extracting all valid URLs...")

            for doc in documents[:5]:
                sitemapurl = doc.metadata.get("loc")
                if sitemapurl and is_valid_url(sitemapurl):
                    linkurls.append(sitemapurl)
                if not linkurls:
                    yield yield_runtime_status("🔗 No valid URLs found in sitemap...")

            for i, linkurl in enumerate(linkurls, 1):
                yield yield_runtime_status(f"🌐 Validating URL {i}/{len(linkurls)}: {linkurl}")
                valid_url = validate_and_fetch_url.invoke({
                    "url": linkurl,
                })
                comparison_result[linkurl] = {
                    "url": linkurl,
                    #"document_ids": ", ".join(document_ids),
                    "document_content":reg_result,
                    "validate_url_result" : valid_url,
                }
            yield yield_runtime_status("📋 Preparing detailed analysis...")
            for i, linkurl in enumerate(linkurls, 1):
                try:
                    yield yield_runtime_status(f"📋 Preparing URL detailed analysis {i}/{len(linkurls)}: {linkurl}")
                    pre_result = seo_url_parser.invoke({
                        "url": linkurl,
                    })
                    parsed_data = json.loads(pre_result)
                    comparison_result[linkurl].update({
                        "validation": parsed_data.get("validation", {}),
                        "seo_analysis": {
                            "metadata": parsed_data.get("metadata", {}),
                            "headings": parsed_data.get("headings", {}),
                            "faq_data": parsed_data.get("faq_data", []),
                            "additional_data": parsed_data.get("additional_data", {})
                        }
                    })



                except Exception as url_error:
                    
                    yield f"\nValidation failed for {linkurl}: {str(url_error)}"

            
        else:
            yield yield_runtime_status(f"📋Preparing URL detailed analysis: {url}")
            pre_result = seo_url_parser.invoke({
                "url": url,
            })
            parsed_data = json.loads(pre_result)
            comparison_result[url] = {
                "url": url,
                #"document_ids": ", ".join(document_ids),
                "document_content":reg_result,
                "validate_url_result" : valid_url,
                "validation": parsed_data.get("validation", {}),
                "seo_analysis": {
                    "metadata": parsed_data.get("metadata", {}),
                    "headings": parsed_data.get("headings", {}),
                    "faq_data": parsed_data.get("faq_data", []),
                    "additional_data": parsed_data.get("additional_data", {})
                }                
            }

    yield yield_runtime_status("📊 Creating site-wide compliance report...")
    total = len(comparison_result)   # total URLs
    for i, (url, data) in enumerate(comparison_result.items(), 1):
        yield yield_runtime_status(f"📊 Creating site-wide compliance report {i}/{total}: {data['url']}")
        if xcount == 0:
            messages.append(
                ToolMessage(content=str(data), tool_call_id=tool_call_id)
            )
        else:
            messages.append(
                HumanMessage(content=json.dumps(data))
            )
        sitemap_response2 = sitemap_build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system, count)
        messages.append(
            AIMessage(content=sitemap_response2)
        )
        sitemap_response += sitemap_response2
        xcount = xcount + 1
        lenusername =  len(json.dumps(data))

    total_tokens = lenusername + len(sitemap_response.split())
    
    # Call your existing function


   


       
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
    messagedata[session_id].append({
        "message_type": "ai",
        "content": sitemap_response,
        "message_id": generate_message_id(),
        "tool_calls": '',
        "tool_call_id": '',
        "count": count,
        "response_type": 'Tool Usage',
        "tools_used": True,
        "source": chat_system,
        "input_tokens": lenusername,
        "output_tokens": len(sitemap_response.split()),
        "total_tokens": total_tokens
    })
    yield yield_runtime_status("✨ Analysis complete! Generating response...")
    # Save to database
    session, created = get_or_create_chat_session(session_id, generate_run_id(), None, False, chat_system)
    save_message_to_db(session, messagedata, generate_run_id(), count)


    json_ready_messages = serialize_messages(messages)
    pdf_id = save_to_json(json_ready_messages)
    tokenresponse[session_id].update({
        'total_tokens': total_tokens,
        'input_tokens': lenusername,
        'output_tokens': len(sitemap_response.split()),
        'response_type': 'Tool Usage',
        'tools_used': True,
        'source': chat_system,
        'log_input': f"<a href='{APP_URL}documents/{pdf_id}'>Click here to view the Log input to LLM</a>",
        'run_id': generate_message_id()
        
    }) 


    try:
        yield yield_runtime_status("📄 Generating PDF report...")
        pdf_id = pdfgenrateprocess(sitemap_response)
        # Yield a final chunk with PDF link
        yield f"\n\n📄 Response saved in PDF: <a href='{APP_URL}documents/{pdf_id}'>Click here to view the detailed report</a>\n"

    except Exception as e:
        yield f"\n[PDF generation failed: {str(e)}]"
    
def sitemap_build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system, count):
    """Unified streaming response generator with token usage tracking"""
    try:
        final_response_content = ""

        for chunk in llm_with_tools.stream(messages):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                final_response_content += chunk.content
                run_id = chunk.id
        return final_response_content+"\n\n++++ ++++ ++++ ++++ ++++\n"               
    except (OpenAIError, APITimeoutError) as e:
        return f"\n[Error occurred: {str(e)}]"

def build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system, count,tool_result):
    """Unified streaming response generator with token usage tracking"""
    try:
        final_response_content = ""
        usage_metadata = None
        run_id = None
        log_id = None
              
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
            
            if chat_system == "Tool call - Name Validation": 
                first_name = extract_name(final_response_content)
                session, create = get_or_create_chat_session(session_id, run_id, first_name, True,chat_system)

            else:
                session, create = get_or_create_chat_session(session_id, run_id,None,False,chat_system)

            # === PDF GENERATION: Save entire final_response_content ===
            if chat_system in ["Tool call - SEO URL Parser"]:
                try:

                    pdf_id = pdfgenrateprocess(final_response_content)
                    

                    # Yield a final chunk with PDF link
                    yield f"\n\n📄 Response saved in PDF: <a href='{APP_URL}documents/{pdf_id}'>Click here to view the detailed report</a>\n"

                except Exception as e:
                    yield f"\n[PDF generation failed: {str(e)}]"
            messages.append(
                AIMessage(content=final_response_content)
            )
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

            json_ready_messages = serialize_messages(messages)
            log_id = save_to_json(json_ready_messages)
                
            log_input = ''
            if log_id:
                log_input = f"<a href='{APP_URL}documents/{log_id}'>Click here to view the Log input to LLM</a>"
            tokenresponse[session_id].update({
                'total_tokens': usage_metadata['total_tokens'],
                'input_tokens': usage_metadata['input_tokens'],
                'output_tokens': usage_metadata['output_tokens'],
                'response_type': response_type,
                'tools_used': tools_used,
                'source': chat_system,
                'run_id': run_id,
                'log_input': log_input
            })

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


def generate_run_id():
    """Generate a unique run ID for conversation tracking"""
    return f"run--{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:12]}"


def extract_metadata(soup, url):
    """Extract all metadata from the page"""
    metadata = {
        "title": "",
        "description": "",
        "keywords": "",
        "og_tags": {},
        "twitter_tags": {},
        "canonical": "",
        "robots": "",
        "author": "",
        "language": "",
        "charset": "",
        "viewport": "",
        "schema_org": []
    }
    
    # Basic metadata
    if soup.title:
        metadata["title"] = soup.title.string.strip() if soup.title.string else ""
    
    # Meta tags
    meta_tags = soup.find_all('meta')
    for tag in meta_tags:
        name = tag.get('name', '').lower()
        property_attr = tag.get('property', '').lower()
        content = tag.get('content', '')
        
        if name == 'description':
            metadata["description"] = content
        elif name == 'keywords':
            metadata["keywords"] = content
        elif name == 'robots':
            metadata["robots"] = content
        elif name == 'author':
            metadata["author"] = content
        elif name == 'language':
            metadata["language"] = content
        elif name == 'viewport':
            metadata["viewport"] = content
        elif tag.get('charset'):
            metadata["charset"] = tag.get('charset')
        elif property_attr.startswith('og:'):
            metadata["og_tags"][property_attr] = content
        elif name.startswith('twitter:'):
            metadata["twitter_tags"][name] = content
    
    # Canonical URL
    canonical = soup.find('link', {'rel': 'canonical'})
    if canonical:
        metadata["canonical"] = canonical.get('href', '')
    
    # Schema.org structured data
    schema_scripts = soup.find_all('script', {'type': 'application/ld+json'})
    for script in schema_scripts:
        try:
            schema_data = json.loads(script.string)
            metadata["schema_org"].append(schema_data)
        except:
            continue
    
    return metadata

def extract_headings(soup):
    """Extract all headings (H1-H6) from the page"""
    headings = {
        "h1": [],
        "h2": [],
        "h3": [],
        "h4": [],
        "h5": [],
        "h6": [],
        "heading_structure": []
    }
    
    for i in range(1, 7):
        heading_tags = soup.find_all(f'h{i}')
        headings[f'h{i}'] = [
            {
                "text": tag.get_text().strip(),
                "id": tag.get('id', ''),
                "class": tag.get('class', [])
            }
            for tag in heading_tags
        ]
    
    # Create hierarchical structure
    all_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in all_headings:
        headings["heading_structure"].append({
            "level": int(heading.name[1]),
            "tag": heading.name,
            "text": heading.get_text().strip(),
            "id": heading.get('id', ''),
            "class": heading.get('class', [])
        })
    
    return headings

def extract_faq_data(soup):
    """Extract FAQ data using multiple detection methods"""
    faq_data = []
    
    # Question-like headings followed by content
    question_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in question_headings:
        heading_text = heading.get_text().strip()
        
        # Check if heading looks like a question
        if any(keyword in heading_text.lower() for keyword in ['?', 'what', 'how', 'why', 'when', 'where', 'which', 'who']):
            # Get the next content element
            next_element = heading.find_next_sibling(['p', 'div', 'ul', 'ol'])
            if next_element:
                answer_text = next_element.get_text().strip()
                if len(answer_text) > 20:  # Filter out very short answers
                    faq_data.append({
                        "question": heading_text,
                        "answer": answer_text,
                        "source": "heading_pattern"
                    })
    
    # Remove duplicates based on question similarity
    unique_faqs = []
    seen_questions = set()
    
    for faq in faq_data:
        question_normalized = re.sub(r'[^\w\s]', '', faq['question'].lower()).strip()
        if question_normalized not in seen_questions and len(question_normalized) > 5:
            seen_questions.add(question_normalized)
            unique_faqs.append(faq)
    
    return unique_faqs

def extract_additional_seo_data(soup, url):
    """Extract additional SEO-relevant data"""
    additional_data = {
        "images": [],
        "links": {
            "internal": [],
            "external": []
        },
        "social_media_links": [],
        "contact_info": {},
        "performance_hints": {},
        "content_analysis": {}
    }
    
    parsed_url = urlparse(url)
    base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    
    # Extract images
    images = soup.find_all('img')
    for img in images[:10]:  # Limit to first 10 images
        img_data = {
            "src": urljoin(url, img.get('src', '')),
            "alt": img.get('alt', ''),
            "title": img.get('title', ''),
            "loading": img.get('loading', ''),
            "width": img.get('width', ''),
            "height": img.get('height', '')
        }
        additional_data["images"].append(img_data)
    
    # Extract links
    links = soup.find_all('a', href=True)
    for link in links:
        href = link.get('href')
        if href:
            full_url = urljoin(url, href)
            link_data = {
                "url": full_url,
                "text": link.get_text().strip(),
                "title": link.get('title', ''),
                "rel": link.get('rel', [])
            }
            
            if full_url.startswith(base_domain):
                additional_data["links"]["internal"].append(link_data)
            else:
                additional_data["links"]["external"].append(link_data)
    
    # Extract social media links
    social_patterns = ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'youtube.com', 'tiktok.com']
    for link in additional_data["links"]["external"]:
        if any(pattern in link["url"].lower() for pattern in social_patterns):
            additional_data["social_media_links"].append(link)
    
    # Content analysis
    text_content = soup.get_text()
    words = text_content.split()
    additional_data["content_analysis"] = {
        "word_count": len(words),
        "character_count": len(text_content),
        "readability_score": calculate_simple_readability(text_content)
    }
    
    return additional_data

def calculate_simple_readability(text):
    """Calculate a simple readability score"""
    sentences = text.count('.') + text.count('!') + text.count('?')
    words = len(text.split())
    
    if sentences == 0:
        return 0
    
    avg_words_per_sentence = words / sentences if sentences > 0 else 0
    
    # Simple readability score (lower is better)
    if avg_words_per_sentence <= 15:
        return "Easy"
    elif avg_words_per_sentence <= 20:
        return "Medium"
    else:
        return "Hard"

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

def is_valid_url(url: str) -> bool:
    """Check if URL is valid (not 404) and not excluded by extension.
       Skip URLs that contain query parameters (?)."""
    # Skip URL if it contains '?'
    if "?" in url:
        return False  

    # Get file extension (without query string)
    ext = os.path.splitext(url.split("?")[0])[1].lower()
    if ext in excluded_extensions:
        return False
    
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False
    
def pdfgenrateprocess(message: str) -> str:
    pdf_id = str(uuid.uuid4())
    pdf_filename = f"chat_output_{pdf_id}.pdf"
    pdf_path = os.path.join("documents", pdf_filename)
    os.makedirs("documents", exist_ok=True)

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>SEO + Semantic Guideline Validation Report</b>", styles["Heading1"]))
    story.append(Spacer(1, 12))
    
    # Split by your delimiter
    sections = message.split("++++ ++++ ++++ ++++ ++++")

    for i, section in enumerate(sections):
        if section.strip():
            # Split each section into lines
            lines = section.strip().split("\n")
            for line in lines:
                if line.strip():
                    story.append(Paragraph(line.strip(), styles["Normal"]))
                    story.append(Spacer(1, 6))  # small gap between lines

            # Add page break after each section except the last
            if i < len(sections) - 1:
                story.append(PageBreak())

    doc.build(story)
    file_size = os.path.getsize(pdf_path)   # in bytes
    file_ext  = os.path.splitext(pdf_filename)[1].lstrip('.')  # pdf
    mime_type, _ = mimetypes.guess_type(pdf_path)  # 'application/pdf'

    # Save to DB (Document model)
    Document.objects.create(
        document_id=pdf_id,
        original_name=pdf_filename,
        file_name=pdf_filename,
        file_size=file_size,
        file_type=file_ext,
        mime_type=mime_type or "application/octet-stream",
        file_path=pdf_path,
        content="",  # or extracted text if you have it
        upload_date=datetime.now().isoformat(),
        url=f"/{pdf_filename}"  # or file URL if you want to store one
    )
    return pdf_id

# Add this new function to handle runtime status messages
def yield_runtime_status(message, delay=0.1):
    """Yield runtime status messages with optional delay"""
    time.sleep(delay)
    return f"🔄 {message}\n\n"

def save_to_json(data: dict) -> str:
    json_id = str(uuid.uuid4())
    file_name = f"request_data_{json_id}.json"
    file_path = os.path.join("documents", file_name)

    # Ensure folder exists
    os.makedirs("documents", exist_ok=True)

    # Save dict → JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


    file_size = os.path.getsize(file_path)   # in bytes
    file_ext  = os.path.splitext(file_name)[1].lstrip('.')  # pdf
    mime_type, _ = mimetypes.guess_type(file_path)  # 'application/pdf'
    # Save to DB (Document model)
    Document.objects.create(
        document_id=json_id,
        original_name=file_name,
        file_name=file_name,
        file_size=file_size,
        file_type=file_ext,
        mime_type=mime_type or "application/json",
        file_path=file_path,
        content="",  # or extracted text if you have it
        upload_date=datetime.now().isoformat(),
        url=f"/{file_name}"  # or file URL if you want to store one
    )


    return json_id

def is_valid_json(data: str) -> bool:
    try:
        json.loads(data)
        return True
    except (ValueError, TypeError):
        return False

def serialize_messages(messages):
    """
    Convert LangChain messages into proper JSON format.
    Handles dict-like strings and ensures content is always valid.
    """
    json_ready = []

    for msg in messages:
        # ---- role mapping ----
        if isinstance(msg, HumanMessage):
            role = "human"
        elif isinstance(msg, AIMessage):
            role = "ai"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "tool"

        # ---- initialize content safely ----
        content = msg.content if hasattr(msg, "content") else ""

        # ---- try JSON parsing ----
        if isinstance(content, str):
            parsed = None
            try:
                # 1️⃣ Try JSON first
                parsed = json.loads(content)
            except Exception:
                try:
                    # 2️⃣ Try Python dict string (single quotes → dict)
                    parsed = ast.literal_eval(content)
                except Exception:
                    pass

            if isinstance(parsed, (dict, list)):
                content = parsed   # keep JSON object
            else:
                content = content  # leave as string

        json_ready.append({
            "role": role,
            "content": content
        })

    return json_ready


# -------------------------------
# Enhanced RAG functions
# -------------------------------
def get_relevant_documents(query: str, llm_metadata: dict, session_id: str = None, n_results: int = 4) -> List[Dict[str, Any]]:
    """Get relevant documents for RAG using semantic search with chapter information."""
    try:

        # Fix: Access dictionary keys instead of Pydantic model attributes
        primary_topic = llm_metadata.get("primary_topic")
        source_brand = llm_metadata.get("source_brand")

        # Add validation to ensure required fields exist
        if not primary_topic or not source_brand:
            print(f"Warning: Missing required metadata fields. primary_topic: {primary_topic}, source_brand: {source_brand}")
            # Fallback to search without filters if metadata is incomplete
            results = vectorstore.similarity_search(query, k=n_results)
        else:
            # Use proper Chroma filter syntax with $and operator
            # filter_condition = {
            #     "$or": [
            #         {"source_brand": {"$eq": source_brand}},
            #         {"primary_topic": {"$eq": primary_topic}}
            #     ]
            # }

            filter_condition = {"primary_topic": {"$eq": primary_topic}}
            
            try:
                results = vectorstore.similarity_search(
                    query, 
                    k=n_results,
                    filter=filter_condition
                )
            except Exception as filter_error:
                print(f"Filter search failed: {filter_error}")
                # Fallback to unfiltered search
                results = vectorstore.similarity_search(query, k=n_results)

        relevant_docs = []
        seen_content = set()  # Track seen content to avoid duplicates
        seen_combinations = set()  # Track document_id + chunk_index combinations
        
        for doc in results:
            metadata = doc.metadata
            content = doc.page_content.strip()
            
            # Create unique identifiers for deduplication
            content_hash = hash(content)
            doc_chunk_combo = f"{metadata.get('document_id', '')}_{metadata.get('chunk_index', 0)}"
            
            # # Skip if we've seen this exact content or document chunk combination
            # if content_hash in seen_content or doc_chunk_combo in seen_combinations:
            #     continue
                
            # # Skip very short or empty content
            # if len(content) < 50:
            #     continue
                
            # Add to tracking sets
            seen_content.add(content_hash)
            seen_combinations.add(doc_chunk_combo)
            
            relevant_docs.append({
                "content": content,
                "document_id": metadata.get("document_id", ""),
                "source_brand": metadata.get("source_brand", ""),
                "primary_topic": metadata.get("primary_topic", ""),
                "content_type": metadata.get("content_type", ""),
                "specific_element": metadata.get("specific_element", ""),
            })

        return relevant_docs
        
    except Exception as e:
        print(f"Error retrieving relevant documents: {str(e)}")
        return []

def generate_chunk_label(content: str) -> str:
    
    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=OPENAIKEY,
        model=MODEL
    )

    # 1. Get the schema definition from your Pydantic model
    schema_definition = json.dumps(SeoAuditMetadata.model_json_schema(), indent=2)

    # 2. The prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert SEO analyst and a meticulous data processor. Your task is to analyze a given text chunk and extract structured metadata from it.
        
        You must strictly adhere to the following JSON schema. Do not add any fields that are not in the schema. Ensure the values you provide for 'primary_topic', 'content_type', and 'source_brand' are ONLY from the allowed 'Literal' options.

        Respond ONLY with a single, valid JSON object. Do not include any other text, explanations, or apologies in your response.

        JSON Schema:
        {pydantic_schema}
        """),
        ("human", """
        Please analyze the following text chunk and generate the corresponding JSON object.

        --- TEXT CHUNK ---
        {chunk_text}
        --- END TEXT CHUNK ---
        """)
    ])

    # 3. Create the chain (assuming 'llm' is your initialized ChatOpenAI model)
    extraction_chain = prompt | llm.with_structured_output(SeoAuditMetadata)

    # The .invoke method will correctly format the prompt with these variables
    generated_metadata = extraction_chain.invoke({
        "chunk_text": content,
        "pydantic_schema": schema_definition
    })

    return generated_metadata.dict()
