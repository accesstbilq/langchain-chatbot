from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import validators
import requests, traceback
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import mimetypes
import uuid
from django.http import StreamingHttpResponse

from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, BaseMessage
from langchain_core.messages import ToolMessage,HumanMessage,AIMessageChunk
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from openai import OpenAIError, APITimeoutError

# Import database utilities
from .chat_utils import (
    get_or_create_chat_session, 
    save_message_to_db, 
    load_chat_history_from_db,
    generate_run_id,
    generate_message_id,
    count_tokens,
    count_tokens_from_string,
    save_token_usage,
    get_token_usage
)


# Global variables for chat system and messages
chat_system = ""
messages = []

# Load ENV file
load_dotenv()
OPENAIKEY = os.getenv('OPEN_AI_KEY')

MODEL = "gpt-3.5-turbo"

# Document storage configuration
UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'documents')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# In-memory document store (in production, use database)
uploaded_documents = {}


system_message = """
You are an expert SEO assistant with document analysis capabilities. 
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
1. Only answer SEO-related questions in detail.
2. If the user asks a question unrelated to SEO, politely say you specialize in SEO and cannot answer other topics.
3. If the user clearly asks for a basic arithmetic calculation (e.g., multiplication, addition, subtraction, division), call the appropriate calculation tool.
4. If the user provides or asks to validate a URL, call the `validate_and_fetch_url` tool to check its validity and fetch its title.
5. You can analyze uploaded documents for SEO-related insights when asked.
6. Always provide actionable, practical SEO recommendations with clear steps.
"""

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

def chatbot_view(request):
    return render(request, 'chatbot.html')

def document_view(request):
    return render(request, 'document.html')

def dashboard_view(request):
    return render(request, 'dashboard.html')

@csrf_exempt
def upload_documents(request):
    """Handle multiple document uploads"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests allowed'})
    
    if 'documents' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No documents provided'})
    
    uploaded_files = []
    errors = []
    
    # Allowed file types
    ALLOWED_EXTENSIONS = {
        'txt', 'pdf', 'doc', 'docx', 'html', 'css', 'js', 
        'json', 'xml', 'csv', 'xlsx', 'ppt', 'pptx'
    }
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    files = request.FILES.getlist('documents')
    
    for file in files:
        try:
            # Validate file size
            if file.size > MAX_FILE_SIZE:
                errors.append(f"File '{file.name}' is too large (max 10MB)")
                continue
            
            # Validate file extension
            file_extension = file.name.split('.')[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                errors.append(f"File type '{file_extension}' not allowed for '{file.name}'")
                continue
            
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            file_name = f"{unique_id}_{file.name}"
            
            # Save file
            fs = FileSystemStorage(location=UPLOAD_DIR)
            filename = fs.save(file_name, file)
            file_path = fs.path(filename)
            
            # Store document metadata
            doc_info = {
                'id': unique_id,
                'original_name': file.name,
                'file_name': filename,
                'file_path': file_path,
                'size': file.size,
                'file_type': file_extension,
                'mime_type': mimetypes.guess_type(file.name)[0] or 'application/octet-stream',
                'upload_date': datetime.now().isoformat(),
                'url': fs.url(filename) if hasattr(fs, 'url') else None
            }
            
            uploaded_documents[unique_id] = doc_info






            uploaded_files.append({
                'id': unique_id,
                'name': file.name,
                'size': file.size,
                'type': file_extension,
                'upload_date': doc_info['upload_date']
            })
            
        except Exception as e:
            errors.append(f"Error uploading '{file.name}': {str(e)}")
    
    return JsonResponse({
        'success': len(uploaded_files) > 0,
        'uploaded_files': uploaded_files,
        'errors': errors,
        'total_uploaded': len(uploaded_files),
        'total_errors': len(errors)
    })

@csrf_exempt
def list_documents(request):
    """Return list of uploaded documents"""
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Only GET requests allowed'})
    
    documents_list = []
    for doc_id, doc_info in uploaded_documents.items():
        documents_list.append({
            'id': doc_id,
            'name': doc_info['original_name'],
            'size': doc_info['size'],
            'type': doc_info['file_type'],
            'upload_date': doc_info['upload_date'],
            'mime_type': doc_info['mime_type']
        })
    
    # Sort by upload date (newest first)
    documents_list.sort(key=lambda x: x['upload_date'], reverse=True)
    
    return JsonResponse({
        'success': True,
        'documents': documents_list,
        'total_count': len(documents_list)
    })

@csrf_exempt
def delete_document(request):
    """Delete an uploaded document"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests allowed'})
    
    document_id = request.POST.get('document_id', '').strip()
    if not document_id:
        return JsonResponse({'success': False, 'error': 'Document ID required'})
    
    if document_id not in uploaded_documents:
        return JsonResponse({'success': False, 'error': 'Document not found'})
    
    try:
        doc_info = uploaded_documents[document_id]
        
        # Delete physical file
        if os.path.exists(doc_info['file_path']):
            os.remove(doc_info['file_path'])
        
        # Remove from memory store
        del uploaded_documents[document_id]
        
        return JsonResponse({
            'success': True,
            'message': f"Document '{doc_info['original_name']}' deleted successfully"
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f"Error deleting document: {str(e)}"
        })
    

# Add this at the top with other global variables
chat_history = {}  # Store chat history by session

def get_token_usage_view(request, run_id):
    token_usage = get_token_usage(run_id=run_id)
    return JsonResponse({
        'response_type': token_usage.response_type,
        'tools_used': token_usage.tools_used,
        'source': token_usage.source,
        'message_id': token_usage.message_id,
        'run_id': token_usage.run_id,
        'session': token_usage.session.session_id,
        'input_tokens': token_usage.input_tokens,
        'output_tokens': token_usage.output_tokens,
        'total_tokens': token_usage.total_tokens
    })

@csrf_exempt
def chatbot_input(request):
    """SEO-focused chatbot with database-backed chat history"""
    global chat_system

    if not hasattr(request, 'POST') or request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})
    
    data = json.loads(request.body.decode('utf-8'))
    usermessage = data.get("message", "")
    session_id = data.get("session_id", "default")
    run_id = data.get("run_id", None)  # Optional run_id from frontend
    user = request.user if request.user.is_authenticated else None
    
    if not usermessage:
        return JsonResponse({'success': False, 'error': 'No message provided'})

    # Track token usage for streaming response
    output_token_count = 0
    try:
        # SEO + arithmetic tools
        tools = [multiply, validate_and_fetch_url]

        llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=OPENAIKEY,
            model=MODEL
        )
        llm_with_tools = llm.bind_tools(tools)
        chat_system = "LLM Call"
        
        # Get or create chat session in database
        session, created = get_or_create_chat_session(session_id, user, run_id)
        
        # Generate new run_id if not provided
        current_run_id = run_id or session.run_id or generate_run_id()
        
        # Load existing chat history from database
        messages = load_chat_history_from_db(session_id)
        
        # If no messages exist or this is a new session, start with system message
        if not messages or created:
            system_msg = SystemMessage(content=system_message)
            save_message_to_db(
                session, 
                'system', 
                system_message, 
                run_id=current_run_id,
                message_id=generate_message_id()
            )
            messages = [system_msg]
        
        # Add the new user message
        user_msg = HumanMessage(content=usermessage)
        messages.append(user_msg)

        # Count input tokens
        input_tokens = count_tokens(messages)

        save_message_to_db(
            session, 
            'human', 
            usermessage, 
            run_id=current_run_id,
            message_id=generate_message_id()
        )

        # Get initial AI response
        ai_msg = llm_with_tools.invoke(messages)
        
        

        # Process tool calls if any
        if ai_msg.tool_calls:
            messages.append(ai_msg)
            
            # Save AI message with tool calls to database
            save_message_to_db(
                session, 
                'ai', 
                ai_msg.content, 
                tool_calls=ai_msg.tool_calls,
                run_id=current_run_id,
                message_id=generate_message_id()
            )
            
            # Process tool calls...
            tool_mapping = {
                "multiply": multiply,
                "validate_and_fetch_url": validate_and_fetch_url
            }

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                print(f"Processing tool call: {tool_name} with args: {tool_args}")
                
                if tool_name in tool_mapping:
                    selected_tool = tool_mapping[tool_name]
                    try:
                        # Run the tool with args
                        tool_result = selected_tool.invoke(tool_args)
                        print(f"Tool result: {tool_result}")
                        
                        # Append ToolMessage with matching tool_call_id
                        tool_message = ToolMessage(
                            content=str(tool_result), 
                            tool_call_id=tool_call_id
                        )
                        messages.append(tool_message)
                        
                        # Save tool message to database
                        save_message_to_db(
                            session, 
                            'tool', 
                            str(tool_result), 
                            tool_call_id=tool_call_id,
                            run_id=current_run_id,
                            message_id=generate_message_id()
                        )
                        
                    except Exception as e:
                        print(f"Error running tool {tool_name}: {str(e)}")
                        error_content = f"⚠ Error running tool {tool_name}: {str(e)}"
                        error_message = ToolMessage(
                            content=error_content, 
                            tool_call_id=tool_call_id
                        )
                        messages.append(error_message)
                        
                        # Save error message to database
                        save_message_to_db(
                            session, 
                            'tool', 
                            error_content, 
                            tool_call_id=tool_call_id,
                            run_id=current_run_id,
                            message_id=generate_message_id()
                        )
                else:
                    error_content = f"Unknown tool: {tool_name}"
                    error_message = ToolMessage(
                        content=error_content, 
                        tool_call_id=tool_call_id
                    )
                    messages.append(error_message)
                    
                    # Save error message to database
                    save_message_to_db(
                        session, 
                        'tool', 
                        error_content, 
                        tool_call_id=tool_call_id,
                        run_id=current_run_id,
                        message_id=generate_message_id()
                    )
            
            def stream_response():
                nonlocal output_token_count
                try:
                    final_response_content = ""
                    for chunk in llm_with_tools.stream(messages):
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            final_response_content += chunk.content
                            # Count tokens in this chunk
                            chunk_tokens = count_tokens_from_string(chunk.content)
                            output_token_count += chunk_tokens
                            yield chunk.content
                    
                    # Save the final AI response to database
                    if final_response_content:
                        savemessage = save_message_to_db(
                            session, 
                            'ai', 
                            final_response_content,
                            run_id=current_run_id,
                            message_id=generate_message_id()
                        )
                        # Save token usage
                        total_tokens = input_tokens + output_token_count
                        response_type='Tool Usage' if ai_msg.tool_calls else 'General Conversation'
                        tools_used=len(ai_msg.tool_calls) > 0 if ai_msg.tool_calls else False
                        source=chat_system
                        save_token_usage(session, current_run_id, input_tokens, output_token_count, total_tokens,response_type,tools_used,source,savemessage)
                except (OpenAIError, APITimeoutError) as e:
                    error_msg = f"\n[Error occurred: {str(e)}]"
                    save_message_to_db(
                        session, 
                        'ai', 
                        error_msg,
                        run_id=current_run_id,
                        message_id=generate_message_id()
                    )
                    yield error_msg

            streaming_response =  StreamingHttpResponse(stream_response(), content_type='text/plain')
            streaming_response['X-Run-ID'] = current_run_id  # Add as header
            return streaming_response

        else:
            # No tools called, stream response directly
            def stream_response():
                try:
                    nonlocal output_token_count
                    final_response_content = ""
                    for chunk in llm_with_tools.stream(messages):
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            final_response_content += chunk.content
                            # Count tokens in this chunk
                            chunk_tokens = count_tokens_from_string(chunk.content)
                            output_token_count += chunk_tokens
                            yield chunk.content
                    
                    # Save the final AI response to database
                    if final_response_content:
                        savemessage = save_message_to_db(
                            session, 
                            'ai', 
                            final_response_content,
                            run_id=current_run_id,
                            message_id=generate_message_id()
                        )
                        # Save token usage
                        total_tokens = input_tokens + output_token_count

                        response_type='Tool Usage' if ai_msg.tool_calls else 'General Conversation'
                        tools_used=len(ai_msg.tool_calls) > 0 if ai_msg.tool_calls else False
                        source=chat_system
                        save_token_usage(session, current_run_id, input_tokens, output_token_count, total_tokens,response_type,tools_used,source,savemessage)

                except (OpenAIError, APITimeoutError) as e:
                    error_msg = f"\n[Error occurred: {str(e)}]"
                    save_message_to_db(
                        session, 
                        'ai', 
                        error_msg,
                        run_id=current_run_id,
                        message_id=generate_message_id()
                    )
                    yield error_msg

            streaming_response =  StreamingHttpResponse(stream_response(), content_type='text/plain')
            streaming_response['X-Run-ID'] = current_run_id  # Add as header
            return streaming_response

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Processing error: {str(e)}'
        })