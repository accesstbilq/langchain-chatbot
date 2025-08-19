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
from .models import ChatSession, ChatMessage

from bs4 import BeautifulSoup

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, BaseMessage
from langchain_core.messages import ToolMessage,HumanMessage,AIMessageChunk
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from openai import OpenAIError, APITimeoutError

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
tokenresponse = {}  # Store chat history by session
messagedata = {}  # Store chat history by session
count = 0
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


@csrf_exempt
def chatbot_input(request):
    """SEO-focused chatbot with arithmetic tool support and chat history"""
    global chat_system, chat_history,tokenresponse,count

    if not hasattr(request, 'POST') or request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests are allowed'})
    
    data = json.loads(request.body.decode('utf-8'))
    usermessage = data.get("message", "")
    session_id = data.get("session_id", "default")  # Get session ID from frontend

    runid = generate_run_id()

    if not usermessage:
        return JsonResponse({'success': False, 'error': 'No message provided'})

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
        
        # Add the new user message
        messages.append(HumanMessage(content=usermessage))


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
                "validate_and_fetch_url": validate_and_fetch_url
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
                            ToolMessage(content=f"⚠ Error running tool {tool_name}: {str(e)}", tool_call_id=tool_call_id)
                        )

                        messagedata[session_id].append({
                            "message_type": "tool",
                            "content": f"⚠ Error running tool {tool_name}: {str(e)}",
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
                build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system,count),
                content_type='text/plain'
            )

        else:
            # No tools called, store conversation and stream response
            chat_history[session_id].append(HumanMessage(content=usermessage))
            
            return StreamingHttpResponse(
                build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system,count),
                content_type='text/plain'
            )

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({
            'success': False,
            'error': f'Processing error: {str(e)}'
        })
    
def build_stream_response(llm_with_tools, messages, ai_msg, session_id, chat_system,count):
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
            session, create = get_or_create_chat_session(session_id,run_id)

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

            save_message_to_db(session,messagedata,run_id,count)

            

            

    except (OpenAIError, APITimeoutError) as e:
        yield f"\n[Error occurred: {str(e)}]"

def get_or_create_chat_session(session_id, run_id=None):
    """Get existing chat session or create new one"""
    try:
        session, created = ChatSession.objects.get_or_create(
            session_id=session_id,
            defaults={
                'run_id': run_id,
                'is_active': True
            }
        )
        
        # Update run_id if provided and different
        if run_id and session.run_id != run_id:
            session.run_id = run_id
            session.save(update_fields=['run_id', 'updated_at'])
            
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

def generate_message_id():
    """Generate a unique message ID"""
    return f"msg--{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:12]}"

def generate_run_id():
    """Generate a unique run ID for conversation tracking"""
    return f"run--{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:12]}"

def load_chat_history_from_db(session_id):
    """Load chat history from database and convert to LangChain message format"""
    try:
        session = ChatSession.objects.get(session_id=session_id)
        messages = ChatMessage.objects.filter(session=session).order_by('order')
        
        langchain_messages = []
        
        for msg in messages:
            if msg.message_type == 'system':
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.message_type == 'human':
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.message_type == 'ai':
                ai_message = AIMessage(content=msg.content)
                # If there are tool calls, add them to the AI message
                if msg.tool_calls:
                    ai_message.tool_calls = msg.tool_calls
                langchain_messages.append(ai_message)
            elif msg.message_type == 'tool':
                langchain_messages.append(
                    ToolMessage(
                        content=msg.content, 
                        tool_call_id=msg.tool_call_id
                    )
                )
        
        return langchain_messages
    except ChatSession.DoesNotExist:
        return None
    except Exception as e:
        print(f"Error loading chat history: {str(e)}")
        return None