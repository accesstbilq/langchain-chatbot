# =============================
# Django & Library Imports
# =============================
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
from dotenv import load_dotenv
from datetime import datetime
import mimetypes
import uuid
from .models import ChatSession, ChatMessage

# -------------------------------
# Global variables
# -------------------------------
chat_system = ""     # system prompt storage
messages = []        # in-memory messages list


# -------------------------------
# Environment setup
# -------------------------------
load_dotenv()
OPENAIKEY = os.getenv('OPEN_AI_KEY')
MODEL = "gpt-3.5-turbo"


# -------------------------------
# Document storage configuration
# -------------------------------
UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'documents')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# In-memory document store (use DB in production)
uploaded_documents = {}


# -------------------------------
# Chat history view
# -------------------------------
def chathistory(request):
    """Render chat history page with all sessions and last session messages."""
    sessions, last_session_id = load_chat_session() or ([], None)

    # if no last session, set messages empty
    if last_session_id:
        messages_qs = load_chat_history_from_db(last_session_id) or []
    else:
        messages_qs = []

    return render(request, 'chathistory.html', {
        'sessions': sessions,
        'last_session_id': last_session_id,
        'messages': messages_qs,
    })

# -------------------------------
# Document view
# -------------------------------
def document_view(request):
    """Render document upload page."""
    print(uploaded_documents)
    return render(request, 'document.html')


# -------------------------------
# Upload documents
# -------------------------------
@csrf_exempt
def upload_documents(request):
    """Handle multiple document uploads with validation and metadata storage."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests allowed'})
    
    if 'documents' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No documents provided'})
    
    uploaded_files = []
    errors = []
    
    # Allowed file extensions
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
            
            # Save file on disk
            fs = FileSystemStorage(location=UPLOAD_DIR)
            filename = fs.save(file_name, file)
            file_path = fs.path(filename)
            
            # Store metadata
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
            
            # Prepare response entry
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


# -------------------------------
# List uploaded documents
# -------------------------------
@csrf_exempt
def list_documents(request):
    """Return list of uploaded documents (sorted by newest first)."""
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


# -------------------------------
# Delete document
# -------------------------------
@csrf_exempt
def delete_document(request):
    """Delete an uploaded document from memory + disk."""
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


# -------------------------------
# Chat session helpers
# -------------------------------
def load_chat_session():
    """Fetch all chat sessions and return with first session id (if exists)."""
    sessions = ChatSession.objects.all().order_by("-created_at")
    if sessions.exists():
        return sessions, sessions.first().session_id
    return [], None   # always return tuple

# -------------------------------
# Chat History helpers
# -------------------------------
def load_chat_history(request):
    """Load messages for a given session id (used for chat UI)."""
    session_id = request.GET.get("session_id")
    messages = load_chat_history_from_db(session_id)
    return render(request, "chat_messages.html", {"messages": messages})

# -------------------------------
# Load Message using session id
# -------------------------------
def load_chat_history_from_db(session_id):
    """Fetch chat history (human + ai messages only) from DB."""
    if not session_id:
        return []
    session = ChatSession.objects.get(session_id=session_id)

    # Filter messages with non-empty content
    messages = ChatMessage.objects.filter(
        session=session,
        message_type__in=['human', 'ai']  # only include human/ai
    ).exclude(content__isnull=True).exclude(content__exact='').order_by('-created_at')
    
    return messages
