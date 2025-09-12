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
from .models import ChatSession, ChatMessage, Document
from django.http import FileResponse, Http404
import chromadb
from chromadb.config import Settings
import openai
from typing import List, Dict, Any
import json
import PyPDF2
import docx
import csv
import openpyxl
from pptx import Presentation
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from django.utils import timezone
import os
import re

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    CharacterTextSplitter
)
from langchain.docstore.document import Document as LangchainDocument

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
MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize OpenAI client
openai.api_key = OPENAIKEY

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

# -------------------------------
# Document storage configuration
# -------------------------------
UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'documents')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# -------------------------------
# Text extraction functions
# -------------------------------
def extract_text_from_file(file_path: str, file_type: str) -> str:
    """Extract text content from various file types."""
    try:
        if file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_type == 'pdf':
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    # Add page markers for better splitting
                    text += f"\n\n--- Page {page_num + 1} ---\n\n" + page_text + "\n"
            return text
        
        elif file_type in ['doc', 'docx']:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                # Preserve heading structure for chapter detection
                if paragraph.style.name.startswith('Heading'):
                    text += f"\n\n# {paragraph.text}\n\n"
                else:
                    text += paragraph.text + "\n"
            return text
        
        elif file_type == 'html':
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        
        elif file_type in ['css', 'js', 'json', 'xml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_type == 'csv':
            text = ""
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    text += ", ".join(row) + "\n"
            return text
        
        elif file_type == 'xlsx':
            workbook = openpyxl.load_workbook(file_path)
            text = ""
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"\n\n# Sheet: {sheet_name}\n\n"
                for row in sheet.iter_rows(values_only=True):
                    if row:
                        text += ", ".join([str(cell) if cell is not None else "" for cell in row]) + "\n"
            return text
        
        elif file_type in ['ppt', 'pptx']:
            prs = Presentation(file_path)
            text = ""
            for slide_num, slide in enumerate(prs.slides):
                text += f"\n\n# Slide {slide_num + 1}\n\n"
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        
        else:
            return ""
    
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        return ""

# -------------------------------
# Enhanced Chapter-wise Text Splitting Functions
# -------------------------------

def detect_chapter_patterns(text: str, file_type: str) -> List[str]:
    """Detect chapter/section patterns based on file type and content."""
    patterns = []
    
    # Common chapter patterns
    common_patterns = [
        r'^(Chapter\s+\d+[:\.\s])',
        r'^(CHAPTER\s+\d+[:\.\s])',
        r'^(Ch\.\s*\d+[:\.\s])',
        r'^(\d+\.\s+)',
        r'^(Part\s+\d+[:\.\s])',
        r'^(Section\s+\d+[:\.\s])',
        r'^(Unit\s+\d+[:\.\s])',
    ]
    
    # Markdown-style headers
    markdown_patterns = [
        r'^(#{1,6}\s+)',  # # Header, ## Header, etc.
    ]
    
    # HTML headers (if HTML content)
    if file_type == 'html':
        patterns.extend([
            r'<h[1-6][^>]*>',
        ])
    else:
        patterns.extend(common_patterns + markdown_patterns)
    
    return patterns

def split_text_by_chapters(text: str, file_type: str, doc_metadata: Dict) -> List[LangchainDocument]:
    """Split text into chapters/sections using LangChain text splitters."""
    documents = []
    
    try:
        # For HTML files, use HTMLHeaderTextSplitter
        if file_type == 'html':
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ]
            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            html_header_splits = html_splitter.split_text(text)
            
            for i, doc in enumerate(html_header_splits):
                metadata = doc_metadata.copy()
                metadata.update({
                    'chunk_type': 'html_section',
                    'chunk_index': i,
                    'section_header': doc.metadata.get('Header 1', '') or doc.metadata.get('Header 2', ''),
                })
                doc.metadata = metadata
                documents.append(doc)
        
        # For Markdown-like content, try MarkdownHeaderTextSplitter
        elif '# ' in text or '## ' in text:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False
            )
            md_header_splits = markdown_splitter.split_text(text)
            
            for i, doc in enumerate(md_header_splits):
                metadata = doc_metadata.copy()
                metadata.update({
                    'chunk_type': 'markdown_section',
                    'chunk_index': i,
                    'section_header': doc.metadata.get('Header 1', '') or doc.metadata.get('Header 2', ''),
                })
                doc.metadata = metadata
                documents.append(doc)
        
        # For other files, try chapter pattern detection
        else:
            chapter_splits = split_by_chapter_patterns(text, file_type, doc_metadata)
            if len(chapter_splits) > 1:
                documents.extend(chapter_splits)
            else:
                # Fall back to recursive character splitting
                documents.extend(fallback_recursive_split(text, doc_metadata))
        
        # If no documents were created, use fallback
        if not documents:
            documents.extend(fallback_recursive_split(text, doc_metadata))
            
    except Exception as e:
        print(f"Error in chapter splitting: {str(e)}")
        documents.extend(fallback_recursive_split(text, doc_metadata))
    
    return documents

def split_by_chapter_patterns(text: str, file_type: str, doc_metadata: Dict) -> List[LangchainDocument]:
    """Split text by detected chapter patterns."""
    documents = []
    patterns = detect_chapter_patterns(text, file_type)
    
    # Try each pattern to find the best split
    for pattern in patterns:
        try:
            # Find all matches
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            
            if len(matches) >= 2:  # At least 2 chapters found
                for i, match in enumerate(matches):
                    start_pos = match.start()
                    end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    
                    chapter_text = text[start_pos:end_pos].strip()
                    if len(chapter_text) > 100:  # Minimum chapter length
                        
                        # Extract chapter title
                        chapter_title = match.group(1).strip() if match.groups() else f"Section {i+1}"
                        
                        metadata = doc_metadata.copy()
                        metadata.update({
                            'chunk_type': 'chapter',
                            'chunk_index': i,
                            'chapter_title': chapter_title,
                            'chapter_number': i + 1,
                        })
                        
                        doc = LangchainDocument(
                            page_content=chapter_text,
                            metadata=metadata
                        )
                        documents.append(doc)
                
                if documents:
                    return documents  # Return if successful split found
                    
        except Exception as e:
            print(f"Error with pattern {pattern}: {str(e)}")
            continue
    
    return documents

def fallback_recursive_split(text: str, doc_metadata: Dict) -> List[LangchainDocument]:
    """Fallback to recursive character text splitting."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        splits = text_splitter.split_text(text)
        documents = []
        
        for i, split in enumerate(splits):
            metadata = doc_metadata.copy()
            metadata.update({
                'chunk_type': 'recursive_split',
                'chunk_index': i,
                'chunk_size': len(split),
            })
            
            doc = LangchainDocument(
                page_content=split,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        print(f"Error in fallback splitting: {str(e)}")
        # Last resort: return single document
        metadata = doc_metadata.copy()
        metadata.update({
            'chunk_type': 'single_document',
            'chunk_index': 0,
        })
        
        return [LangchainDocument(page_content=text, metadata=metadata)]

def store_document_embeddings(doc_dict: dict, text_content: str):
    """Store document embeddings in Chroma DB with chapter-wise splitting."""
    try:
        # Prepare base metadata
        base_metadata = {
            "document_id": doc_dict.get("document_id", ""),
            "session_id": doc_dict.get("session_id", ""),
            "original_name": doc_dict.get("original_name", ""),
            "file_name": doc_dict.get("file_name", ""),
            "file_size": int(doc_dict.get("file_size", 0)),
            "file_type": doc_dict.get("file_type", ""),
            "mime_type": doc_dict.get("mime_type", "application/octet-stream"),
            "file_path": doc_dict.get("file_path", ""),
            "url": doc_dict.get("url", ""),
            "upload_date": timezone.now().isoformat(),
        }
        
        # Split text by chapters/sections
        documents = split_text_by_chapters(text_content, doc_dict.get("file_type", ""), base_metadata)
        
        if not documents:
            return False
        
        # Prepare data for Chroma
        texts = []
        metadatas = []
        ids = []
        
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(f"{doc_dict.get('document_id')}_{doc.metadata.get('chunk_index', 0)}")
        
        # Add to Chroma via LangChain
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully stored {len(documents)} chunks for document {doc_dict.get('original_name')}")
        return True
        
    except Exception as e:
        print(f"Error storing embeddings for document {doc_dict.get('document_id')}: {str(e)}")
        return False

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
    return render(request, 'document.html')

# -------------------------------
# Upload documents with Chroma integration
# -------------------------------
@csrf_exempt
def upload_documents(request):
    """Handle multiple document uploads with validation, metadata storage, and embedding generation."""
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
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 10MB
    
    files = request.FILES.getlist('documents')
    session_id = request.POST.get('session_id')  # Optional session association
    
    # Get session object if provided
    session = None
    if session_id:
        try:
            session = ChatSession.objects.get(session_id=session_id)
        except ChatSession.DoesNotExist:
            pass
    
    for file in files:
        try:
            # Validate file size
            if file.size > MAX_FILE_SIZE:
                errors.append(f"File '{file.name}' is too large (max 50MB)")
                continue
            
            # Validate file extension
            file_extension = file.name.split('.')[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS:
                errors.append(f"File type '{file_extension}' not allowed for '{file.name}'")
                continue
            
            # Generate unique filename
            unique_id = uuid.uuid4()
            file_name = f"{unique_id}_{file.name}"
            
            # Save file on disk
            fs = FileSystemStorage(location=UPLOAD_DIR)
            filename = fs.save(file_name, file)
            file_path = fs.path(filename)
            
            # Extract text content
            text_content = extract_text_from_file(file_path, file_extension)
            
            # Create Document model instance
            doc_dict = {
                "document_id": str(unique_id),
                "session_id": str(session.session_id) if session else None,
                "original_name": file.name,
                "file_name": filename,
                "file_size": file.size,
                "file_type": file_extension,
                "mime_type": mimetypes.guess_type(file.name)[0] or "application/octet-stream",
                "content": text_content[:10000],  # first 10k chars
                "file_path": file_path,
                "url": fs.url(filename) if hasattr(fs, "url") else None,
            }

            # Generate and store embeddings in Chroma (with chapter-wise splitting)
            embedding_success = store_document_embeddings(doc_dict, text_content)
            
            # Prepare response entry
            uploaded_files.append({
                'id': unique_id,
                'name': file.name,
                'size': file.size,
                'type': file_extension,
                'upload_date': datetime.now().isoformat(),
                'embedding_success': embedding_success,
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
    """Return list of uploaded documents from Chroma with chapter information."""
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Only GET requests allowed'})
    
    try:
        # Fetch all documents metadata from Chroma
        all_docs = vectorstore._collection.get(include=["metadatas"])   

        # Dict to keep only one entry per document_id with chapter count
        unique_docs = {}

        for metadata in all_docs.get("metadatas", []):
            if metadata:
                doc_id = metadata.get("document_id")
                if doc_id:
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = {
                            "id": doc_id,
                            "name": metadata.get("original_name", ""),
                            "size": metadata.get("file_size", 0),
                            "type": metadata.get("file_type", ""),
                            "upload_date": metadata.get("upload_date", ""),
                            "mime_type": metadata.get("mime_type", "application/octet-stream"),
                            "chunk_count": 0,
                            "chunk_types": set(),
                        }
                    
                    # Count chunks and track chunk types
                    unique_docs[doc_id]["chunk_count"] += 1
                    chunk_type = metadata.get("chunk_type", "unknown")
                    unique_docs[doc_id]["chunk_types"].add(chunk_type)

        # Convert set to list for JSON serialization
        for doc in unique_docs.values():
            doc["chunk_types"] = list(doc["chunk_types"])

        # Convert dict → list of unique docs
        results = list(unique_docs.values())

        return JsonResponse({
            'success': True,
            'documents': results,
            'total_count': len(results)
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f"Error fetching documents: {str(e)}",
            'documents': [],
            'total_count': 0
        })

# -------------------------------
# Delete document (updated for Chroma)
# -------------------------------
@csrf_exempt
def delete_document(request):
    """Delete an uploaded document from LangChain Chroma by document_id."""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST requests allowed'})
    
    document_id = request.POST.get('document_id', '').strip()
    if not document_id:
        return JsonResponse({'success': False, 'error': 'Document ID required'})
    
    try:
        original_name = None
        chunk_count = 0

        # Fetch metadata for this document before deleting
        try:
            chunk_results = vectorstore._collection.get(
                where={"document_id": str(document_id)},
                include=["metadatas"]
            )
            chunk_count = len(chunk_results.get("metadatas", []))
            for metadata in chunk_results.get("metadatas", []):
                if metadata and "original_name" in metadata:
                    original_name = metadata["original_name"]
                    break   # first occurrence is enough
        except Exception as e:
            print(f"Error fetching metadata from Chroma: {str(e)}")

        # Delete from Chroma
        try:
            vectorstore._collection.delete(where={"document_id": str(document_id)})
        except Exception as e:
            print(f"Error deleting from Chroma: {str(e)}")
        
        if not original_name:
            original_name = "Unknown"

        return JsonResponse({
            'success': True,
            'message': f"Document '{original_name}' with {chunk_count} chunks deleted successfully from Chroma"
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

def download_document(request, doc_id):
    """Serve uploaded documents by ID"""
    try:
        document = Document.objects.get(document_id=doc_id)
        file_path = document.file_path
        
        response = FileResponse(
            open(file_path, "rb"), 
            as_attachment=True, 
            filename=document.original_name
        )
        
        return response
        
    except Document.DoesNotExist:
        raise Http404("Document not found")
    except FileNotFoundError:
        raise Http404("File not found on server")

# -------------------------------
# Enhanced RAG functions
# -------------------------------
def get_relevant_documents(query: str, session_id: str = None, n_results: int = 5) -> List[Dict[str, Any]]:
    """Get relevant documents for RAG using semantic search with chapter information."""
    try:
        results = vectorstore.similarity_search_with_score(query, k=n_results)

        relevant_docs = []
        for doc, score in results:
            metadata = doc.metadata
            relevant_docs.append({
                "content": doc.page_content,
                "source": metadata.get("original_name", ""),
                "file_type": metadata.get("file_type", ""),
                "document_id": metadata.get("document_id", ""),
                "chunk_type": metadata.get("chunk_type", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "chapter_title": metadata.get("chapter_title", ""),
                "chapter_number": metadata.get("chapter_number", ""),
                "section_header": metadata.get("section_header", ""),
                "similarity": 1 - score
            })

        return relevant_docs
        
    except Exception as e:
        print(f"Error retrieving relevant documents: {str(e)}")
        return []

@csrf_exempt
def get_document_chapters(request):
    """Get chapter/section information for a specific document."""
    if request.method != 'GET':
        return JsonResponse({'success': False, 'error': 'Only GET requests allowed'})
    
    document_id = request.GET.get('document_id', '').strip()
    if not document_id:
        return JsonResponse({'success': False, 'error': 'Document ID required'})
    
    try:
        # Fetch all chunks for this document
        chunk_results = vectorstore._collection.get(
            where={"document_id": str(document_id)},
            include=["metadatas"]
        )
        
        chapters = []
        for metadata in chunk_results.get("metadatas", []):
            if metadata:
                chapter_info = {
                    "chunk_index": metadata.get("chunk_index", 0),
                    "chunk_type": metadata.get("chunk_type", ""),
                    "chapter_title": metadata.get("chapter_title", ""),
                    "chapter_number": metadata.get("chapter_number", ""),
                    "section_header": metadata.get("section_header", ""),
                }
                chapters.append(chapter_info)
        
        # Sort by chunk_index
        chapters.sort(key=lambda x: x["chunk_index"])
        
        return JsonResponse({
            'success': True,
            'document_id': document_id,
            'chapters': chapters,
            'total_chapters': len(chapters)
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f"Error fetching chapters: {str(e)}"
        })