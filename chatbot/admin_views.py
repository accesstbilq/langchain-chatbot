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
import openai
from typing import List, Dict, Any
import json
from django.utils import timezone
import os
import re
from pydantic import BaseModel, Field
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument






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
APP_URL = "http://127.0.0.1:8000/"
# Initialize OpenAI client
openai.api_key = OPENAIKEY


llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=OPENAIKEY,
    model=MODEL
)

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

class PDFChunker:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def clean_text(self, text: str) -> str:
        """Clean common PDF parsing artifacts"""
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix spacing around numbers (common OCR issue)
        text = re.sub(r"(\d)\s+(\d)", r"\1.\2", text)
        # Remove excessive whitespace
        text = re.sub(r" {2,}", " ", text)
        # Fix common hyphenation issues
        text = re.sub(r"-\s*\n\s*", "", text)
        return text.strip()
    
    def method_2_semantic_heading_splitter(self) -> List[Dict[str, Any]]:
        """
        Method 2: Improved Semantic Heading-Based Splitting
        Best for: Structured documents with clear headings (like your SEO guides)
        """
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])
        full_text = self.clean_text(full_text)
        
        # Enhanced heading pattern - covers more cases
        heading_patterns = [
            r"(?:^|\n)\s*(\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,\-():]+)",  # 4.6.1 Title
            r"(?:^|\n)\s*(Chapter\s+\d+[^\n]*)",                        # Chapter 1 Title  
            r"(?:^|\n)\s*([A-Z][A-Z\s]{3,}[A-Z])\s*(?:\n|$)",          # ALL CAPS HEADINGS
            r"(?:^|\n)\s*(###?\s+[^\n]+)",                              # ### Markdown headings
            r"(?:^|\n)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:)\s*\n"       # Title: format
        ]
        
        # Combine all patterns
        combined_pattern = "|".join(f"({pattern})" for pattern in heading_patterns)
        heading_regex = re.compile(combined_pattern, re.MULTILINE | re.IGNORECASE)
        
        chunks = []
        last_end = 0
        current_heading = "Introduction"
        
        for match in heading_regex.finditer(full_text):
            # Add previous section
            if last_end < match.start():
                content = full_text[last_end:match.start()].strip()
                if content:
                    chunks.append({
                        "heading": current_heading,
                        "content": content,
                        "start_pos": last_end,
                        "end_pos": match.start(),
                        "word_count": len(content.split())
                    })
            
            # Update heading
            current_heading = match.group().strip()
            last_end = match.end()
        
        # Add final section
        if last_end < len(full_text):
            content = full_text[last_end:].strip()
            if content:
                chunks.append({
                    "heading": current_heading,
                    "content": content,
                    "start_pos": last_end,
                    "end_pos": len(full_text),
                    "word_count": len(content.split())
                })
        
        return chunks
    
    
    def method_4_hybrid_approach(self, max_chunk_size: int = 3000, min_chunk_size: int = 500) -> List[Dict[str, Any]]:
        """
        Method 4: Hybrid Semantic + Size-Based Splitting
        Best for: When you want semantic chunks but with size constraints
        """
        # First, get semantic chunks
        semantic_chunks = self.method_2_semantic_heading_splitter()
        
        final_chunks = []
        
        for chunk in semantic_chunks:
            content = chunk['content']
            heading = chunk['heading']
            
            # If chunk is too large, split it further
            if len(content) > max_chunk_size:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chunk_size,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                sub_docs = text_splitter.create_documents([content])
                
                for i, sub_doc in enumerate(sub_docs):
                    final_chunks.append({
                        'heading': f"{heading} (Part {i+1})",
                        'content': sub_doc.page_content,
                        'word_count': len(sub_doc.page_content.split()),
                        'chunk_type': 'hybrid_split',
                        'parent_heading': heading
                    })
            
            # If chunk is too small, consider merging (optional)
            elif len(content) < min_chunk_size and final_chunks:
                # Merge with previous chunk if from same section
                if final_chunks[-1].get('parent_heading') == heading:
                    final_chunks[-1]['content'] += f"\n\n{content}"
                    final_chunks[-1]['word_count'] += len(content.split())
                else:
                    final_chunks.append({
                        'heading': heading,
                        'content': content,
                        'word_count': len(content.split()),
                        'chunk_type': 'semantic',
                        'parent_heading': heading
                    })
            else:
                final_chunks.append({
                    'heading': heading,
                    'content': content,
                    'word_count': len(content.split()),
                    'chunk_type': 'semantic',
                    'parent_heading': heading
                })
        
        return final_chunks

# -------------------------------
# Document storage configuration
# -------------------------------
UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'documents')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def save_docs_to_json_and_chroma(docs, base_dir="documents", filename=None):
    """
    Save docs into JSON file and also into ChromaDB with extra metadata.
    """
    # Ensure folder exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Generate unique JSON filename if not provided
    if not filename:
        json_id = str(uuid.uuid4())
        filename = f"chunk_data_{json_id}.json"

    output_file = os.path.join(base_dir, filename)

    enriched_docs = []
    data = []
    for doc in docs:
        meta = doc.metadata.copy()
        meta["json_file"] = filename   # ✅ add JSON file name to metadata
        data.append({
            "metadata": meta,
            "page_content": doc.page_content
        })
        enriched_docs.append(LCDocument(page_content=doc.page_content, metadata=meta))

    # Save JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(docs)} chunks into {output_file}")

    # Save into Chroma
    vectorstore.add_documents(enriched_docs)
    print(f"✅ Stored {len(enriched_docs)} chunks in ChromaDB")

    return output_file

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
            
            # loader = PyPDFLoader(file_path)
            # chunk_documents = loader.load()
            

            base_metadata = {
                "document_id": str(unique_id),
                "session_id": str(session.session_id) if session else None,
                "original_name": file.name,
                "file_name": filename,
                "file_size": file.size,
                "file_type": file_extension,
                "mime_type": mimetypes.guess_type(file.name)[0] or "application/octet-stream",
                "file_path": file_path,
                "url": fs.url(filename) if hasattr(fs, "url") else None,
                "upload_date": timezone.now().isoformat()
            }

            chunker = PDFChunker(file_path)
            chunks_method4 = chunker.method_4_hybrid_approach()
            print(f"Generated {len(chunks_method4)} chunks")
            docs = []
            for i, chunk in enumerate(chunks_method4):
                
                meta = base_metadata.copy()
                meta.update({
                    "chunk_index": i,
                    "heading": chunk.get("heading"),
                    "parent_heading": chunk.get("parent_heading"),
                    "chunk_type": chunk.get("chunk_type", "hybrid"),
                    "word_count": chunk.get("word_count", 0),
                })

                # Auto-generate structured metadata with LLM
                try:
                    llm_metadata = generate_chunk_label(chunk["content"])
                    print(f"Json for chunk {i}: {llm_metadata}")
                    # merge Pydantic object into dict
                    meta.update(llm_metadata.model_dump())
                except Exception as e:
                    print(f"⚠️ Metadata generation failed for chunk {i}: {e}")

                docs.append(LCDocument(page_content=chunk["content"], metadata=meta))

            #docs = extract_text_and_split(file_path, file_extension, base_metadata)
            savefile = save_docs_to_json_and_chroma(docs, base_dir="documents")

            print(f"Split into {len(docs)} chunks with metadata")
            
            # Prepare response entry
            uploaded_files.append({
                'id': unique_id,
                'name': file.name,
                'size': file.size,
                'type': file_extension,
                'savefile': savefile,
                'upload_date': datetime.now().isoformat(),
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


def generate_chunk_label(content: str) -> str:
    
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

    return generated_metadata

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
                            "primary_topic": metadata.get("primary_topic", ""),
                            "specific_element": metadata.get("specific_element", ""),
                            "content_type": metadata.get("content_type", ""),
                            "source_brand": metadata.get("source_brand", ""),
                            "size": metadata.get("file_size", 0),
                            "type": metadata.get("file_type", ""),
                            "upload_date": metadata.get("upload_date", ""),
                            "mime_type": metadata.get("mime_type", "application/octet-stream"),
                            "json_file": metadata.get("json_file", ""),
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
    
def chunk_document(request, filename):
    file_path = os.path.join(settings.BASE_DIR, "documents", filename)
    if os.path.exists(file_path):
        return FileResponse(open(file_path, "rb"), as_attachment=True, filename=filename)
    raise Http404("File not found")
