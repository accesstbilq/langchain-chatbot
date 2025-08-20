from django.urls import path
from . import views
from . import admin_views

# =============================
# URL Patterns for Chatbot App
# =============================

urlpatterns = [
    # -------------------------
    # Main Dashboard & Chatbot
    # -------------------------
    #path('', views.dashboard_view, name='dashboard'),                  # Dashboard landing page
    path('', views.chatbot_view, name='chatbot'),              # Chatbot UI page
    path('chat-input/', views.chatbot_input, name='chatbot-input'),    # Main chatbot API endpoint (handles user input)
    path('get-token-usage/<str:run_id>/', views.get_token_usage_view, name='get_token_usage'),  
    # → Returns token usage (input/output/total) for a given run/session
    
    path('streaming-welcome/', views.streaming_welcome_message, name='streaming_welcome_message'),  
    # → Streams welcome message with typing effect when chatbot loads

    # -------------------------
    # Admin / Document Features
    # -------------------------
    path('chathistory/', admin_views.chathistory, name='chathistory'),             # Admin view: chat session history
    path("chathistory/load/", admin_views.load_chat_history, name="load_chat_history"),  # AJAX endpoint: load chat history dynamically

    path('document/', admin_views.document_view, name='document'),                 # Admin document management page
    path('documents/upload/', admin_views.upload_documents, name='upload_documents'), # Upload new documents
    path('documents/list/', admin_views.list_documents, name='list_documents'),       # List uploaded documents
    path('documents/delete/', admin_views.delete_document, name='delete_document'),   # Delete a document
]
