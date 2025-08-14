from django.urls import path
from . import views
from . import chatviews

urlpatterns = [
    #path('', views.chatbot_view, name='chat'),
    path('', chatviews.dashboard_view, name='dashboard'),
    path('chatbot/', chatviews.chatbot_view, name='chatbot'),
    path('chat-input/', chatviews.chatbot_input, name='chatbot-input'),
    path('get-response/', views.validate_url_view, name='get_message'),
    path('get-token-usage/<str:run_id>/', chatviews.get_token_usage_view, name='get_token_usage'),

    # New chat history endpoints
    path('chat-history/', views.get_chat_history_view, name='get_chat_history'),
    path('search-history/', views.search_chat_history_view, name='search_chat_history'),
    path('clear-history/', views.clear_chat_history_view, name='clear_chat_history'),

    # New document management routes
    path('document/', chatviews.document_view, name='document'),
    path('documents/upload/', chatviews.upload_documents, name='upload_documents'),
    path('documents/list/', chatviews.list_documents, name='list_documents'),
    path('documents/delete/', chatviews.delete_document, name='delete_document'),
    
]
