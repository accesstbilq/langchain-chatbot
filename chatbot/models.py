from django.db import models
from django.conf import settings

from django.contrib.auth.models import User
import json
from django.utils import timezone

from django.contrib.postgres.fields import ArrayField
import uuid


class ChatSession(models.Model):
    """Model to store chat sessions"""
    session_id = models.CharField(max_length=100, unique=True, db_index=True)
    run_id = models.CharField(max_length=100, blank=True, db_index=True)  # Track conversation runs
    user_name = models.CharField(max_length=100, blank=True, null=True)  # Store validated user name
    name_validated = models.BooleanField(default=False)  # Track name validation status
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"Session {self.session_id} - {self.created_at}"

class ChatMessage(models.Model):
    """Model to store individual chat messages"""
    MESSAGE_TYPES = (
        ('system', 'System'),
        ('human', 'Human'),
        ('ai', 'AI'),
        ('tool', 'Tool'),
    )
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message_id = models.CharField(max_length=100, unique=True, db_index=True)  # Unique message identifier
    run_id = models.CharField(max_length=100, blank=True, db_index=True)  # Link to conversation run
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    tool_call_id = models.CharField(max_length=100, blank=True, null=True)  # For tool messages
    tool_calls = models.JSONField(default=dict, blank=True)  # Store tool calls data
    response_type = models.CharField(max_length=100, null=True, blank=True)
    tools_used = models.CharField(max_length=100, null=True, blank=True)
    source = models.CharField(max_length=100, null=True, blank=True)
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    order = models.PositiveIntegerField()  # To maintain message order
    
    class Meta:
        ordering = ['session', 'order']
        indexes = [
            models.Index(fields=['session', 'order']),
            models.Index(fields=['session', 'created_at']),
            models.Index(fields=['message_id']),
            models.Index(fields=['run_id']),
            models.Index(fields=['session', 'run_id']),
        ]
    
    #def __str__(self):
        #return f"{self.message_type}: {self.content[:50]}..."

# class Document(models.Model):
#     DOCUMENT_TYPES = [
#         ('txt', 'Text'),
#         ('pdf', 'PDF'),
#         ('doc', 'Word Document'),
#         ('docx', 'Word Document'),
#         ('html', 'HTML'),
#         ('css', 'CSS'),
#         ('js', 'JavaScript'),
#         ('json', 'JSON'),
#         ('xml', 'XML'),
#         ('csv', 'CSV'),
#         ('xlsx', 'Excel'),
#         ('ppt', 'PowerPoint'),
#         ('pptx', 'PowerPoint'),
#     ]
    
#     document_id = models.UUIDField(default=uuid.uuid4, unique=True, primary_key=True)
#     session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='documents', null=True, blank=True)
#     original_name = models.CharField(max_length=255)
#     file_name = models.CharField(max_length=255)
#     file_size = models.PositiveIntegerField()
#     file_type = models.CharField(max_length=10, choices=DOCUMENT_TYPES)
#     mime_type = models.CharField(max_length=100)
#     content = models.TextField()  # Extracted text content
#     upload_date = models.DateTimeField(auto_now_add=True)
    
#     # Embedding field - using JSONField for cross-database compatibility
#     embedding = models.JSONField(blank=True, null=True)
    
#     # Metadata
#     embedding_model = models.CharField(max_length=100, blank=True, null=True)  # e.g., 'text-embedding-3-small'
#     embedding_created_at = models.DateTimeField(blank=True, null=True)
    
#     class Meta:
#         ordering = ['-upload_date']
    
#     def __str__(self):
#         return f"{self.original_name} ({self.file_type})"
    
#     @property
#     def has_embedding(self):
#         return self.embedding is not None and len(self.embedding) > 0