from django.db import models
from django.conf import settings

from django.contrib.auth.models import User
import json
from django.utils import timezone

class ChatSession(models.Model):
    """Model to store chat sessions"""
    session_id = models.CharField(max_length=100, unique=True, db_index=True)
    run_id = models.CharField(max_length=100, blank=True, db_index=True)  # Track conversation runs
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    title = models.CharField(max_length=200, blank=True)  # Optional session title
    
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
    metadata = models.JSONField(default=dict, blank=True)  # Store additional metadata
    created_at = models.DateTimeField(auto_now_add=True)
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
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."

class DocumentUpload(models.Model):
    """Model to store uploaded documents metadata"""
    document_id = models.CharField(max_length=100, unique=True, db_index=True)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='documents', null=True, blank=True)
    original_name = models.CharField(max_length=255)
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField()
    file_type = models.CharField(max_length=10)
    mime_type = models.CharField(max_length=100)
    upload_date = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)
    processing_status = models.CharField(max_length=50, default='pending')
    
    class Meta:
        ordering = ['-upload_date']
    
    def __str__(self):
        return f"{self.original_name} ({self.file_type})"
    

class TokenUsage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='token_usages', null=True, blank=True)
    run_id = models.CharField(max_length=100, db_index=True)
    message_id = models.CharField(max_length=100, null=True, blank=True)
    response_type = models.CharField(max_length=100, null=True, blank=True)
    tools_used = models.CharField(max_length=100, null=True, blank=True)
    source = models.CharField(max_length=100, null=True, blank=True)
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['session', 'run_id']
        indexes = [
            models.Index(fields=['run_id']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"Tokens for Run {self.run_id}: {self.total_tokens}"