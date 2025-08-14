from django.db import transaction
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain_core.messages import ToolMessage
from .models import ChatSession, ChatMessage, TokenUsage
import json
import tiktoken
import uuid

# Load ENV file
MODEL = "gpt-3.5-turbo"

def generate_run_id():
    """Generate a unique run ID for conversation tracking"""
    return f"run_{uuid.uuid4().hex[:12]}"

def generate_message_id():
    """Generate a unique message ID"""
    return f"msg_{uuid.uuid4().hex[:12]}"

def get_or_create_chat_session(session_id, user=None, run_id=None):
    """Get existing chat session or create new one"""
    try:
        session, created = ChatSession.objects.get_or_create(
            session_id=session_id,
            defaults={
                'user': user,
                'run_id': run_id or generate_run_id(),
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

def save_message_to_db(session, message_type, content, tool_call_id=None, tool_calls=None, metadata=None, run_id=None, message_id=None):
    """Save a single message to database"""
    try:
        # Get the next order number for this session
        last_message = ChatMessage.objects.filter(session=session).order_by('-order').first()
        next_order = (last_message.order + 1) if last_message else 1
        
        # Generate IDs if not provided
        if not message_id:
            message_id = generate_message_id()
        if not run_id:
            run_id = session.run_id or generate_run_id()
        
        # Create the message
        chat_message = ChatMessage.objects.create(
            session=session,
            message_id=message_id,
            run_id=run_id,
            message_type=message_type,
            content=content,
            tool_call_id=tool_call_id or '',
            tool_calls=tool_calls or {},
            metadata=metadata or {},
            order=next_order
        )
        
        # Update session's updated_at timestamp and run_id
        session.run_id = run_id
        session.save(update_fields=['updated_at', 'run_id'])
        
        return message_id
    except Exception as e:
        print(f"Error saving message to DB: {str(e)}")
        raise

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

# Add this function to track tokens
def count_tokens(messages, model="gpt-3.5-turbo"):
    """Count tokens for a list of messages"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens_per_message = 3  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_name = 1  # if there's a name, the role is omitted
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        if hasattr(message, 'content') and message.content:
            num_tokens += len(encoding.encode(str(message.content)))
        if hasattr(message, 'role'):
            num_tokens += len(encoding.encode(message.role))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def count_tokens_from_string(text, model=MODEL):
    """Count tokens in a string"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

# Add this to your chat_utils.py
def save_token_usage(session, run_id, input_tokens, output_tokens, total_tokens,response_type,tools_used,source,savemessage):
    """Save token usage for a run"""

    token_usage, created = TokenUsage.objects.update_or_create(
        session=session,
        run_id=run_id,
        defaults={
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'response_type': response_type,
            'tools_used': tools_used,
            'source': source,
            'message_id':savemessage, 
        }
    )
    
    return token_usage

# Add this to your chat_utils.py
def get_token_usage(run_id):
    """Get token usage record for a run ID"""
    try:
        return TokenUsage.objects.get(run_id=run_id)
    except TokenUsage.DoesNotExist:
        return None
