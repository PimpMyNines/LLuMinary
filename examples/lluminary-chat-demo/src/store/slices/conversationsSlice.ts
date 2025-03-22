import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';
import { Conversation, Message, Model, UsageData } from '../../types';

interface ConversationsState {
  conversations: Conversation[];
  activeConversationId: string | null;
}

const initialState: ConversationsState = {
  conversations: [],
  activeConversationId: null,
};

const conversationsSlice = createSlice({
  name: 'conversations',
  initialState,
  reducers: {
    createConversation: (state, action: PayloadAction<{ model: Model }>) => {
      const id = uuidv4();
      const now = Date.now();
      const newConversation: Conversation = {
        id,
        title: 'New Conversation',
        messages: [],
        model: action.payload.model,
        createdAt: now,
        updatedAt: now,
      };

      state.conversations.push(newConversation);
      state.activeConversationId = id;
    },

    setActiveConversation: (state, action: PayloadAction<string>) => {
      state.activeConversationId = action.payload;
    },

    addMessage: (state, action: PayloadAction<{ conversationId: string; message: Partial<Message> }>) => {
      const { conversationId, message } = action.payload;
      const conversation = state.conversations.find(c => c.id === conversationId);

      if (conversation) {
        const newMessage: Message = {
          id: uuidv4(),
          role: message.role || 'user',
          content: message.content || '',
          timestamp: Date.now(),
          ...message,
        };

        conversation.messages.push(newMessage);
        conversation.updatedAt = Date.now();

        // Update title if it's the first user message
        if (conversation.title === 'New Conversation' && newMessage.role === 'user') {
          conversation.title = newMessage.content.slice(0, 30) + (newMessage.content.length > 30 ? '...' : '');
        }
      }
    },

    updateMessage: (
      state,
      action: PayloadAction<{
        conversationId: string;
        messageId: string;
        content?: string;
        isLoading?: boolean;
        usage?: UsageData;
      }>
    ) => {
      const { conversationId, messageId, content, isLoading, usage } = action.payload;
      const conversation = state.conversations.find(c => c.id === conversationId);

      if (conversation) {
        const message = conversation.messages.find(m => m.id === messageId);
        if (message) {
          if (content !== undefined) message.content = content;
          if (isLoading !== undefined) message.isLoading = isLoading;
          if (usage !== undefined) message.usage = usage;
          conversation.updatedAt = Date.now();
        }
      }
    },

    deleteConversation: (state, action: PayloadAction<string>) => {
      const index = state.conversations.findIndex(c => c.id === action.payload);

      if (index !== -1) {
        state.conversations.splice(index, 1);

        if (state.activeConversationId === action.payload) {
          state.activeConversationId = state.conversations.length > 0 ? state.conversations[0].id : null;
        }
      }
    },

    updateConversationModel: (
      state,
      action: PayloadAction<{ conversationId: string; model: Model }>
    ) => {
      const { conversationId, model } = action.payload;
      const conversation = state.conversations.find(c => c.id === conversationId);

      if (conversation) {
        conversation.model = model;
        conversation.updatedAt = Date.now();
      }
    },
  },
});

export const {
  createConversation,
  setActiveConversation,
  addMessage,
  updateMessage,
  deleteConversation,
  updateConversationModel,
} = conversationsSlice.actions;

export default conversationsSlice.reducer;
