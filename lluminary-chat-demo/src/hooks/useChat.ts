import { useState, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { v4 as uuidv4 } from 'uuid';
import { LLuMinaryApi } from '../services/api';
import { RootState } from '../store';
import { addMessage, updateMessage } from '../store/slices/conversationsSlice';
import { setIsStreaming } from '../store/slices/uiSlice';
import { Message, UsageData } from '../types';

export const useChat = () => {
  const dispatch = useDispatch();
  const lluminaryApi = new LLuMinaryApi();
  const [isGenerating, setIsGenerating] = useState(false);

  const activeConversationId = useSelector((state: RootState) => state.conversations.activeConversationId);
  const conversations = useSelector((state: RootState) => state.conversations.conversations);
  const activeConversation = conversations.find(c => c.id === activeConversationId);
  const apiKeys = useSelector((state: RootState) => state.settings.apiKeys);
  const settings = useSelector((state: RootState) => state.settings);

  // Function to send a message and get a response
  const sendMessage = useCallback(async (content: string, images?: string[]) => {
    if (!activeConversationId || !activeConversation) return;

    try {
      // Add user message
      const userMessageId = uuidv4();
      dispatch(addMessage({
        conversationId: activeConversationId,
        message: {
          id: userMessageId,
          role: 'user',
          content: content,
          images: images,
          timestamp: Date.now(),
        }
      }));

      // Add placeholder for assistant message
      const assistantMessageId = uuidv4();
      dispatch(addMessage({
        conversationId: activeConversationId,
        message: {
          id: assistantMessageId,
          role: 'assistant',
          content: '',
          timestamp: Date.now(),
          isLoading: true,
        }
      }));

      setIsGenerating(true);

      // Get all messages for context (excluding the loading one)
      const messagesForContext = activeConversation.messages.filter(m => !m.isLoading);

      // Decide whether to use streaming based on model capabilities
      const useStreaming = activeConversation.model.capabilities.streaming;

      if (useStreaming) {
        // Use streaming for a better user experience
        dispatch(setIsStreaming(true));

        let assistantResponse = '';

        await lluminaryApi.streamCompletion(
          activeConversation.model.id,
          settings.systemPrompt,
          messagesForContext,
          settings.temperature,
          settings.maxTokens,
          apiKeys,
          // Chunk callback
          (chunk, usageData) => {
            assistantResponse += chunk;
            dispatch(updateMessage({
              conversationId: activeConversationId,
              messageId: assistantMessageId,
              content: assistantResponse,
              usage: usageData,
            }));
          },
          // Done callback
          (fullText, finalUsage) => {
            dispatch(updateMessage({
              conversationId: activeConversationId,
              messageId: assistantMessageId,
              content: fullText,
              isLoading: false,
              usage: finalUsage,
            }));
            setIsGenerating(false);
            dispatch(setIsStreaming(false));
          }
        );
      } else {
        // Use non-streaming for models that don't support it
        const response = await lluminaryApi.generateCompletion(
          activeConversation.model.id,
          settings.systemPrompt,
          messagesForContext,
          settings.temperature,
          settings.maxTokens,
          apiKeys
        );

        dispatch(updateMessage({
          conversationId: activeConversationId,
          messageId: assistantMessageId,
          content: response.text,
          isLoading: false,
          usage: response.usage,
        }));

        setIsGenerating(false);
      }
    } catch (error) {
      console.error('Error generating response:', error);

      // Update the loading message with an error
      if (activeConversationId) {
        const loadingMessage = activeConversation.messages.find(m => m.isLoading);
        if (loadingMessage) {
          dispatch(updateMessage({
            conversationId: activeConversationId,
            messageId: loadingMessage.id,
            content: 'Error generating response. Please try again.',
            isLoading: false,
          }));
        }
      }

      setIsGenerating(false);
      dispatch(setIsStreaming(false));
    }
  }, [
    activeConversationId,
    activeConversation,
    apiKeys,
    dispatch,
    lluminaryApi,
    settings.maxTokens,
    settings.systemPrompt,
    settings.temperature,
  ]);

  return {
    sendMessage,
    isGenerating,
    conversation: activeConversation,
  };
};
