import React, { useState, useRef, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store';
import { useChat } from '../hooks/useChat';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import ModelSelector from './ModelSelector';
import FlowDiagram from './FlowDiagram';
import { updateConversationModel } from '../store/slices/conversationsSlice';
import { toggleFlowDiagram } from '../store/slices/uiSlice';
import { ChartBarIcon, CpuChipIcon } from '@heroicons/react/24/outline';

const ChatInterface: React.FC = () => {
  const dispatch = useDispatch();
  const { sendMessage, isGenerating, conversation } = useChat();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { showFlowDiagram } = useSelector((state: RootState) => state.ui);
  const availableModels = useSelector((state: RootState) => state.models.availableModels);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation?.messages]);

  // Handle sending a message
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || isGenerating || !conversation) return;

    await sendMessage(input);
    setInput('');
  };

  // Handle model change
  const handleModelChange = (modelId: string) => {
    if (!conversation) return;

    const model = availableModels.find(m => m.id === modelId);
    if (model) {
      dispatch(updateConversationModel({
        conversationId: conversation.id,
        model,
      }));
    }
  };

  if (!conversation) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50 p-6">
        <div className="text-center">
          <h3 className="mt-2 text-xl font-semibold text-gray-900">No conversation selected</h3>
          <p className="mt-1 text-gray-500">Create a new conversation to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b bg-white">
        <div className="flex items-center">
          <ModelSelector
            selectedModelId={conversation.model.id}
            onChange={handleModelChange}
          />
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => dispatch(toggleFlowDiagram())}
            className={`p-2 rounded-md ${showFlowDiagram ? 'bg-primary-100 text-primary-800' : 'hover:bg-gray-100'}`}
            title="Show/Hide LLuMinary Flow Diagram"
          >
            <CpuChipIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Chat area with flow diagram */}
      <div className="flex-1 overflow-hidden flex flex-col md:flex-row">
        {/* Messages panel */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
            <MessageList messages={conversation.messages} />
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="border-t bg-white p-4">
            <MessageInput
              value={input}
              onChange={setInput}
              onSubmit={handleSendMessage}
              isLoading={isGenerating}
              placeholder="Type a message..."
            />
          </div>
        </div>

        {/* Flow diagram panel */}
        {showFlowDiagram && (
          <div className="hidden md:block md:w-1/3 border-l bg-white p-4 overflow-auto">
            <h3 className="text-lg font-medium mb-4">LLuMinary Flow</h3>
            <FlowDiagram />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatInterface;
