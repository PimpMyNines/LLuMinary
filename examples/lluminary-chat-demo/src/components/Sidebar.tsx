import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store';
import {
  createConversation,
  setActiveConversation,
  deleteConversation
} from '../store/slices/conversationsSlice';
import { toggleSidebar } from '../store/slices/uiSlice';
import { Model } from '../types';
import {
  PlusIcon,
  ChatBubbleLeftRightIcon,
  XMarkIcon,
  TrashIcon
} from '@heroicons/react/24/outline';

const Sidebar: React.FC = () => {
  const dispatch = useDispatch();
  const { sidebarOpen } = useSelector((state: RootState) => state.ui);
  const { conversations, activeConversationId } = useSelector((state: RootState) => state.conversations);
  const availableModels = useSelector((state: RootState) => state.models.availableModels);
  const { defaultModel } = useSelector((state: RootState) => state.settings);

  // Handle creating a new conversation
  const handleNewConversation = () => {
    if (availableModels.length === 0) {
      alert('Please configure at least one model in Settings before creating a conversation.');
      return;
    }

    // Find the default model or use the first available one
    const model = availableModels.find(m => m.id === defaultModel) || availableModels[0];
    dispatch(createConversation({ model }));
  };

  // Handle selecting a conversation
  const handleSelectConversation = (conversationId: string) => {
    dispatch(setActiveConversation(conversationId));

    // On mobile, close sidebar after selection
    if (window.innerWidth < 768) {
      dispatch(toggleSidebar());
    }
  };

  // Handle deleting a conversation
  const handleDeleteConversation = (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this conversation?')) {
      dispatch(deleteConversation(conversationId));
    }
  };

  return (
    <div className={`
      fixed inset-y-0 left-0 z-20 flex flex-col w-72 bg-white border-r border-gray-200
      transition-transform duration-300 ease-in-out
      ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      md:relative md:translate-x-0
    `}>
      <div className="p-4 border-b border-gray-200 flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-800">LLuMinary Chat</h2>
        <button
          onClick={() => dispatch(toggleSidebar())}
          className="md:hidden p-2 text-gray-500 hover:text-gray-700 focus:outline-none"
        >
          <XMarkIcon className="w-5 h-5" />
        </button>
      </div>

      {/* New Chat Button */}
      <button
        onClick={handleNewConversation}
        className="mx-4 mt-4 mb-2 flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
      >
        <PlusIcon className="w-5 h-5 mr-2" />
        New Chat
      </button>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto p-2">
        {conversations.length === 0 ? (
          <div className="text-center p-4 text-sm text-gray-500">
            No conversations yet
          </div>
        ) : (
          <ul className="space-y-1">
            {conversations.map((conversation) => (
              <li key={conversation.id}>
                <button
                  onClick={() => handleSelectConversation(conversation.id)}
                  className={`
                    w-full flex items-start justify-between p-3 text-left rounded-md hover:bg-gray-100
                    ${conversation.id === activeConversationId ? 'bg-gray-100' : ''}
                  `}
                >
                  <div className="flex items-start space-x-3 min-w-0">
                    <ChatBubbleLeftRightIcon className="w-5 h-5 text-gray-400 mt-0.5" />
                    <div className="flex-1 truncate">
                      <div className="font-medium text-gray-900 truncate">
                        {conversation.title}
                      </div>
                      <div className="text-sm text-gray-500">
                        {new Date(conversation.updatedAt).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={(e) => handleDeleteConversation(e, conversation.id)}
                    className="p-1 text-gray-400 hover:text-red-500"
                    title="Delete conversation"
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 text-xs text-gray-500">
        <p className="mb-1">
          Powered by <span className="font-semibold">LLuMinary</span>
        </p>
        <p>
          Unified API for multiple LLM providers
        </p>
      </div>
    </div>
  );
};

export default Sidebar;
