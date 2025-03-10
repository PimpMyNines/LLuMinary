import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { Message as MessageType } from '../types';
import Message from './Message';

interface MessageListProps {
  messages: MessageType[];
}

const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  const { showTokenCounts, showCosts } = useSelector((state: RootState) => state.settings);

  if (messages.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-gray-400">
        <div className="text-center max-w-md">
          <h3 className="text-xl font-medium mb-2">Welcome to LLuMinary Chat</h3>
          <p className="mb-4">This demo showcases the integration with LLuMinary, a unified API for multiple LLM providers.</p>
          <p className="text-sm italic">Type a message below to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {messages.map((message) => (
        <Message
          key={message.id}
          message={message}
          showTokens={showTokenCounts}
          showCost={showCosts}
        />
      ))}
    </div>
  );
};

export default MessageList;
