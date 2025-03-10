import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { nord } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Message as MessageType } from '../types';
import { UserIcon, ComputerDesktopIcon, WrenchIcon } from '@heroicons/react/24/outline';

interface MessageProps {
  message: MessageType;
  showTokens: boolean;
  showCost: boolean;
}

const Message: React.FC<MessageProps> = ({ message, showTokens, showCost }) => {
  const { role, content, isLoading, usage } = message;

  // Determine message styling based on role
  const isUser = role === 'user';
  const isAssistant = role === 'assistant';
  const isTool = role === 'tool' || role === 'function';

  // Styling based on role
  const messageStyles = isUser
    ? 'bg-primary-50 border-primary-200'
    : isAssistant
      ? 'bg-white border-gray-200'
      : 'bg-neutral-50 border-neutral-200';

  const Icon = isUser
    ? UserIcon
    : isAssistant
      ? ComputerDesktopIcon
      : WrenchIcon;

  // Loading indicator
  const loadingIndicator = isLoading && (
    <div className="flex items-center justify-center h-6 mt-2">
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
      </div>
    </div>
  );

  return (
    <div className={`p-4 rounded-lg border shadow-sm ${messageStyles}`}>
      <div className="flex items-start">
        <div className={`p-2 rounded-full mr-3 ${isUser ? 'bg-primary-100' : isAssistant ? 'bg-gray-100' : 'bg-neutral-100'}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-gray-900">
            {isUser ? 'You' : isAssistant ? 'AI Assistant' : 'Tool'}
          </div>
          <div className="message-content mt-1">
            {content ? (
              <ReactMarkdown
                className="markdown-content"
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={nord}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {content}
              </ReactMarkdown>
            ) : loadingIndicator}
          </div>

          {/* Token and cost information */}
          {usage && (isAssistant || isTool) && (
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500">
              {showTokens && usage.total_tokens !== undefined && (
                <div className="flex items-center">
                  <span className="font-medium">Tokens:</span>{' '}
                  <span className="ml-1">
                    {usage.read_tokens !== undefined ? usage.read_tokens : 0} in
                    {' + '}
                    {usage.write_tokens !== undefined ? usage.write_tokens : 0} out
                    {' = '}
                    {usage.total_tokens}
                  </span>
                </div>
              )}

              {showCost && usage.total_cost !== undefined && (
                <div className="flex items-center">
                  <span className="font-medium">Cost:</span>{' '}
                  <span className="ml-1">${usage.total_cost.toFixed(6)}</span>
                </div>
              )}

              {usage.tool_use && (
                <div className="flex items-center">
                  <span className="font-medium">Tool used:</span>{' '}
                  <span className="ml-1">{usage.tool_use.name || usage.tool_use.id}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;
