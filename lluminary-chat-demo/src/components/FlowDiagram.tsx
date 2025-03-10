import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';

const FlowDiagram: React.FC = () => {
  const activeConversation = useSelector((state: RootState) => {
    const { activeConversationId, conversations } = state.conversations;
    return conversations.find(c => c.id === activeConversationId);
  });

  if (!activeConversation) {
    return <div>No active conversation</div>;
  }

  const model = activeConversation.model;

  return (
    <div className="llm-flow-diagram">
      <svg
        width="100%"
        height="500"
        viewBox="0 0 600 500"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Diagram Title */}
        <text x="300" y="30" textAnchor="middle" fontSize="16" fontWeight="bold">
          LLuMinary Request Flow
        </text>

        {/* User Input Box */}
        <rect x="50" y="60" width="180" height="60" rx="8" />
        <text x="140" y="95" textAnchor="middle">User Input</text>

        {/* Arrow down to Handler */}
        <path
          d="M140 120 L140 150"
          fill="none"
          markerEnd="url(#arrowhead)"
        />

        {/* LLuMinary Handler Box */}
        <rect x="50" y="150" width="180" height="60" rx="8" />
        <text x="140" y="185" textAnchor="middle">LLuMinary Handler</text>

        {/* Arrow down to Router */}
        <path
          d="M140 210 L140 240"
          fill="none"
          markerEnd="url(#arrowhead)"
        />

        {/* Router Box */}
        <rect x="50" y="240" width="180" height="60" rx="8" />
        <text x="140" y="275" textAnchor="middle">Model Router</text>

        {/* Arrow to Provider */}
        <path
          d="M140 300 L140 330"
          fill="none"
          markerEnd="url(#arrowhead)"
        />

        {/* Provider Box */}
        <rect x="50" y="330" width="180" height="60" rx="8" />
        <text x="140" y="360" textAnchor="middle">
          {model.provider.name} Provider
        </text>
        <text x="140" y="380" textAnchor="middle" fontSize="12">
          {model.displayName}
        </text>

        {/* Arrow to API */}
        <path
          d="M230 360 L280 360"
          fill="none"
          markerEnd="url(#arrowhead)"
        />

        {/* API Circle */}
        <circle cx="320" y="360" r="40" />
        <text x="320" y="360" textAnchor="middle">
          {model.provider.name}
        </text>
        <text x="320" y="375" textAnchor="middle" fontSize="12">
          API
        </text>

        {/* Return Arrow */}
        <path
          d="M280 390 L230 390"
          fill="none"
          markerEnd="url(#arrowhead)"
        />

        {/* Arrow up from Provider */}
        <path
          d="M140 330 L140 300"
          fill="none"
          strokeDasharray="5,5"
        />

        {/* Response Path */}
        <path
          d="M50 420 L140 420 L140 390"
          fill="none"
          strokeDasharray="5,5"
          markerEnd="url(#arrowhead)"
        />

        {/* Response Box */}
        <rect x="370" y="150" width="180" height="60" rx="8" />
        <text x="460" y="185" textAnchor="middle">Response Processing</text>

        {/* Arrow from Provider to Response */}
        <path
          d="M230 360 L370 185"
          fill="none"
          strokeDasharray="5,5"
          markerEnd="url(#arrowhead)"
        />

        {/* Arrow to Output */}
        <path
          d="M460 210 L460 240"
          fill="none"
          strokeDasharray="5,5"
          markerEnd="url(#arrowhead)"
        />

        {/* Output Box */}
        <rect x="370" y="240" width="180" height="60" rx="8" />
        <text x="460" y="275" textAnchor="middle">AI Response</text>

        {/* Capabilities */}
        <rect x="370" y="330" width="180" height="120" rx="8" />
        <text x="460" y="350" textAnchor="middle" fontWeight="bold" fontSize="14">
          Model Capabilities
        </text>
        <text x="390" y="375" fontSize="12" textAnchor="start">
          Streaming: {model.capabilities.streaming ? "✓" : "✗"}
        </text>
        <text x="390" y="395" fontSize="12" textAnchor="start">
          Images: {model.capabilities.images ? "✓" : "✗"}
        </text>
        <text x="390" y="415" fontSize="12" textAnchor="start">
          Function Calling: {model.capabilities.function_calling ? "✓" : "✗"}
        </text>
        <text x="390" y="435" fontSize="12" textAnchor="start">
          Context: {model.capabilities.token_window.toLocaleString()} tokens
        </text>

        {/* Arrowhead Marker Definition */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#4b5563" />
          </marker>
        </defs>
      </svg>

      <div className="text-sm text-gray-500 mt-4 p-4 bg-gray-50 rounded-lg">
        <p className="mb-2 font-medium">How LLuMinary works:</p>
        <ol className="list-decimal pl-5 space-y-1">
          <li>User input is received and processed</li>
          <li>LLuMinary Handler manages the request</li>
          <li>Model Router selects the appropriate provider</li>
          <li>Provider implements the specific API protocol</li>
          <li>API call is made to the selected provider</li>
          <li>Response is processed and formatted</li>
          <li>Result is displayed to the user</li>
        </ol>
      </div>
    </div>
  );
};

export default FlowDiagram;
