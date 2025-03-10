import React, { useState } from 'react';
import { PaperAirplaneIcon, PhotoIcon } from '@heroicons/react/24/solid';

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  isLoading: boolean;
  placeholder?: string;
}

const MessageInput: React.FC<MessageInputProps> = ({
  value,
  onChange,
  onSubmit,
  isLoading,
  placeholder = 'Type a message...',
}) => {
  const [showImageUpload, setShowImageUpload] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit(e);
    }
  };

  const toggleImageUpload = () => {
    setShowImageUpload(!showImageUpload);
  };

  return (
    <form onSubmit={onSubmit} className="relative">
      <textarea
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
        placeholder={placeholder}
        rows={3}
        className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
      />

      <div className="absolute right-3 bottom-3 flex space-x-2">
        <button
          type="button"
          onClick={toggleImageUpload}
          className={`p-2 rounded-full ${
            showImageUpload
              ? 'bg-primary-100 text-primary-700'
              : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
          }`}
          title="Add image"
        >
          <PhotoIcon className="w-5 h-5" />
        </button>

        <button
          type="submit"
          disabled={isLoading || !value.trim()}
          className={`p-2 rounded-full ${
            isLoading || !value.trim()
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-primary-600 text-white hover:bg-primary-700'
          }`}
          title="Send message"
        >
          <PaperAirplaneIcon className="w-5 h-5" />
        </button>
      </div>

      {showImageUpload && (
        <div className="mt-3 p-3 border border-gray-200 rounded-lg bg-gray-50">
          <p className="text-sm text-gray-600 mb-2">Upload images (Coming soon)</p>
          <input
            type="file"
            accept="image/*"
            disabled
            className="text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
          />
          <p className="mt-1 text-xs text-gray-500">
            Image support requires a multimodal model capable of vision features
          </p>
        </div>
      )}
    </form>
  );
};

export default MessageInput;
