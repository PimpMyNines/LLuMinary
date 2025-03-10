import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { Model } from '../types';
import { ChevronDownIcon } from '@heroicons/react/24/outline';

interface ModelSelectorProps {
  selectedModelId: string;
  onChange: (modelId: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ selectedModelId, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const availableModels = useSelector((state: RootState) => state.models.availableModels);
  const selectedModel = availableModels.find(model => model.id === selectedModelId);

  if (!selectedModel && availableModels.length > 0) {
    // If selected model isn't available, select the first available one
    onChange(availableModels[0].id);
    return null;
  }

  if (!selectedModel) {
    return (
      <div className="text-gray-500 px-3 py-2 text-sm">
        No models available
      </div>
    );
  }

  const toggleDropdown = () => setIsOpen(!isOpen);

  const handleSelect = (modelId: string) => {
    onChange(modelId);
    setIsOpen(false);
  };

  // Group models by provider
  const modelsByProvider: Record<string, Model[]> = {};
  availableModels.forEach(model => {
    const providerId = model.provider.id;
    if (!modelsByProvider[providerId]) {
      modelsByProvider[providerId] = [];
    }
    modelsByProvider[providerId].push(model);
  });

  return (
    <div className="relative">
      <button
        type="button"
        onClick={toggleDropdown}
        className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
      >
        <img
          src={selectedModel.provider.logo}
          alt={selectedModel.provider.name}
          className="w-5 h-5 mr-2"
          onError={(e) => {
            e.currentTarget.src = 'https://via.placeholder.com/20';
          }}
        />
        {selectedModel.displayName}
        <ChevronDownIcon className="ml-2 -mr-0.5 h-4 w-4" aria-hidden="true" />
      </button>

      {isOpen && (
        <div className="origin-top-left absolute left-0 mt-2 w-72 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 focus:outline-none z-10">
          <div className="py-1 max-h-96 overflow-y-auto">
            {Object.entries(modelsByProvider).map(([providerId, models]) => {
              const provider = models[0].provider;
              return (
                <div key={providerId} className="px-2 py-2">
                  <div className="text-xs font-semibold text-gray-500 px-2 mb-1">
                    {provider.name}
                  </div>
                  {models.map(model => (
                    <button
                      key={model.id}
                      onClick={() => handleSelect(model.id)}
                      className={`block w-full text-left px-4 py-2 text-sm rounded-md ${
                        model.id === selectedModelId
                          ? 'bg-primary-50 text-primary-700'
                          : 'text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span>{model.displayName}</span>
                        <div className="flex space-x-1">
                          {model.capabilities.streaming && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                              Stream
                            </span>
                          )}
                          {model.capabilities.images && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                              Vision
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5 truncate">
                        {model.description}
                      </p>
                    </button>
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;
