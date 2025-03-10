import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store';
import { useModels } from '../hooks/useModels';
import {
  setApiKey,
  setDefaultModel,
  setShowTokenCounts,
  setShowCosts,
  setTemperature,
  setMaxTokens,
  setSystemPrompt,
} from '../store/slices/settingsSlice';
import { toggleSettings } from '../store/slices/uiSlice';
import { XMarkIcon } from '@heroicons/react/24/outline';

const SettingsPanel: React.FC = () => {
  const dispatch = useDispatch();
  const { settingsOpen } = useSelector((state: RootState) => state.ui);
  const settings = useSelector((state: RootState) => state.settings);
  const { providers, initializeProvider } = useModels();

  const [apiKeys, setApiKeys] = useState<{ [key: string]: string }>({});
  const [isSubmitting, setIsSubmitting] = useState<{ [key: string]: boolean }>({});

  // Handle API key form submission
  const handleApiKeySubmit = async (e: React.FormEvent, providerId: string) => {
    e.preventDefault();
    const apiKey = apiKeys[providerId];

    if (!apiKey) return;

    setIsSubmitting({ ...isSubmitting, [providerId]: true });

    try {
      const isValid = await initializeProvider(providerId, apiKey);
      if (isValid) {
        // Clear the form input after successful submission
        setApiKeys({ ...apiKeys, [providerId]: '' });
      }
    } finally {
      setIsSubmitting({ ...isSubmitting, [providerId]: false });
    }
  };

  // Handle input change
  const handleApiKeyChange = (providerId: string, value: string) => {
    setApiKeys({ ...apiKeys, [providerId]: value });
  };

  // Check if provider is configured
  const isProviderConfigured = (providerId: string) => {
    return settings.apiKeys.some(key => key.provider === providerId && key.isValid);
  };

  // Handle default model change
  const handleDefaultModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    dispatch(setDefaultModel(e.target.value));
  };

  if (!settingsOpen) return null;

  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-800">Settings</h2>
          <button
            onClick={() => dispatch(toggleSettings())}
            className="p-2 text-gray-500 hover:text-gray-700 focus:outline-none"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Provider API Keys */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Provider API Keys</h3>
            <div className="space-y-4">
              {providers.map((provider) => (
                <div key={provider.id} className="border border-gray-200 rounded-md p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center">
                      <img
                        src={provider.logo}
                        alt={provider.name}
                        className="w-6 h-6 mr-2"
                        onError={(e) => {
                          e.currentTarget.src = 'https://via.placeholder.com/24';
                        }}
                      />
                      <h4 className="font-medium text-gray-800">{provider.name}</h4>
                    </div>

                    {isProviderConfigured(provider.id) && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        Configured
                      </span>
                    )}
                  </div>

                  {isProviderConfigured(provider.id) ? (
                    <div className="bg-gray-50 rounded p-3 text-sm text-gray-500">
                      <p>
                        API key for {provider.name} has been configured. You can overwrite it with a new one.
                      </p>
                    </div>
                  ) : (
                    <form onSubmit={(e) => handleApiKeySubmit(e, provider.id)} className="mt-2">
                      <label htmlFor={`api-key-${provider.id}`} className="block text-sm font-medium text-gray-700 mb-1">
                        {provider.apiKeyName}
                      </label>
                      <div className="flex space-x-2">
                        <input
                          type="text"
                          id={`api-key-${provider.id}`}
                          value={apiKeys[provider.id] || ''}
                          onChange={(e) => handleApiKeyChange(provider.id, e.target.value)}
                          placeholder={`Enter ${provider.apiKeyName}`}
                          className="flex-1 shadow-sm focus:ring-primary-500 focus:border-primary-500 block w-full sm:text-sm border-gray-300 rounded-md"
                        />
                        <button
                          type="submit"
                          disabled={isSubmitting[provider.id] || !apiKeys[provider.id]}
                          className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                            isSubmitting[provider.id] || !apiKeys[provider.id]
                              ? 'bg-gray-300 cursor-not-allowed'
                              : 'bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500'
                          }`}
                        >
                          {isSubmitting[provider.id] ? 'Saving...' : 'Save'}
                        </button>
                      </div>
                    </form>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* General Settings */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">General Settings</h3>

            <div className="space-y-4">
              {/* Default Model */}
              <div>
                <label htmlFor="default-model" className="block text-sm font-medium text-gray-700 mb-1">
                  Default Model
                </label>
                <select
                  id="default-model"
                  value={settings.defaultModel}
                  onChange={handleDefaultModelChange}
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm rounded-md"
                >
                  <option value="" disabled>Select a model</option>
                  {providers
                    .filter(p => isProviderConfigured(p.id))
                    .flatMap(p => p.models)
                    .map(model => (
                      <option key={model.id} value={model.id}>
                        {model.displayName} ({model.provider.name})
                      </option>
                    ))}
                </select>
              </div>

              {/* Temperature */}
              <div>
                <label htmlFor="temperature" className="block text-sm font-medium text-gray-700 mb-1">
                  Temperature: {settings.temperature}
                </label>
                <input
                  type="range"
                  id="temperature"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => dispatch(setTemperature(parseFloat(e.target.value)))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>More precise</span>
                  <span>More creative</span>
                </div>
              </div>

              {/* Max Tokens */}
              <div>
                <label htmlFor="max-tokens" className="block text-sm font-medium text-gray-700 mb-1">
                  Max Tokens: {settings.maxTokens}
                </label>
                <input
                  type="range"
                  id="max-tokens"
                  min="100"
                  max="4000"
                  step="100"
                  value={settings.maxTokens}
                  onChange={(e) => dispatch(setMaxTokens(parseInt(e.target.value)))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Shorter</span>
                  <span>Longer</span>
                </div>
              </div>

              {/* System Prompt */}
              <div>
                <label htmlFor="system-prompt" className="block text-sm font-medium text-gray-700 mb-1">
                  System Prompt
                </label>
                <textarea
                  id="system-prompt"
                  value={settings.systemPrompt}
                  onChange={(e) => dispatch(setSystemPrompt(e.target.value))}
                  rows={3}
                  className="shadow-sm focus:ring-primary-500 focus:border-primary-500 block w-full sm:text-sm border-gray-300 rounded-md"
                  placeholder="Enter system instructions for the AI"
                />
              </div>

              {/* Toggle Settings */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Show Token Counts</span>
                  <button
                    type="button"
                    onClick={() => dispatch(setShowTokenCounts(!settings.showTokenCounts))}
                    className={`relative inline-flex flex-shrink-0 h-6 w-11 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 ${
                      settings.showTokenCounts ? 'bg-primary-600' : 'bg-gray-200'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform ring-0 transition ease-in-out duration-200 ${
                        settings.showTokenCounts ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Show Costs</span>
                  <button
                    type="button"
                    onClick={() => dispatch(setShowCosts(!settings.showCosts))}
                    className={`relative inline-flex flex-shrink-0 h-6 w-11 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 ${
                      settings.showCosts ? 'bg-primary-600' : 'bg-gray-200'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform ring-0 transition ease-in-out duration-200 ${
                        settings.showCosts ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;
