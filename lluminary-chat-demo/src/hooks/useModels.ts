import { useCallback, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { LLuMinaryApi } from '../services/api';
import { RootState } from '../store';
import {
  setProviderConfigured,
  setAvailableModels,
  setLoading,
  setError
} from '../store/slices/modelsSlice';
import { setApiKey } from '../store/slices/settingsSlice';
import { Model } from '../types';

export const useModels = () => {
  const dispatch = useDispatch();
  const { providers, availableModels, isLoading, error } = useSelector((state: RootState) => state.models);
  const apiKeys = useSelector((state: RootState) => state.settings.apiKeys);
  const lluminaryApi = new LLuMinaryApi();

  // Initialize API key for a provider
  const initializeProvider = useCallback(async (providerId: string, apiKey: string) => {
    dispatch(setLoading(true));
    try {
      // Attempt to validate the API key
      const isValid = await lluminaryApi.validateApiKey(providerId, apiKey);

      // Store the API key and update provider status
      dispatch(setApiKey({ provider: providerId, key: apiKey, isValid }));
      dispatch(setProviderConfigured({ providerId, isConfigured: isValid }));

      // Update available models
      refreshAvailableModels();

      return isValid;
    } catch (error) {
      dispatch(setError(error instanceof Error ? error.message : 'Failed to initialize provider'));
      return false;
    } finally {
      dispatch(setLoading(false));
    }
  }, [dispatch, lluminaryApi]);

  // Refresh the list of available models based on configured providers
  const refreshAvailableModels = useCallback(async () => {
    dispatch(setLoading(true));
    try {
      // In a real application, this would fetch from the backend
      // which would use LLuMinary to get the available models
      const models = await lluminaryApi.getAvailableModels(apiKeys);

      // For now, we'll just filter the predefined models by configured providers
      const configuredProviderIds = providers
        .filter(p => p.isConfigured)
        .map(p => p.id);

      const availableModels = providers
        .filter(p => configuredProviderIds.includes(p.id))
        .flatMap(p => p.models);

      dispatch(setAvailableModels(availableModels));
    } catch (error) {
      dispatch(setError(error instanceof Error ? error.message : 'Failed to fetch available models'));
    } finally {
      dispatch(setLoading(false));
    }
  }, [dispatch, lluminaryApi, providers, apiKeys]);

  // Get a specific model by ID
  const getModelById = useCallback((modelId: string): Model | undefined => {
    return availableModels.find(m => m.id === modelId);
  }, [availableModels]);

  // Initialize providers from stored API keys on component mount
  useEffect(() => {
    const initializeProviders = async () => {
      for (const apiKeyConfig of apiKeys) {
        await initializeProvider(apiKeyConfig.provider, apiKeyConfig.key);
      }
    };

    initializeProviders();
  }, []);

  return {
    providers,
    availableModels,
    isLoading,
    error,
    initializeProvider,
    refreshAvailableModels,
    getModelById,
  };
};
