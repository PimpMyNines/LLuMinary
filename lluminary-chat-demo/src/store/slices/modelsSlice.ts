import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Model, Provider } from '../../types';

interface ModelsState {
  providers: Provider[];
  availableModels: Model[];
  isLoading: boolean;
  error: string | null;
}

// Initial data with common LLM providers and their models
const initialState: ModelsState = {
  providers: [
    {
      id: 'openai',
      name: 'OpenAI',
      logo: '/logos/openai.svg', // Need to create these logos
      apiKeyName: 'OPENAI_API_KEY',
      models: [],
      isConfigured: false,
    },
    {
      id: 'anthropic',
      name: 'Anthropic',
      logo: '/logos/anthropic.svg',
      apiKeyName: 'ANTHROPIC_API_KEY',
      models: [],
      isConfigured: false,
    },
    {
      id: 'google',
      name: 'Google',
      logo: '/logos/google.svg',
      apiKeyName: 'GOOGLE_API_KEY',
      models: [],
      isConfigured: false,
    },
    {
      id: 'cohere',
      name: 'Cohere',
      logo: '/logos/cohere.svg',
      apiKeyName: 'COHERE_API_KEY',
      models: [],
      isConfigured: false,
    },
    {
      id: 'bedrock',
      name: 'AWS Bedrock',
      logo: '/logos/aws.svg',
      apiKeyName: 'AWS_PROFILE',
      models: [],
      isConfigured: false,
    },
  ],
  availableModels: [],
  isLoading: false,
  error: null,
};

// Define models with their capabilities
const getDefaultModels = (): Model[] => [
  // OpenAI models
  {
    id: 'gpt-4o',
    provider: initialState.providers.find(p => p.id === 'openai')!,
    displayName: 'GPT-4o',
    description: 'Most capable OpenAI model with visual understanding',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: false,
      token_window: 128000,
    },
  },
  {
    id: 'gpt-4-turbo',
    provider: initialState.providers.find(p => p.id === 'openai')!,
    displayName: 'GPT-4 Turbo',
    description: 'Powerful model with strong reasoning capabilities',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: false,
      token_window: 128000,
    },
  },
  {
    id: 'gpt-4o-mini',
    provider: initialState.providers.find(p => p.id === 'openai')!,
    displayName: 'GPT-4o Mini',
    description: 'Efficient version of GPT-4o with lower cost',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: false,
      token_window: 128000,
    },
  },
  // Anthropic models
  {
    id: 'claude-3-opus-20240229',
    provider: initialState.providers.find(p => p.id === 'anthropic')!,
    displayName: 'Claude 3 Opus',
    description: 'Most powerful Claude model with deep reasoning',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: true,
      token_window: 200000,
    },
  },
  {
    id: 'claude-3-sonnet-20240229',
    provider: initialState.providers.find(p => p.id === 'anthropic')!,
    displayName: 'Claude 3 Sonnet',
    description: 'Balanced Claude model for general use',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: true,
      token_window: 200000,
    },
  },
  {
    id: 'claude-3-haiku-20240307',
    provider: initialState.providers.find(p => p.id === 'anthropic')!,
    displayName: 'Claude 3 Haiku',
    description: 'Fast and efficient Claude model',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: true,
      token_window: 200000,
    },
  },
  // Google models
  {
    id: 'gemini-pro',
    provider: initialState.providers.find(p => p.id === 'google')!,
    displayName: 'Gemini Pro',
    description: 'Google\'s flagship multimodal model',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: false,
      token_window: 32000,
    },
  },
  {
    id: 'gemini-2.0-flash',
    provider: initialState.providers.find(p => p.id === 'google')!,
    displayName: 'Gemini 2.0 Flash',
    description: 'Fast Google model for efficient processing',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: false,
      token_window: 32000,
    },
  },
  // Cohere models
  {
    id: 'command-r-plus',
    provider: initialState.providers.find(p => p.id === 'cohere')!,
    displayName: 'Command R+',
    description: 'Cohere\'s top-tier reasoning model',
    capabilities: {
      streaming: true,
      images: false,
      embeddings: true,
      reranking: true,
      function_calling: true,
      thinking_budget: false,
      token_window: 128000,
    },
  },
  {
    id: 'command-r',
    provider: initialState.providers.find(p => p.id === 'cohere')!,
    displayName: 'Command R',
    description: 'Balanced Cohere model with strong reasoning',
    capabilities: {
      streaming: true,
      images: false,
      embeddings: true,
      reranking: true,
      function_calling: true,
      thinking_budget: false,
      token_window: 128000,
    },
  },
  // AWS Bedrock models
  {
    id: 'anthropic.claude-3-opus-20240229-v1:0',
    provider: initialState.providers.find(p => p.id === 'bedrock')!,
    displayName: 'Claude 3 Opus (Bedrock)',
    description: 'Claude 3 Opus via AWS Bedrock',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: true,
      token_window: 200000,
    },
  },
  {
    id: 'anthropic.claude-3-sonnet-20240229-v1:0',
    provider: initialState.providers.find(p => p.id === 'bedrock')!,
    displayName: 'Claude 3 Sonnet (Bedrock)',
    description: 'Claude 3 Sonnet via AWS Bedrock',
    capabilities: {
      streaming: true,
      images: true,
      embeddings: false,
      reranking: false,
      function_calling: true,
      thinking_budget: true,
      token_window: 200000,
    },
  }
];

// Add models to providers
const modelsWithProviders = getDefaultModels();
const providersWithModels = [...initialState.providers];

modelsWithProviders.forEach(model => {
  const provider = providersWithModels.find(p => p.id === model.provider.id);
  if (provider) {
    if (!provider.models) provider.models = [];
    provider.models.push(model);
  }
});

initialState.providers = providersWithModels;
initialState.availableModels = modelsWithProviders;

const modelsSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {
    setProviderConfigured: (state, action: PayloadAction<{ providerId: string; isConfigured: boolean }>) => {
      const { providerId, isConfigured } = action.payload;
      const provider = state.providers.find(p => p.id === providerId);

      if (provider) {
        provider.isConfigured = isConfigured;

        // Update available models based on configured providers
        state.availableModels = getDefaultModels().filter(
          model => state.providers.find(p => p.id === model.provider.id)?.isConfigured
        );
      }
    },

    setAvailableModels: (state, action: PayloadAction<Model[]>) => {
      state.availableModels = action.payload;
    },

    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },

    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const {
  setProviderConfigured,
  setAvailableModels,
  setLoading,
  setError,
} = modelsSlice.actions;

export default modelsSlice.reducer;
