export interface Message {
  id: string;
  role: 'system' | 'user' | 'assistant' | 'tool' | 'function';
  content: string;
  timestamp: number;
  images?: string[]; // Base64 encoded images or URLs
  isLoading?: boolean;
  usage?: UsageData;
}

export interface UsageData {
  read_tokens?: number;
  write_tokens?: number;
  total_tokens?: number;
  total_cost?: number;
  images_cost?: number;
  tool_use?: {
    id?: string;
    name?: string;
    arguments?: Record<string, any>;
    success?: boolean;
    result?: string;
    error?: string;
  };
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  model: Model;
  createdAt: number;
  updatedAt: number;
}

export interface Model {
  id: string; // e.g., "gpt-4o"
  provider: Provider;
  displayName: string; // e.g., "GPT-4o"
  description: string;
  capabilities: ModelCapabilities;
}

export interface ModelCapabilities {
  streaming: boolean;
  images: boolean;
  embeddings: boolean;
  reranking: boolean;
  function_calling: boolean;
  thinking_budget: boolean;
  token_window: number;
}

export interface Provider {
  id: string; // e.g., "openai"
  name: string; // e.g., "OpenAI"
  logo: string; // Path to logo
  apiKeyName: string; // Environment variable name
  models: Model[];
  isConfigured: boolean;
}

export interface ApiKeyConfig {
  provider: string;
  key: string;
  isValid: boolean;
}

export interface AppSettings {
  apiKeys: ApiKeyConfig[];
  defaultModel: string;
  theme: 'light' | 'dark' | 'system';
  showTokenCounts: boolean;
  showCosts: boolean;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
}

export interface LLuMinaryResponse {
  response: string;
  usage: UsageData;
  messages: Array<{
    message_type: string;
    message: string;
    image_paths?: string[];
    image_urls?: string[];
  }>;
}
