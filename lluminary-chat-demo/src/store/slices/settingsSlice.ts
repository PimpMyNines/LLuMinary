import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { AppSettings, ApiKeyConfig } from '../../types';

const initialState: AppSettings = {
  apiKeys: [],
  defaultModel: 'gpt-4o-mini',
  theme: 'light',
  showTokenCounts: true,
  showCosts: true,
  temperature: 0.7,
  maxTokens: 1000,
  systemPrompt: 'You are a helpful assistant.',
};

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    setApiKey: (state, action: PayloadAction<{ provider: string; key: string; isValid: boolean }>) => {
      const { provider, key, isValid } = action.payload;
      const existing = state.apiKeys.findIndex(k => k.provider === provider);

      if (existing !== -1) {
        state.apiKeys[existing] = { provider, key, isValid };
      } else {
        state.apiKeys.push({ provider, key, isValid });
      }
    },

    removeApiKey: (state, action: PayloadAction<string>) => {
      state.apiKeys = state.apiKeys.filter(k => k.provider !== action.payload);
    },

    setDefaultModel: (state, action: PayloadAction<string>) => {
      state.defaultModel = action.payload;
    },

    setTheme: (state, action: PayloadAction<'light' | 'dark' | 'system'>) => {
      state.theme = action.payload;
    },

    setShowTokenCounts: (state, action: PayloadAction<boolean>) => {
      state.showTokenCounts = action.payload;
    },

    setShowCosts: (state, action: PayloadAction<boolean>) => {
      state.showCosts = action.payload;
    },

    setTemperature: (state, action: PayloadAction<number>) => {
      state.temperature = action.payload;
    },

    setMaxTokens: (state, action: PayloadAction<number>) => {
      state.maxTokens = action.payload;
    },

    setSystemPrompt: (state, action: PayloadAction<string>) => {
      state.systemPrompt = action.payload;
    },

    updateSettings: (state, action: PayloadAction<Partial<AppSettings>>) => {
      return { ...state, ...action.payload };
    },
  },
});

export const {
  setApiKey,
  removeApiKey,
  setDefaultModel,
  setTheme,
  setShowTokenCounts,
  setShowCosts,
  setTemperature,
  setMaxTokens,
  setSystemPrompt,
  updateSettings,
} = settingsSlice.actions;

export default settingsSlice.reducer;
