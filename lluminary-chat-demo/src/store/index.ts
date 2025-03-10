import { configureStore } from '@reduxjs/toolkit';
import conversationsReducer from './slices/conversationsSlice';
import modelsReducer from './slices/modelsSlice';
import settingsReducer from './slices/settingsSlice';
import uiReducer from './slices/uiSlice';

export const store = configureStore({
  reducer: {
    conversations: conversationsReducer,
    models: modelsReducer,
    settings: settingsReducer,
    ui: uiReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
