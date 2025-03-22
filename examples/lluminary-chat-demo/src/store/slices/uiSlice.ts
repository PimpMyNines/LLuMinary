import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UiState {
  sidebarOpen: boolean;
  settingsOpen: boolean;
  activeTab: 'chat' | 'settings' | 'about';
  showFlowDiagram: boolean;
  isStreaming: boolean;
}

const initialState: UiState = {
  sidebarOpen: true,
  settingsOpen: false,
  activeTab: 'chat',
  showFlowDiagram: true,
  isStreaming: false,
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },

    toggleSettings: (state) => {
      state.settingsOpen = !state.settingsOpen;
    },

    setActiveTab: (state, action: PayloadAction<'chat' | 'settings' | 'about'>) => {
      state.activeTab = action.payload;
    },

    toggleFlowDiagram: (state) => {
      state.showFlowDiagram = !state.showFlowDiagram;
    },

    setIsStreaming: (state, action: PayloadAction<boolean>) => {
      state.isStreaming = action.payload;
    },
  },
});

export const {
  toggleSidebar,
  toggleSettings,
  setActiveTab,
  toggleFlowDiagram,
  setIsStreaming,
} = uiSlice.actions;

export default uiSlice.reducer;
