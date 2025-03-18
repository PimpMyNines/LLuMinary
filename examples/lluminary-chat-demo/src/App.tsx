import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from './store';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import SettingsPanel from './components/SettingsPanel';
import { toggleSidebar, toggleSettings } from './store/slices/uiSlice';
import { Bars3Icon, Cog6ToothIcon } from '@heroicons/react/24/outline';

const App: React.FC = () => {
  const dispatch = useDispatch();
  const { sidebarOpen } = useSelector((state: RootState) => state.ui);
  const { providers } = useSelector((state: RootState) => state.models);
  const { apiKeys } = useSelector((state: RootState) => state.settings);

  // Check if any provider is configured
  const isAnyProviderConfigured = apiKeys.length > 0;

  // Open settings when no provider is configured
  useEffect(() => {
    if (!isAnyProviderConfigured && providers.length > 0) {
      dispatch(toggleSettings());
    }
  }, [isAnyProviderConfigured, providers, dispatch]);

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b px-4 py-3 flex items-center justify-between">
          <button
            onClick={() => dispatch(toggleSidebar())}
            className="md:hidden p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none"
          >
            <Bars3Icon className="w-6 h-6" />
          </button>

          <h1 className="text-xl font-semibold text-gray-800 hidden md:block">
            LLuMinary Demo Chat
          </h1>

          <button
            onClick={() => dispatch(toggleSettings())}
            className="p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none"
            title="Settings"
          >
            <Cog6ToothIcon className="w-6 h-6" />
          </button>
        </header>

        {/* Chat interface */}
        <main className="flex-1 overflow-hidden">
          <ChatInterface />
        </main>
      </div>

      {/* Settings panel */}
      <SettingsPanel />

      {/* Overlay when sidebar is open on mobile */}
      {sidebarOpen && (
        <div
          className="md:hidden fixed inset-0 z-10 bg-black bg-opacity-30"
          onClick={() => dispatch(toggleSidebar())}
        />
      )}
    </div>
  );
};

export default App;
