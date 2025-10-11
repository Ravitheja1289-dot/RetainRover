import { ReactNode } from 'react';

interface TabsProps {
  tabs: { id: string; label: string; icon: ReactNode }[];
  activeTab: string;
  onChange: (tabId: string) => void;
}

export const Tabs = ({ tabs, activeTab, onChange }: TabsProps) => {
  return (
    <div className="border-b border-gray-200 dark:border-gray-700">
      <nav className="flex space-x-8 px-6" aria-label="Tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={`
              flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors
              ${
                activeTab === tab.id
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
              }
            `}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
};
