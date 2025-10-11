import { ReactNode } from 'react';

interface TabsProps {
  tabs: { id: string; label: string; icon: ReactNode }[];
  activeTab: string;
  onChange: (tabId: string) => void;
}

export const Tabs = ({ tabs, activeTab, onChange }: TabsProps) => {
  return (
    <div className="border-b border-slate-200/70 dark:border-slate-800/70">
      <nav className="flex gap-2 px-6" aria-label="Tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={`
              flex items-center gap-2 rounded-t-xl px-4 py-3 font-semibold text-sm transition-all
              ${
                activeTab === tab.id
                  ? 'bg-gradient-to-br from-blue-50 to-cyan-50 text-blue-600 shadow-sm ring-1 ring-blue-500/20 dark:from-blue-900/30 dark:to-cyan-900/30 dark:text-blue-400 dark:ring-blue-400/20'
                  : 'text-slate-500 hover:bg-slate-50/80 hover:text-slate-700 dark:text-slate-400 dark:hover:bg-slate-800/50 dark:hover:text-slate-300'
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
