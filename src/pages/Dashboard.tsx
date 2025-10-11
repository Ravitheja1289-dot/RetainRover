import { useState } from 'react';
import { Users, BarChart3, Map, Lightbulb } from 'lucide-react';
import { Header } from '../components/Header';
import { Tabs } from '../components/Tabs';
import { CustomersTab } from '../components/CustomersTab';
import { FeatureImportanceTab } from '../components/FeatureImportanceTab';
import { RegionalTrendsTab } from '../components/RegionalTrendsTab';
import { InsightsTab } from '../components/InsightsTab';
import { ErrorBoundary } from '../components/ErrorBoundary';

export const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('customers');

  const tabs = [
    { id: 'customers', label: 'Customers', icon: <Users className="w-4 h-4" /> },
    { id: 'features', label: 'Feature Importance', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'regions', label: 'Regional Trends', icon: <Map className="w-4 h-4" /> },
    { id: 'insights', label: 'Insights', icon: <Lightbulb className="w-4 h-4" /> }
  ];

  const handleTabChange = (tabId: string) => {
    console.log('Switching to tab:', tabId);
    setActiveTab(tabId);
  };

  const renderTabContent = () => {
    console.log('Rendering tab content for:', activeTab);
    try {
      switch (activeTab) {
        case 'customers':
          return <CustomersTab />;
        case 'features':
          console.log('Rendering FeatureImportanceTab');
          return <FeatureImportanceTab />;
        case 'regions':
          return <RegionalTrendsTab />;
        case 'insights':
          return <InsightsTab />;
        default:
          return <CustomersTab />;
      }
    } catch (error) {
      console.error('Error rendering tab content:', error);
      return (
        <div className="flex min-h-[400px] items-center justify-center p-6">
          <div className="text-center">
            <p className="text-lg text-red-600 dark:text-red-400">Error loading tab content</p>
            <p className="mt-2 text-sm text-slate-500 dark:text-slate-500">Check console for details</p>
          </div>
        </div>
      );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950">
      <Header />
      <div className="bg-white/80 backdrop-blur-xl dark:bg-slate-900/80">
        <Tabs tabs={tabs} activeTab={activeTab} onChange={handleTabChange} />
      </div>
      <main>
        <ErrorBoundary key={activeTab}>
          {renderTabContent()}
        </ErrorBoundary>
      </main>
    </div>
  );
};
