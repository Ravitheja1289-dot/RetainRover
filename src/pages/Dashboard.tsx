import { useState } from 'react';
import { Users, BarChart3, Map, Lightbulb } from 'lucide-react';
import { Header } from '../components/Header';
import { Tabs } from '../components/Tabs';
import { CustomersTab } from '../components/CustomersTab';
import { FeatureImportanceTab } from '../components/FeatureImportanceTab';
import { RegionalTrendsTab } from '../components/RegionalTrendsTab';
import { InsightsTab } from '../components/InsightsTab';

export const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('customers');

  const tabs = [
    { id: 'customers', label: 'Customers', icon: <Users className="w-4 h-4" /> },
    { id: 'features', label: 'Feature Importance', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'regions', label: 'Regional Trends', icon: <Map className="w-4 h-4" /> },
    { id: 'insights', label: 'Insights', icon: <Lightbulb className="w-4 h-4" /> }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'customers':
        return <CustomersTab />;
      case 'features':
        return <FeatureImportanceTab />;
      case 'regions':
        return <RegionalTrendsTab />;
      case 'insights':
        return <InsightsTab />;
      default:
        return <CustomersTab />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <Header />
      <div className="bg-white dark:bg-gray-900">
        <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />
      </div>
      <main>
        {renderTabContent()}
      </main>
    </div>
  );
};
