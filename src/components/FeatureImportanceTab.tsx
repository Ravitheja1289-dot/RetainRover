import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, PieChart, Pie, Cell as PieCell } from 'recharts';
import { api, FeatureImportance } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';
import { Info, TrendingUp, Brain, Target } from 'lucide-react';

export const FeatureImportanceTab = () => {
  const [features, setFeatures] = useState<FeatureImportance[]>([]);
  const [modelFeatures, setModelFeatures] = useState<any>(null);
  const [selectedModel, setSelectedModel] = useState<string>('RandomForest');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFeatures = async () => {
      try {
        const [staticFeatures, dynamicFeatures] = await Promise.all([
          api.getFeatureImportance(),
          api.getFeatureImportance(selectedModel).catch(() => null)
        ]);
        setFeatures(staticFeatures);
        setModelFeatures(dynamicFeatures);
      } catch (error) {
        console.error('Failed to fetch feature importance:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchFeatures();
  }, [selectedModel]);

  if (loading) return <LoadingSpinner />;

  const colors = ['#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1', '#14b8a6'];

  const displayFeatures = modelFeatures?.feature_importance || features;
  const totalImportance = displayFeatures.reduce((sum: number, feature: any) => sum + feature.importance, 0);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Feature Importance Analysis</h2>
            <p className="text-blue-100">
              Understand which factors most influence credit risk predictions using SHAP values
            </p>
          </div>
          <div className="text-right">
            <div className="flex items-center space-x-2 text-blue-200">
              <Brain className="w-5 h-5" />
              <span className="text-sm">AI-Powered Insights</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Select Model for Analysis
          </h3>
          <div className="flex space-x-2">
            {['RandomForest', 'GradientBoosting', 'ExtraTrees'].map((model) => (
              <button
                key={model}
                onClick={() => setSelectedModel(model)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedModel === model
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {model}
              </button>
            ))}
          </div>
        </div>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-medium text-blue-900 dark:text-blue-100">
                SHAP Feature Importance
              </h3>
              <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                These features have the highest impact on credit risk predictions. Higher values indicate stronger influence on the model's decisions.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Feature Importance - {selectedModel}
          </h2>
          <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
            <Target className="w-4 h-4" />
            <span>Total Features: {displayFeatures.length}</span>
          </div>
        </div>

        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={displayFeatures.slice(0, 10)}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis
                type="number"
                domain={[0, Math.max(...displayFeatures.map((f: any) => f.importance))]}
                tick={{ fill: 'currentColor', className: 'text-gray-600 dark:text-gray-400' }}
              />
              <YAxis
                type="category"
                dataKey="feature"
                tick={{ fill: 'currentColor', className: 'text-gray-900 dark:text-white' }}
                width={120}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.5rem',
                  color: '#111827'
                }}
                cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
                formatter={(value: number, _name: string, props: any) => [
                  `${value.toFixed(4)}`,
                  'Importance Score'
                ]}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {displayFeatures.slice(0, 10).map((_entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Additional Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Feature Importance Distribution
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={displayFeatures.slice(0, 6)}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ feature, percent }) => `${feature} (${(percent * 100).toFixed(0)}%)`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="importance"
                >
                  {displayFeatures.slice(0, 6).map((_entry: any, index: number) => (
                    <PieCell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => [value.toFixed(4), 'Importance']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Features Summary */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Top Contributing Features
          </h3>
          <div className="space-y-4">
            {displayFeatures.slice(0, 6).map((feature: any, index: number) => (
              <div key={feature.feature} className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold"
                       style={{ backgroundColor: colors[index % colors.length] }}>
                    {index + 1}
                  </div>
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {feature.feature}
                    </span>
                    <span className="text-sm font-bold text-gray-600 dark:text-gray-400">
                      {(feature.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="h-2 rounded-full"
                      style={{ 
                        width: `${(feature.importance / Math.max(...displayFeatures.map((f: any) => f.importance))) * 100}%`,
                        backgroundColor: colors[index % colors.length]
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Feature Details */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Feature Details & Descriptions
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {displayFeatures.map((feature: any, index: number) => (
            <div
              key={feature.feature}
              className="flex items-start space-x-3 p-4 rounded-lg bg-gray-50 dark:bg-gray-900/50"
            >
              <div
                className="w-3 h-3 rounded-full mt-1 flex-shrink-0"
                style={{ backgroundColor: colors[index % colors.length] }}
              ></div>
              <div>
                <h4 className="text-sm font-medium text-gray-900 dark:text-white">
                  {feature.feature}
                </h4>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                  {feature.description || 'Feature importance score for model prediction'}
                </p>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-500 mt-1">
                  Importance: {(feature.importance * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
