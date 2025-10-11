import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api, FeatureImportance } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';
import { Info } from 'lucide-react';

export const FeatureImportanceTab = () => {
  const [features, setFeatures] = useState<FeatureImportance[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFeatures = async () => {
      try {
        const data = await api.getFeatureImportance();
        setFeatures(data);
      } catch (error) {
        console.error('Failed to fetch feature importance:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchFeatures();
  }, []);

  if (loading) return <LoadingSpinner />;

  const colors = ['#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1', '#14b8a6'];

  return (
    <div className="p-6 space-y-6">
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <Info className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="text-sm font-medium text-blue-900 dark:text-blue-100">
              SHAP Feature Importance
            </h3>
            <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
              These features have the highest impact on churn predictions. Higher values indicate stronger influence on the model's decisions.
            </p>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Global Feature Importance
        </h2>

        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={features}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis
                type="number"
                domain={[0, 0.5]}
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
                  `${value.toFixed(3)}`,
                  props.payload.description
                ]}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {features.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
          {features.map((feature, index) => (
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
                  {feature.description}
                </p>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-500 mt-1">
                  Importance: {(feature.importance * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
