import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import { api, ModelMetrics } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';
import { TrendingUp, Award, Target, Zap, Brain, CheckCircle } from 'lucide-react';

export const ModelComparisonTab = () => {
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<string>('roc_auc');
  const [apiHealth, setApiHealth] = useState<any>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metricsData, healthData] = await Promise.all([
          api.getModelMetrics(),
          api.getApiHealth()
        ]);
        setMetrics(metricsData);
        setApiHealth(healthData);
      } catch (error) {
        console.error('Failed to fetch model data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <LoadingSpinner />;

  const bestModel = metrics.reduce((best, current) => 
    current.roc_auc > best.roc_auc ? current : best, metrics[0]);

  const metricOptions = [
    { key: 'accuracy', label: 'Accuracy', icon: Target, color: '#3b82f6' },
    { key: 'precision', label: 'Precision', icon: CheckCircle, color: '#10b981' },
    { key: 'recall', label: 'Recall', icon: TrendingUp, color: '#f59e0b' },
    { key: 'f1_score', label: 'F1-Score', icon: Award, color: '#8b5cf6' },
    { key: 'roc_auc', label: 'ROC-AUC', icon: Brain, color: '#ef4444' }
  ];

  const getPerformanceLevel = (value: number, metric: string) => {
    const thresholds: Record<string, { excellent: number; good: number; fair: number }> = {
      accuracy: { excellent: 0.95, good: 0.85, fair: 0.75 },
      precision: { excellent: 0.95, good: 0.85, fair: 0.75 },
      recall: { excellent: 0.95, good: 0.85, fair: 0.75 },
      f1_score: { excellent: 0.95, good: 0.85, fair: 0.75 },
      roc_auc: { excellent: 0.95, good: 0.85, fair: 0.75 }
    };

    const threshold = thresholds[metric];
    if (value >= threshold.excellent) return { level: 'Excellent', color: 'text-green-600', bg: 'bg-green-100' };
    if (value >= threshold.good) return { level: 'Good', color: 'text-blue-600', bg: 'bg-blue-100' };
    if (value >= threshold.fair) return { level: 'Fair', color: 'text-yellow-600', bg: 'bg-yellow-100' };
    return { level: 'Poor', color: 'text-red-600', bg: 'bg-red-100' };
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Model Performance Comparison</h2>
            <p className="text-purple-100">
              Compare machine learning models and select the best performer for your use case
            </p>
          </div>
          {apiHealth && (
            <div className="text-right">
              <div className="flex items-center space-x-2 text-green-200">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-sm">API Online</span>
              </div>
              <p className="text-xs text-purple-200 mt-1">
                {apiHealth.models_loaded} models loaded
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Best Model Highlight */}
      {bestModel && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center space-x-4">
            <div className="bg-gradient-to-br from-yellow-400 to-orange-500 p-3 rounded-full">
              <Award className="w-8 h-8 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                Best Performing Model
              </h3>
              <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {bestModel.model_name}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                ROC-AUC: {(bestModel.roc_auc * 100).toFixed(1)}% | 
                Accuracy: {(bestModel.accuracy * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Metric Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Select Performance Metric
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {metricOptions.map((option) => {
            const Icon = option.icon;
            return (
              <button
                key={option.key}
                onClick={() => setSelectedMetric(option.key)}
                className={`p-3 rounded-lg border-2 transition-all ${
                  selectedMetric === option.key
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                <div className="flex flex-col items-center space-y-2">
                  <Icon className="w-6 h-6" style={{ color: option.color }} />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {option.label}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Performance Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          {metricOptions.find(m => m.key === selectedMetric)?.label} Comparison
        </h3>
        
        <div className="h-80 mb-6">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={metrics.map(metric => ({
                ...metric,
                value: metric[selectedMetric as keyof ModelMetrics] as number,
                formattedValue: ((metric[selectedMetric as keyof ModelMetrics] as number) * 100).toFixed(1)
              }))}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis
                dataKey="model_name"
                tick={{ fill: 'currentColor', className: 'text-gray-900 dark:text-white' }}
              />
              <YAxis
                tick={{ fill: 'currentColor', className: 'text-gray-600 dark:text-gray-400' }}
                domain={[0, 1]}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.5rem',
                  color: '#111827'
                }}
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, selectedMetric.replace('_', ' ').toUpperCase()]}
                cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
              />
              <Bar 
                dataKey="value" 
                radius={[4, 4, 0, 0]}
                fill={metricOptions.find(m => m.key === selectedMetric)?.color || '#3b82f6'}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {metrics.map((metric) => {
            const performance = getPerformanceLevel(
              metric[selectedMetric as keyof ModelMetrics] as number,
              selectedMetric
            );
            return (
              <div
                key={metric.model_name}
                className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow"
              >
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-gray-900 dark:text-white">
                    {metric.model_name}
                  </h4>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${performance.bg} ${performance.color}`}>
                    {performance.level}
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Current Metric</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {((metric[selectedMetric as keyof ModelMetrics] as number) * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">CV Score</span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {(metric.cv_score_mean * 100).toFixed(1)}% ± {(metric.cv_score_std * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className="h-2 rounded-full"
                      style={{
                        width: `${(metric[selectedMetric as keyof ModelMetrics] as number) * 100}%`,
                        backgroundColor: metricOptions.find(m => m.key === selectedMetric)?.color || '#3b82f6'
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Comprehensive Metrics Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Comprehensive Performance Metrics
        </h3>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-900/50 border-b border-gray-200 dark:border-gray-700">
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Accuracy
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Precision
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Recall
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  F1-Score
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  ROC-AUC
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  CV Score
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {metrics.map((metric) => (
                <tr key={metric.model_name} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                  <td className="px-4 py-4 whitespace-nowrap">
                    <div className="flex items-center space-x-3">
                      <Brain className="w-5 h-5 text-gray-400" />
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {metric.model_name}
                      </span>
                      {metric.model_name === bestModel?.model_name && (
                        <Award className="w-4 h-4 text-yellow-500" />
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className="text-sm text-gray-900 dark:text-white">
                      {(metric.accuracy * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className="text-sm text-gray-900 dark:text-white">
                      {(metric.precision * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className="text-sm text-gray-900 dark:text-white">
                      {(metric.recall * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className="text-sm text-gray-900 dark:text-white">
                      {(metric.f1_score * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className="text-sm text-gray-900 dark:text-white">
                      {(metric.roc_auc * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className="text-sm text-gray-900 dark:text-white">
                      {(metric.cv_score_mean * 100).toFixed(1)}% ± {(metric.cv_score_std * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
