import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { api, RegionalTrend } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';
import { TrendingUp, Users, Clock } from 'lucide-react';

export const RegionalTrendsTab = () => {
  const [trends, setTrends] = useState<RegionalTrend[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTrends = async () => {
      try {
        const data = await api.getRegionalTrends();
        setTrends(data);
      } catch (error) {
        console.error('Failed to fetch regional trends:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchTrends();
  }, []);

  if (loading) return <LoadingSpinner />;

  const getRiskColor = (rate: number) => {
    if (rate < 35) return 'bg-green-500';
    if (rate < 55) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getRiskBorderColor = (rate: number) => {
    if (rate < 35) return 'border-green-200 dark:border-green-800';
    if (rate < 55) return 'border-orange-200 dark:border-orange-800';
    return 'border-red-200 dark:border-red-800';
  };

  const getRiskBgColor = (rate: number) => {
    if (rate < 35) return 'bg-green-50 dark:bg-green-900/20';
    if (rate < 55) return 'bg-orange-50 dark:bg-orange-900/20';
    return 'bg-red-50 dark:bg-red-900/20';
  };

  return (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {trends.map((trend) => (
          <div
            key={trend.region}
            className={`rounded-lg border-2 ${getRiskBorderColor(trend.churnRate)} ${getRiskBgColor(trend.churnRate)} p-5 transition-all hover:shadow-md`}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                {trend.region}
              </h3>
              <div className={`w-3 h-3 rounded-full ${getRiskColor(trend.churnRate)}`}></div>
            </div>

            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Churn Rate</span>
                  <span className="text-lg font-bold text-gray-900 dark:text-white">
                    {trend.churnRate.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full ${getRiskColor(trend.churnRate)}`}
                    style={{ width: `${Math.min(trend.churnRate, 100)}%` }}
                  ></div>
                </div>
              </div>

              <div className="flex items-center space-x-2 text-xs text-gray-600 dark:text-gray-400">
                <Users className="w-4 h-4" />
                <span>{trend.totalCustomers.toLocaleString()} customers</span>
              </div>

              <div className="flex items-center space-x-2 text-xs text-gray-600 dark:text-gray-400">
                <Clock className="w-4 h-4" />
                <span>Avg tenure: {trend.avgTenure} years</span>
              </div>

              <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-2 text-xs">
                  <TrendingUp className="w-4 h-4 text-red-500" />
                  <span className="text-gray-700 dark:text-gray-300">
                    {trend.churnedCustomers.toLocaleString()} at risk
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Churn Rate Comparison by Region
        </h2>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={trends}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis
                dataKey="region"
                tick={{ fill: 'currentColor', className: 'text-gray-900 dark:text-white' }}
              />
              <YAxis
                tick={{ fill: 'currentColor', className: 'text-gray-600 dark:text-gray-400' }}
                label={{ value: 'Rate (%)', angle: -90, position: 'insideLeft', fill: 'currentColor', className: 'text-gray-600 dark:text-gray-400' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.5rem',
                  color: '#111827'
                }}
                cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
              />
              <Legend />
              <Bar dataKey="churnRate" name="Churn Rate (%)" fill="#ef4444" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
          Average Tenure by Region
        </h2>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={trends}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis
                dataKey="region"
                tick={{ fill: 'currentColor', className: 'text-gray-900 dark:text-white' }}
              />
              <YAxis
                tick={{ fill: 'currentColor', className: 'text-gray-600 dark:text-gray-400' }}
                label={{ value: 'Years', angle: -90, position: 'insideLeft', fill: 'currentColor', className: 'text-gray-600 dark:text-gray-400' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.5rem',
                  color: '#111827'
                }}
                cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
              />
              <Legend />
              <Bar dataKey="avgTenure" name="Average Tenure (years)" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
