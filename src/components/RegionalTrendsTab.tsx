import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { api, RegionalTrend } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';
import { TrendingUp, Users, Clock, Map, Sparkles, MapPin } from 'lucide-react';

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
    <div className="space-y-8 p-6">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl border border-blue-500/20 bg-gradient-to-br from-blue-600/10 via-cyan-500/10 to-transparent p-8 shadow-lg transition-shadow hover:shadow-2xl dark:border-blue-400/20 dark:from-blue-500/15 dark:via-sky-500/10">
        <div className="absolute -top-24 -right-12 h-64 w-64 rounded-full bg-blue-500/30 blur-3xl animate-pulse-slow" aria-hidden="true"></div>
        <div className="absolute -bottom-20 -left-10 h-64 w-64 rounded-full bg-cyan-500/20 blur-3xl animate-pulse-slow" aria-hidden="true"></div>
        <div className="relative space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full bg-white/70 px-4 py-2 text-sm font-medium text-blue-700 shadow-sm ring-1 ring-blue-500/20 backdrop-blur dark:bg-slate-900/70 dark:text-blue-200">
            <Map className="h-4 w-4" />
            Geographic intelligence dashboard
          </div>
          <div className="space-y-3">
            <h2 className="text-3xl font-semibold text-slate-900 dark:text-white sm:text-4xl">
              Decode regional churn patterns and optimize by location
            </h2>
            <p className="text-base text-slate-600 dark:text-slate-300 sm:max-w-3xl">
              Identify geographic hotspots, understand regional differences in customer behavior, and tailor retention strategies for maximum local impact.
            </p>
          </div>
          <div className="flex flex-wrap gap-3 text-sm text-slate-600 dark:text-slate-300">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
              <MapPin className="h-4 w-4 text-blue-500" />
              {trends.length} regions tracked
            </span>
            <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
              <Sparkles className="h-4 w-4 text-blue-500" />
              Real-time analytics
            </span>
          </div>
        </div>
      </div>

      {/* Regional Cards */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        {trends.map((trend) => (
          <div
            key={trend.region}
            className={`group relative overflow-hidden rounded-3xl border ${getRiskBorderColor(trend.churnRate)} ${getRiskBgColor(trend.churnRate)} p-6 shadow-lg backdrop-blur transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl`}
          >
            <div className="relative space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                  {trend.region}
                </h3>
                <div className={`h-3 w-3 rounded-full ${getRiskColor(trend.churnRate)} shadow-lg`}></div>
              </div>

              <div className="space-y-3">
                <div>
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">Churn Rate</span>
                    <span className="text-2xl font-bold text-slate-900 dark:text-white">
                      {trend.churnRate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/80 dark:bg-slate-700/80">
                    <div
                      className={`h-2 rounded-full ${getRiskColor(trend.churnRate)} transition-all duration-500`}
                      style={{ width: `${Math.min(trend.churnRate, 100)}%` }}
                    ></div>
                  </div>
                </div>

                <div className="space-y-2 rounded-2xl border border-slate-200/70 bg-white/70 p-3 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
                  <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
                    <Users className="h-4 w-4 text-blue-500" />
                    <span className="font-medium">{trend.totalCustomers.toLocaleString()} customers</span>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
                    <Clock className="h-4 w-4 text-blue-500" />
                    <span className="font-medium">Avg tenure: {trend.avgTenure} years</span>
                  </div>

                  <div className="flex items-center gap-2 border-t border-slate-200/70 pt-2 text-xs dark:border-slate-800/70">
                    <TrendingUp className="h-4 w-4 text-rose-500" />
                    <span className="font-semibold text-slate-900 dark:text-white">
                      {trend.churnedCustomers.toLocaleString()} at risk
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Churn Rate Chart */}
      <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-6 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-rose-500 to-orange-500 text-white shadow-lg">
            <TrendingUp className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
              Churn Rate Comparison by Region
            </h2>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Identify high-risk geographic areas
            </p>
          </div>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={trends}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-700" />
              <XAxis
                dataKey="region"
                tick={{ fill: 'currentColor', className: 'text-slate-900 dark:text-white' }}
              />
              <YAxis
                tick={{ fill: 'currentColor', className: 'text-slate-600 dark:text-slate-400' }}
                label={{ value: 'Rate (%)', angle: -90, position: 'insideLeft', fill: 'currentColor', className: 'text-slate-600 dark:text-slate-400' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.75rem',
                  color: '#111827'
                }}
                cursor={{ fill: 'rgba(59, 130, 246, 0.08)' }}
              />
              <Legend />
              <Bar dataKey="churnRate" name="Churn Rate (%)" fill="#ef4444" radius={[12, 12, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tenure Chart */}
      <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-6 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-lg">
            <Clock className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white">
              Average Tenure by Region
            </h2>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Customer loyalty patterns across locations
            </p>
          </div>
        </div>

        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={trends}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-700" />
              <XAxis
                dataKey="region"
                tick={{ fill: 'currentColor', className: 'text-slate-900 dark:text-white' }}
              />
              <YAxis
                tick={{ fill: 'currentColor', className: 'text-slate-600 dark:text-slate-400' }}
                label={{ value: 'Years', angle: -90, position: 'insideLeft', fill: 'currentColor', className: 'text-slate-600 dark:text-slate-400' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  border: '1px solid #e5e7eb',
                  borderRadius: '0.75rem',
                  color: '#111827'
                }}
                cursor={{ fill: 'rgba(59, 130, 246, 0.08)' }}
              />
              <Legend />
              <Bar dataKey="avgTenure" name="Average Tenure (years)" fill="#3b82f6" radius={[12, 12, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
