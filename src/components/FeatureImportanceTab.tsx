import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Cell as PieCell, Legend } from 'recharts';
import { api, FeatureImportance } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';
import { Brain, Target, Sparkles, BarChart3, TrendingUp, Zap } from 'lucide-react';

export const FeatureImportanceTab = () => {
  const [features, setFeatures] = useState<FeatureImportance[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchFeatures = async () => {
      try {
        console.log('Fetching feature importance data...');
        const data = await api.getFeatureImportance();
        console.log('Feature importance data:', data);
        
        // Handle both array and object responses
        const featuresArray = Array.isArray(data) ? data : (data?.feature_importance || []);
        setFeatures(featuresArray);
      } catch (error) {
        console.error('Failed to fetch feature importance:', error);
        setError('Failed to load feature importance data');
        setFeatures([]);
      } finally {
        setLoading(false);
      }
    };

    fetchFeatures();
  }, []);

  if (loading) {
    console.log('Loading feature importance...');
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <div className="flex min-h-[400px] items-center justify-center p-6">
        <div className="text-center">
          <p className="text-lg text-red-600 dark:text-red-400">{error}</p>
          <p className="mt-2 text-sm text-slate-500 dark:text-slate-500">Please check the console for more details</p>
        </div>
      </div>
    );
  }

  if (!features || features.length === 0) {
    return (
      <div className="flex min-h-[400px] items-center justify-center p-6">
        <div className="text-center">
          <p className="text-lg text-slate-600 dark:text-slate-400">No feature importance data available</p>
          <p className="mt-2 text-sm text-slate-500 dark:text-slate-500">Please check your data source</p>
        </div>
      </div>
    );
  }

  console.log('Rendering feature importance with', features.length, 'features');

  const colors = ['#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#6366f1', '#14b8a6'];

  const totalImportance = Array.isArray(features) 
    ? features.reduce((sum: number, feature: FeatureImportance) => sum + feature.importance, 0)
    : 0;

  return (
    <div className="space-y-8 p-6">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-3xl border border-blue-500/20 bg-gradient-to-br from-blue-600/10 via-cyan-500/10 to-transparent p-8 shadow-lg transition-shadow hover:shadow-2xl dark:border-blue-400/20 dark:from-blue-500/15 dark:via-sky-500/10">
        <div className="absolute -top-24 -right-12 h-64 w-64 rounded-full bg-blue-500/30 blur-3xl animate-pulse-slow" aria-hidden="true"></div>
        <div className="absolute -bottom-20 -left-10 h-64 w-64 rounded-full bg-cyan-500/20 blur-3xl animate-pulse-slow" aria-hidden="true"></div>
        <div className="relative grid gap-8 lg:grid-cols-[1.2fr_1fr]">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full bg-white/70 px-4 py-2 text-sm font-medium text-blue-700 shadow-sm ring-1 ring-blue-500/20 backdrop-blur dark:bg-slate-900/70 dark:text-blue-200">
              <Brain className="h-4 w-4" />
              Feature importance intelligence
            </div>
            <div className="space-y-3">
              <h2 className="text-3xl font-semibold text-slate-900 dark:text-white sm:text-4xl">
                Spotlight the signals that influence retention most
              </h2>
              <p className="text-base text-slate-600 dark:text-slate-300 sm:max-w-xl">
                Explore the top drivers behind churn predictions, understand their relative weight, and translate insights into targeted playbooks just like in the customers workspace.
              </p>
            </div>
            <div className="flex flex-wrap gap-3 text-sm text-slate-600 dark:text-slate-300">
              <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
                <Target className="h-4 w-4 text-blue-500" />
                {features.length} tracked features
              </span>
              <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
                <Sparkles className="h-4 w-4 text-blue-500" />
                {(totalImportance * 100).toFixed(0)}% total attribution
              </span>
            </div>
          </div>

          <div className="relative overflow-hidden rounded-3xl bg-white/80 p-6 shadow-xl ring-1 ring-slate-200/80 backdrop-blur dark:bg-slate-900/80 dark:ring-slate-800/80">
            <div className="absolute -right-6 -top-6 h-32 w-32 rounded-full bg-blue-500/20 blur-2xl" aria-hidden="true"></div>
            <div className="relative space-y-4">
              <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">Top signals</p>
              {features.slice(0, 3).map((feature, index) => (
                <div key={feature.feature} className="group space-y-2 rounded-2xl border border-slate-200/70 bg-white/80 p-4 backdrop-blur transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg dark:border-slate-800/70 dark:bg-slate-900/70">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl font-bold text-white shadow-lg transition-transform duration-300 group-hover:scale-110"
                        style={{ backgroundColor: colors[index % colors.length] }}
                      >
                        {index + 1}
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-slate-900 dark:text-white">{feature.feature}</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">Primary churn driver</p>
                      </div>
                    </div>
                    <span className="rounded-full bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-600 ring-1 ring-blue-500/20 dark:bg-blue-500/20 dark:text-blue-300">
                      {(feature.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                    {feature.description || 'Key factor influencing customer risk for this model.'}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Core chart */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.2fr_1fr]">
        <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-6 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
          <div className="mb-6 flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-lg">
              <BarChart3 className="h-6 w-6" />
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Top feature contribution</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Ranked by SHAP importance</p>
            </div>
            <div className="inline-flex items-center gap-2 rounded-full bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-600 ring-1 ring-blue-500/20 dark:bg-blue-500/20 dark:text-blue-200">
              <Sparkles className="h-3 w-3" />
              Live data
            </div>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={features.slice(0, 10)}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-700" />
                <XAxis
                  type="number"
                  domain={[0, Math.max(...features.map((f) => f.importance), 1)]}
                  tick={{ fill: 'currentColor', className: 'text-slate-600 dark:text-slate-400' }}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fill: 'currentColor', className: 'text-slate-900 dark:text-white' }}
                  width={140}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    border: '1px solid #e5e7eb',
                    borderRadius: '0.75rem',
                    color: '#111827'
                  }}
                  cursor={{ fill: 'rgba(59, 130, 246, 0.08)' }}
                  formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Importance']}
                />
                <Bar dataKey="importance" name="Feature Importance" radius={[0, 12, 12, 0]}>
                  {features.slice(0, 10).map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-6 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg">
                <TrendingUp className="h-6 w-6" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Importance distribution</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                  Top six contributors to churn
                </p>
              </div>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={features.slice(0, 6)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry) => `${entry.feature} ${(entry.importance * 100).toFixed(0)}%`}
                    outerRadius={100}
                    dataKey="importance"
                  >
                    {features.slice(0, 6).map((_entry, index) => (
                      <PieCell key={`cell-${index}`} fill={colors[index % colors.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #e5e7eb',
                      borderRadius: '0.75rem',
                      color: '#111827'
                    }}
                    formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Importance']}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-6 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500 text-white shadow-lg">
                <Zap className="h-6 w-6" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Activation playbook</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">Action-ready insights</p>
              </div>
            </div>
            <ul className="space-y-3 text-sm text-slate-600 dark:text-slate-300">
              {features.slice(0, 4).map((feature, index) => (
                <li key={feature.feature} className="group flex items-start gap-3 rounded-2xl border border-slate-200/70 bg-white/70 p-3 backdrop-blur transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md dark:border-slate-800/70 dark:bg-slate-900/70">
                  <span
                    className="mt-1 h-2 w-2 flex-shrink-0 rounded-full shadow-sm transition-transform duration-300 group-hover:scale-125"
                    style={{ backgroundColor: colors[index % colors.length] }}
                  ></span>
                  <div className="flex-1">
                    <p className="font-semibold text-slate-900 dark:text-white">{feature.feature}</p>
                    <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                      {feature.description || 'Focus outreach on profiles where this signal spikes.'}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Detail grid */}
      <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-6 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 text-white shadow-lg">
              <Target className="h-6 w-6" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Signal library</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Complete feature catalog</p>
            </div>
          </div>
          <span className="inline-flex items-center gap-2 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-600 ring-1 ring-emerald-500/20 dark:bg-emerald-500/20 dark:text-emerald-300">
            <Brain className="h-3 w-3" />
            {features.length} features
          </span>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {features.map((feature, index) => (
            <div
              key={feature.feature}
              className="group space-y-2 rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg dark:border-slate-800/70 dark:bg-slate-900/70"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <div
                    className="h-2 w-2 flex-shrink-0 rounded-full shadow-sm transition-transform duration-300 group-hover:scale-125"
                    style={{ backgroundColor: colors[index % colors.length] }}
                  ></div>
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">{feature.feature}</p>
                </div>
                <span className="rounded-full bg-blue-500/10 px-2.5 py-1 text-xs font-semibold text-blue-600 ring-1 ring-blue-500/20 dark:bg-blue-500/20 dark:text-blue-300">
                  {(feature.importance * 100).toFixed(1)}%
                </span>
              </div>
              <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                {feature.description || 'Feature importance score for churn prediction models.'}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};