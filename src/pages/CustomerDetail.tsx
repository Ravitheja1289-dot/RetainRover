import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, User, MapPin, Calendar, DollarSign, FileText, Phone, AlertTriangle, CheckCircle, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { api, Customer } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';

export const CustomerDetail = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [customer, setCustomer] = useState<Customer | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCustomer = async () => {
      if (!id) return;
      try {
        const data = await api.getCustomerById(parseInt(id));
        setCustomer(data || null);
      } catch (error) {
        console.error('Failed to fetch customer:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchCustomer();
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
        <LoadingSpinner />
      </div>
    );
  }

  if (!customer) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-12">
            <p className="text-gray-600 dark:text-gray-400">Customer not found</p>
            <button
              onClick={() => navigate('/')}
              className="mt-4 text-blue-600 hover:text-blue-700 dark:text-blue-400"
            >
              Return to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  const getRiskLevel = (probability: number) => {
    if (probability < 30) return { label: 'Low Risk', color: 'text-green-600 dark:text-green-400', bg: 'bg-green-50 dark:bg-green-900/20', icon: CheckCircle, borderColor: 'border-green-200 dark:border-green-800' };
    if (probability < 60) return { label: 'Medium Risk', color: 'text-orange-600 dark:text-orange-400', bg: 'bg-orange-50 dark:bg-orange-900/20', icon: AlertCircle, borderColor: 'border-orange-200 dark:border-orange-800' };
    return { label: 'High Risk', color: 'text-red-600 dark:text-red-400', bg: 'bg-red-50 dark:bg-red-900/20', icon: AlertTriangle, borderColor: 'border-red-200 dark:border-red-800' };
  };

  const risk = getRiskLevel(customer.churnProbability);
  const RiskIcon = risk.icon;

  const featureContributions = [
    { feature: 'Claims Count', value: customer.claimsCount > 4 ? 0.35 : -0.15, impact: customer.claimsCount > 4 ? 'High' : 'Low' },
    { feature: 'Tenure', value: customer.tenure < 10 ? 0.28 : -0.22, impact: customer.tenure < 10 ? 'High' : 'Low' },
    { feature: 'Premium Amount', value: customer.premiumAmount > 1400 ? 0.22 : -0.10, impact: customer.premiumAmount > 1400 ? 'Medium' : 'Low' },
    { feature: 'Last Interaction', value: new Date().getTime() - new Date(customer.lastInteraction).getTime() > 60 * 24 * 60 * 60 * 1000 ? 0.18 : -0.12, impact: new Date().getTime() - new Date(customer.lastInteraction).getTime() > 60 * 24 * 60 * 60 * 1000 ? 'Medium' : 'Low' },
    { feature: 'Age', value: customer.age < 35 ? 0.08 : -0.05, impact: customer.age < 35 ? 'Low' : 'Low' }
  ];

  const getExplanation = () => {
    const reasons = [];

    if (customer.claimsCount > 4) {
      reasons.push(`filed ${customer.claimsCount} claims (high volume indicates dissatisfaction)`);
    }
    if (customer.tenure < 10) {
      reasons.push(`relatively short tenure of ${customer.tenure} years (newer customers more likely to switch)`);
    }
    if (customer.premiumAmount > 1400) {
      reasons.push(`high premium amount of $${customer.premiumAmount}/month (price sensitivity)`);
    }

    const daysSinceInteraction = Math.floor((new Date().getTime() - new Date(customer.lastInteraction).getTime()) / (1000 * 60 * 60 * 24));
    if (daysSinceInteraction > 60) {
      reasons.push(`no interaction in ${daysSinceInteraction} days (disengagement signal)`);
    }

    if (reasons.length === 0) {
      return 'This customer shows positive indicators with low claims, good tenure, and regular engagement.';
    }

    return `This customer is ${customer.churnProbability >= 60 ? 'likely' : customer.churnProbability >= 30 ? 'somewhat likely' : 'unlikely'} to churn because they have ${reasons.join(', ')}.`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950">
      <div className="mx-auto max-w-7xl space-y-8 p-6">
        <button
          onClick={() => navigate('/')}
          className="inline-flex items-center gap-2 rounded-xl bg-white/80 px-4 py-2 text-sm font-medium text-slate-600 shadow-sm ring-1 ring-slate-200/70 backdrop-blur transition-all hover:-translate-y-0.5 hover:shadow-lg dark:bg-slate-900/80 dark:text-slate-300 dark:ring-slate-800/70"
        >
          <ArrowLeft className="h-4 w-4" />
          <span>Back to Dashboard</span>
        </button>

        <div className={`relative overflow-hidden rounded-3xl border-2 ${risk.borderColor} ${risk.bg} p-8 shadow-xl backdrop-blur`}>
          <div className="relative">
            <div className="mb-6 flex flex-col items-start justify-between gap-6 lg:flex-row lg:items-center">
              <div className="flex items-center gap-4">
                <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 text-3xl font-semibold text-white shadow-lg">
                  {customer.name ? customer.name.charAt(0).toUpperCase() : '?'}
                </div>
                <div>
                  <h1 className="text-4xl font-bold text-slate-900 dark:text-white">
                    {customer.name}
                  </h1>
                  <p className="text-sm font-medium text-slate-600 dark:text-slate-400">Customer ID: {customer.id}</p>
                </div>
              </div>
              <div className="text-left lg:text-right">
                <div className={`mb-3 inline-flex items-center gap-2 rounded-full px-5 py-2 text-sm font-semibold ${risk.bg} ${risk.color} border-2 ${risk.borderColor} shadow-lg`}>
                  <RiskIcon className="h-5 w-5" />
                  <span>{risk.label}</span>
                </div>
                <div>
                  <span className="text-5xl font-bold text-slate-900 dark:text-white">
                    {customer.churnProbability}%
                  </span>
                  <p className="text-sm font-medium text-slate-600 dark:text-slate-400">Churn Probability</p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
              <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
                <div className="mb-2 flex items-center gap-2">
                  <MapPin className="h-5 w-5 text-blue-500" />
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Region</p>
                </div>
                <p className="text-base font-semibold text-slate-900 dark:text-white">{customer.region}</p>
              </div>
              <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
                <div className="mb-2 flex items-center gap-2">
                  <Calendar className="h-5 w-5 text-blue-500" />
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Age</p>
                </div>
                <p className="text-base font-semibold text-slate-900 dark:text-white">{customer.age} years</p>
              </div>
              <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
                <div className="mb-2 flex items-center gap-2">
                  <Calendar className="h-5 w-5 text-blue-500" />
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Tenure</p>
                </div>
                <p className="text-base font-semibold text-slate-900 dark:text-white">{customer.tenure} years</p>
              </div>
              <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
                <div className="mb-2 flex items-center gap-2">
                  <DollarSign className="h-5 w-5 text-blue-500" />
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Premium</p>
                </div>
                <p className="text-base font-semibold text-slate-900 dark:text-white">${customer.premiumAmount}/mo</p>
              </div>
              <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
                <div className="mb-2 flex items-center gap-2">
                  <FileText className="h-5 w-5 text-blue-500" />
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Claims</p>
                </div>
                <p className="text-base font-semibold text-slate-900 dark:text-white">{customer.claimsCount}</p>
              </div>
            </div>

            <div className="mt-6 rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
              <div className="flex items-center gap-2">
                <Phone className="h-4 w-4 text-blue-500" />
                <p className="text-sm font-medium text-slate-600 dark:text-slate-400">
                  Last interaction: <span className="font-semibold text-slate-900 dark:text-white">{new Date(customer.lastInteraction).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span>
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-8 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
          <h2 className="mb-4 text-2xl font-semibold text-slate-900 dark:text-white">
            Churn Prediction Explanation
          </h2>
          <p className="leading-relaxed text-slate-700 dark:text-slate-300">
            {getExplanation()}
          </p>
        </div>

        <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-8 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
          <h2 className="mb-6 text-2xl font-semibold text-slate-900 dark:text-white">
            Feature Contributions to Churn Score
          </h2>

          <div className="h-80 mb-6">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featureContributions}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
                <XAxis
                  type="number"
                  domain={[-0.3, 0.4]}
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
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {featureContributions.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {featureContributions.map((contribution) => (
              <div
                key={contribution.feature}
                className="flex items-center justify-between rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`h-3 w-3 rounded-full shadow-lg ${contribution.value > 0 ? 'bg-rose-500' : 'bg-emerald-500'}`}
                  ></div>
                  <span className="text-sm font-semibold text-slate-900 dark:text-white">
                    {contribution.feature}
                  </span>
                </div>
                <div className="text-right">
                  <span className={`text-sm font-bold ${contribution.value > 0 ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                    {contribution.value > 0 ? '+' : ''}{(contribution.value * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 rounded-2xl border border-blue-200/70 bg-blue-50/80 p-4 backdrop-blur dark:border-blue-800/70 dark:bg-blue-900/30">
            <p className="text-sm leading-relaxed text-blue-900 dark:text-blue-100">
              <span className="font-semibold">How to read this chart:</span> Positive values (red bars) increase churn probability,
              while negative values (green bars) decrease it. Longer bars indicate stronger influence on the prediction.
            </p>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-8 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
          <h2 className="mb-6 text-2xl font-semibold text-slate-900 dark:text-white">
            Recommended Actions
          </h2>
          <div className="space-y-4">
            {customer.churnProbability >= 60 && (
              <>
                <div className="flex items-start gap-4 rounded-2xl border border-rose-200/70 bg-rose-50/80 p-4 backdrop-blur dark:border-rose-800/70 dark:bg-rose-900/30">
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-rose-500/10 ring-1 ring-rose-500/30">
                    <AlertTriangle className="h-5 w-5 text-rose-600 dark:text-rose-400" />
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold text-slate-900 dark:text-white">Urgent outreach required</p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Schedule immediate call with retention specialist</p>
                  </div>
                </div>
                <div className="flex items-start gap-4 rounded-2xl border border-amber-200/70 bg-amber-50/80 p-4 backdrop-blur dark:border-amber-800/70 dark:bg-amber-900/30">
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-amber-500/10 ring-1 ring-amber-500/30">
                    <DollarSign className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold text-slate-900 dark:text-white">Offer premium discount or loyalty incentive</p>
                    <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Consider 10-15% discount for annual commitment</p>
                  </div>
                </div>
              </>
            )}
            {customer.churnProbability >= 30 && customer.churnProbability < 60 && (
              <div className="flex items-start gap-4 rounded-2xl border border-amber-200/70 bg-amber-50/80 p-4 backdrop-blur dark:border-amber-800/70 dark:bg-amber-900/30">
                <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-amber-500/10 ring-1 ring-amber-500/30">
                  <Phone className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-slate-900 dark:text-white">Proactive engagement recommended</p>
                  <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Reach out to check satisfaction and address concerns</p>
                </div>
              </div>
            )}
            <div className="flex items-start gap-4 rounded-2xl border border-blue-200/70 bg-blue-50/80 p-4 backdrop-blur dark:border-blue-800/70 dark:bg-blue-900/30">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-blue-500/10 ring-1 ring-blue-500/30">
                <CheckCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="flex-1">
                <p className="font-semibold text-slate-900 dark:text-white">Monitor customer sentiment</p>
                <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">Track engagement metrics and satisfaction scores monthly</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
