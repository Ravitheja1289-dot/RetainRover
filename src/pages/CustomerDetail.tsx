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
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        <button
          onClick={() => navigate('/')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back to Dashboard</span>
        </button>

        <div className={`rounded-lg border-2 ${risk.borderColor} ${risk.bg} p-6`}>
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-4">
              <div className="bg-white dark:bg-gray-800 p-4 rounded-full">
                <User className="w-8 h-8 text-gray-600 dark:text-gray-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                  {customer.name}
                </h1>
                <p className="text-gray-600 dark:text-gray-400">Customer ID: {customer.id}</p>
              </div>
            </div>
            <div className="text-right">
              <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium ${risk.bg} ${risk.color} border-2 ${risk.borderColor}`}>
                <RiskIcon className="w-5 h-5" />
                <span>{risk.label}</span>
              </div>
              <div className="mt-2">
                <span className="text-4xl font-bold text-gray-900 dark:text-white">
                  {customer.churnProbability}%
                </span>
                <p className="text-sm text-gray-600 dark:text-gray-400">Churn Probability</p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
            <div className="flex items-center space-x-2">
              <MapPin className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Region</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">{customer.region}</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Calendar className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Age</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">{customer.age} years</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Calendar className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Tenure</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">{customer.tenure} years</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <DollarSign className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Premium</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">${customer.premiumAmount}/mo</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Claims</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">{customer.claimsCount}</p>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <Phone className="w-4 h-4 text-gray-500 dark:text-gray-400" />
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Last interaction: {new Date(customer.lastInteraction).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Churn Prediction Explanation
          </h2>
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            {getExplanation()}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
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

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {featureContributions.map((contribution) => (
              <div
                key={contribution.feature}
                className="flex items-center justify-between p-4 rounded-lg bg-gray-50 dark:bg-gray-900/50"
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-3 h-3 rounded-full ${contribution.value > 0 ? 'bg-red-500' : 'bg-green-500'}`}
                  ></div>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {contribution.feature}
                  </span>
                </div>
                <div className="text-right">
                  <span className={`text-sm font-bold ${contribution.value > 0 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                    {contribution.value > 0 ? '+' : ''}{(contribution.value * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
            <p className="text-sm text-blue-900 dark:text-blue-100">
              <span className="font-medium">How to read this chart:</span> Positive values (red bars) increase churn probability,
              while negative values (green bars) decrease it. Longer bars indicate stronger influence on the prediction.
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Recommended Actions
          </h2>
          <div className="space-y-3">
            {customer.churnProbability >= 60 && (
              <>
                <div className="flex items-start space-x-3 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-white">Urgent outreach required</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Schedule immediate call with retention specialist</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3 p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <DollarSign className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-white">Offer premium discount or loyalty incentive</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Consider 10-15% discount for annual commitment</p>
                  </div>
                </div>
              </>
            )}
            {customer.churnProbability >= 30 && customer.churnProbability < 60 && (
              <div className="flex items-start space-x-3 p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                <Phone className="w-5 h-5 text-orange-600 dark:text-orange-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">Proactive engagement recommended</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Reach out to check satisfaction and address concerns</p>
                </div>
              </div>
            )}
            <div className="flex items-start space-x-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <CheckCircle className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium text-gray-900 dark:text-white">Monitor customer sentiment</p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Track engagement metrics and satisfaction scores monthly</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
