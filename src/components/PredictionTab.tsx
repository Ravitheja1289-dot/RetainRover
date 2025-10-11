import { useState } from 'react';
import { Play, RefreshCw, AlertCircle, CheckCircle, TrendingUp, Brain, Target } from 'lucide-react';
import { api, PredictionRequest, PredictionResponse } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';

export const PredictionTab = () => {
  const [formData, setFormData] = useState({
    age: '',
    marital_status: 'MARRIED',
    home_market_value: 'HIGH',
    annual_income: ''
  });
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const maritalStatusOptions = [
    { value: 'MARRIED', label: 'Married' },
    { value: 'SINGLE', label: 'Single' },
    { value: 'DIVORCED', label: 'Divorced' },
    { value: 'WIDOWED', label: 'Widowed' }
  ];

  const homeValueOptions = [
    { value: 'HIGH', label: 'High Value' },
    { value: 'MEDIUM', label: 'Medium Value' },
    { value: 'LOW', label: 'Low Value' },
    { value: 'VERY_LOW', label: 'Very Low Value' }
  ];

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    setError(null);
  };

  const validateForm = () => {
    if (!formData.age || parseInt(formData.age) < 18 || parseInt(formData.age) > 100) {
      setError('Age must be between 18 and 100');
      return false;
    }
    if (formData.annual_income && parseFloat(formData.annual_income) < 0) {
      setError('Annual income cannot be negative');
      return false;
    }
    return true;
  };

  const handlePredict = async () => {
    if (!validateForm()) return;

    setLoading(true);
    setError(null);

    try {
      const request: PredictionRequest = {
        customers: [{
          age: parseInt(formData.age),
          marital_status: formData.marital_status,
          home_market_value: formData.home_market_value,
          annual_income: formData.annual_income ? parseFloat(formData.annual_income) : undefined
        }]
      };

      const result = await api.predictCreditRisk(request);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel: string) => {
    const colors: Record<string, { bg: string; text: string; border: string }> = {
      'Very Low Risk': { bg: 'bg-green-50 dark:bg-green-900/20', text: 'text-green-800 dark:text-green-200', border: 'border-green-200 dark:border-green-800' },
      'Low Risk': { bg: 'bg-green-50 dark:bg-green-900/20', text: 'text-green-800 dark:text-green-200', border: 'border-green-200 dark:border-green-800' },
      'Medium Risk': { bg: 'bg-yellow-50 dark:bg-yellow-900/20', text: 'text-yellow-800 dark:text-yellow-200', border: 'border-yellow-200 dark:border-yellow-800' },
      'High Risk': { bg: 'bg-orange-50 dark:bg-orange-900/20', text: 'text-orange-800 dark:text-orange-200', border: 'border-orange-200 dark:border-orange-800' },
      'Very High Risk': { bg: 'bg-red-50 dark:bg-red-900/20', text: 'text-red-800 dark:text-red-200', border: 'border-red-200 dark:border-red-800' }
    };
    return colors[riskLevel] || colors['Medium Risk'];
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600 dark:text-green-400';
    if (confidence >= 0.7) return 'text-blue-600 dark:text-blue-400';
    if (confidence >= 0.5) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-center space-x-4">
          <div className="bg-white/20 p-3 rounded-lg">
            <Brain className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-2xl font-bold mb-2">Real-time Credit Risk Prediction</h2>
            <p className="text-indigo-100">
              Get instant credit risk assessments using our advanced AI models
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Customer Information
          </h3>

          <div className="space-y-4">
            {/* Age */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Age *
              </label>
              <input
                type="number"
                value={formData.age}
                onChange={(e) => handleInputChange('age', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                placeholder="Enter age (18-100)"
                min="18"
                max="100"
              />
            </div>

            {/* Marital Status */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Marital Status *
              </label>
              <select
                value={formData.marital_status}
                onChange={(e) => handleInputChange('marital_status', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
              >
                {maritalStatusOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Home Market Value */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Home Market Value *
              </label>
              <select
                value={formData.home_market_value}
                onChange={(e) => handleInputChange('home_market_value', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
              >
                {homeValueOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Annual Income */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Annual Income (Optional)
              </label>
              <input
                type="number"
                value={formData.annual_income}
                onChange={(e) => handleInputChange('annual_income', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                placeholder="Enter annual income"
                min="0"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-center space-x-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
                <span className="text-sm text-red-800 dark:text-red-200">{error}</span>
              </div>
            )}

            {/* Predict Button */}
            <button
              onClick={handlePredict}
              disabled={loading || !formData.age}
              className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-3 px-4 rounded-lg font-semibold hover:from-indigo-700 hover:to-purple-700 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span>Predicting...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  <span>Get Prediction</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
            Prediction Results
          </h3>

          {prediction ? (
            <div className="space-y-6">
              {/* Main Prediction */}
              {prediction.predictions.map((pred, index) => (
                <div key={index}>
                  <div className={`rounded-lg border-2 p-6 ${getRiskColor(pred.prediction.risk_level).border} ${getRiskColor(pred.prediction.risk_level).bg}`}>
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className="bg-white dark:bg-gray-800 p-3 rounded-full">
                          <Target className="w-6 h-6 text-gray-600 dark:text-gray-400" />
                        </div>
                        <div>
                          <h4 className={`text-lg font-bold ${getRiskColor(pred.prediction.risk_level).text}`}>
                            {pred.prediction.risk_level}
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Credit Risk Assessment
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-3xl font-bold ${getConfidenceColor(pred.prediction.confidence)}`}>
                          {(pred.prediction.good_credit_probability * 100).toFixed(1)}%
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          Good Credit Probability
                        </p>
                      </div>
                    </div>

                    {/* Probability Breakdown */}
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Good Credit
                          </span>
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        </div>
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {(pred.prediction.good_credit_probability * 100).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${pred.prediction.good_credit_probability * 100}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Bad Credit
                          </span>
                          <AlertCircle className="w-4 h-4 text-red-500" />
                        </div>
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                          {(pred.prediction.bad_credit_probability * 100).toFixed(1)}%
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mt-2">
                          <div
                            className="bg-red-500 h-2 rounded-full"
                            style={{ width: `${pred.prediction.bad_credit_probability * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    {/* Confidence Score */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                          Model Confidence
                        </span>
                        <TrendingUp className="w-4 h-4 text-blue-500" />
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="flex-1">
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${pred.prediction.confidence * 100}%` }}
                            ></div>
                          </div>
                        </div>
                        <span className={`text-lg font-bold ${getConfidenceColor(pred.prediction.confidence)}`}>
                          {(pred.prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}

              {/* Model Information */}
              <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
                <h5 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                  Model Information
                </h5>
                <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <p>Model: {prediction.model_info.model_used}</p>
                  <p>Accuracy: {(prediction.model_info.model_metrics.accuracy * 100).toFixed(1)}%</p>
                  <p>ROC-AUC: {(prediction.model_info.model_metrics.roc_auc * 100).toFixed(1)}%</p>
                  <p>Features: {prediction.model_info.feature_count}</p>
                  <p>Generated: {new Date(prediction.timestamp).toLocaleString()}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 dark:text-gray-400">
                Enter customer information and click "Get Prediction" to see the AI-powered risk assessment.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
