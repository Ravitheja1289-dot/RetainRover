import customersData from '../data/customers.json';
import featureImportanceData from '../data/featureImportance.json';
import regionalTrendsData from '../data/regionalTrends.json';

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const USE_REAL_API = process.env.REACT_APP_USE_REAL_API === 'true';

export interface Customer {
  id: number;
  name: string;
  region: string;
  churnProbability: number;
  age: number;
  tenure: number;
  premiumAmount: number;
  claimsCount: number;
  lastInteraction: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  description: string;
}

export interface RegionalTrend {
  region: string;
  churnRate: number;
  totalCustomers: number;
  churnedCustomers: number;
  avgTenure: number;
}

export interface ModelMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  cv_score_mean: number;
  cv_score_std: number;
}

export interface PredictionRequest {
  customers: Array<{
    age: number;
    marital_status: string;
    home_market_value: string;
    annual_income?: number;
  }>;
}

export interface PredictionResponse {
  predictions: Array<{
    customer_id: number;
    customer_data: any;
    prediction: {
      good_credit_probability: number;
      bad_credit_probability: number;
      predicted_class: number;
      confidence: number;
      risk_level: string;
    };
  }>;
  model_info: any;
  timestamp: string;
}

// Mutable customers data
let currentCustomersData = customersData as Customer[];

export const api = {
  async getCustomers(): Promise<Customer[]> {
    await delay(500);
    return currentCustomersData;
  },

  async getCustomerById(id: number): Promise<Customer | undefined> {
    await delay(300);
    return currentCustomersData.find(c => c.id === id);
  },

  setCustomers(newCustomers: Customer[]): void {
    currentCustomersData = newCustomers;
  },

  async getFeatureImportance(): Promise<FeatureImportance[]> {
    await delay(400);
    return featureImportanceData as FeatureImportance[];
  },

  async getRegionalTrends(): Promise<RegionalTrend[]> {
    await delay(450);
    return regionalTrendsData as RegionalTrend[];
  },

  // Real API methods
  async getModelMetrics(): Promise<ModelMetrics[]> {
    if (!USE_REAL_API) {
      await delay(300);
      return [
        {
          model_name: 'RandomForest',
          accuracy: 0.942,
          precision: 0.938,
          recall: 0.945,
          f1_score: 0.941,
          roc_auc: 0.956,
          cv_score_mean: 0.954,
          cv_score_std: 0.012
        },
        {
          model_name: 'GradientBoosting',
          accuracy: 0.939,
          precision: 0.935,
          recall: 0.942,
          f1_score: 0.938,
          roc_auc: 0.951,
          cv_score_mean: 0.949,
          cv_score_std: 0.015
        }
      ];
    }

    try {
      const response = await fetch(`${API_BASE_URL}/metrics`);
      if (!response.ok) throw new Error('Failed to fetch model metrics');
      const data = await response.json();
      return data.models;
    } catch (error) {
      console.error('Error fetching model metrics:', error);
      throw error;
    }
  },

  async predictCreditRisk(request: PredictionRequest): Promise<PredictionResponse> {
    if (!USE_REAL_API) {
      await delay(500);
      return {
        predictions: request.customers.map((customer, index) => ({
          customer_id: index + 1,
          customer_data: customer,
          prediction: {
            good_credit_probability: Math.random() * 0.4 + 0.6, // 0.6-1.0
            bad_credit_probability: Math.random() * 0.4, // 0.0-0.4
            predicted_class: 1,
            confidence: Math.random() * 0.3 + 0.7, // 0.7-1.0
            risk_level: 'Low Risk'
          }
        })),
        model_info: {
          model_used: 'RandomForest',
          model_metrics: {
            accuracy: 0.942,
            precision: 0.938,
            recall: 0.945,
            f1: 0.941,
            roc_auc: 0.956
          },
          feature_count: 8
        },
        timestamp: new Date().toISOString()
      };
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) throw new Error('Failed to get prediction');
      return await response.json();
    } catch (error) {
      console.error('Error getting prediction:', error);
      throw error;
    }
  },

  async getApiHealth(): Promise<any> {
    if (!USE_REAL_API) {
      await delay(200);
      return {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        models_loaded: 4,
        model_status: 'loaded',
        preprocessor_loaded: true
      };
    }

    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) throw new Error('API health check failed');
      return await response.json();
    } catch (error) {
      console.error('Error checking API health:', error);
      throw error;
    }
  },

  async getFeatureImportance(modelName: string): Promise<any> {
    if (!USE_REAL_API) {
      await delay(300);
      return {
        model_name: modelName,
        feature_importance: featureImportanceData.map(f => ({
          feature: f.feature,
          importance: f.importance
        })),
        total_features: featureImportanceData.length
      };
    }

    try {
      const response = await fetch(`${API_BASE_URL}/feature_importance/${modelName}`);
      if (!response.ok) throw new Error('Failed to fetch feature importance');
      return await response.json();
    } catch (error) {
      console.error('Error fetching feature importance:', error);
      throw error;
    }
  }
};
