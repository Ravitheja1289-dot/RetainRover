import customersData from '../data/customers.json';
import featureImportanceData from '../data/featureImportance.json';
import regionalTrendsData from '../data/regionalTrends.json';

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

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
  }
};
