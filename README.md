# InsuraSense - AI-Powered Churn Prediction Dashboard

A modern React.js dashboard for visualizing customer churn predictions, SHAP-based feature importance, regional trends, and individual customer insights.

## Features

### Dashboard Tabs

1. **Customers Tab**
   - Comprehensive customer table with color-coded risk indicators
   - Churn probability visualization with progress bars
   - Quick access to detailed customer views
   - Risk levels: Low (<30%), Medium (30-60%), High (>60%)

2. **Feature Importance Tab**
   - SHAP global feature importance visualization
   - Interactive bar charts showing feature contributions
   - Detailed descriptions for each feature
   - Color-coded importance levels

3. **Regional Trends Tab**
   - Regional churn rate comparison
   - Customer distribution by region
   - Average tenure analysis
   - Visual cards with key metrics

4. **Insights Tab**
   - AI-generated actionable insights
   - Strategic recommendations
   - Impact-level categorization
   - Overall churn prevention strategy

5. **Customer Detail Page**
   - Individual customer profile
   - Churn probability breakdown
   - Feature contribution analysis
   - AI-generated explanations
   - Recommended retention actions

### Additional Features

- **Dark Mode**: Toggle between light and dark themes with persistent preference
- **Responsive Design**: Fully optimized for mobile, tablet, and desktop
- **Loading States**: Smooth loading indicators for all data fetches
- **Professional UI**: Clean, modern SaaS-style interface with gradients and animations

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **React Router** for navigation
- **Recharts** for data visualizations
- **Tailwind CSS** for styling
- **Lucide React** for icons

## Project Structure

```
src/
├── components/           # Reusable UI components
│   ├── CustomersTab.tsx
│   ├── FeatureImportanceTab.tsx
│   ├── RegionalTrendsTab.tsx
│   ├── InsightsTab.tsx
│   ├── Header.tsx
│   ├── Tabs.tsx
│   └── LoadingSpinner.tsx
├── contexts/            # React context providers
│   └── ThemeContext.tsx
├── data/                # Mock API data
│   ├── customers.json
│   ├── featureImportance.json
│   └── regionalTrends.json
├── pages/               # Page components
│   ├── Dashboard.tsx
│   └── CustomerDetail.tsx
├── services/            # API service layer
│   └── api.ts
├── App.tsx
└── main.tsx
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Clone the repository or extract the project files

2. Install dependencies:
```bash
npm install
```

### Running the Application

Development mode with hot reload:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Mock API Endpoints

The application uses a simulated API layer that mimics the following endpoints:

- `GET /api/customers` - List of all customers with churn predictions
- `GET /api/customer/:id` - Individual customer details
- `GET /api/feature_importance` - SHAP feature importance values
- `GET /api/region_trends` - Regional churn statistics

### Integrating with Real API

To connect to your FastAPI backend:

1. Update `src/services/api.ts` to replace mock data with actual fetch calls:

```typescript
export const api = {
  async getCustomers(): Promise<Customer[]> {
    const response = await fetch('YOUR_API_URL/api/customers');
    return response.json();
  },
  // ... other methods
};
```

2. Add environment variables in `.env`:

```
VITE_API_BASE_URL=http://your-backend-url
```

3. Update the API service to use the environment variable:

```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;
```

## Data Structure

### Customer Object
```typescript
interface Customer {
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
```

### Feature Importance Object
```typescript
interface FeatureImportance {
  feature: string;
  importance: number;
  description: string;
}
```

### Regional Trend Object
```typescript
interface RegionalTrend {
  region: string;
  churnRate: number;
  totalCustomers: number;
  churnedCustomers: number;
  avgTenure: number;
}
```

## Customization

### Colors & Theme

Modify `tailwind.config.js` to customize the color palette and design tokens.

### Adding New Features

1. Create feature-specific data in `src/data/`
2. Add API method in `src/services/api.ts`
3. Create component in `src/components/`
4. Add to Dashboard tabs or routing as needed

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance

- Code splitting implemented via React Router
- Lazy loading for heavy components
- Optimized bundle size with tree shaking
- Efficient re-renders with proper React patterns

## Future Enhancements

- Export data to CSV/PDF
- Advanced filtering and search
- Real-time data updates via WebSocket
- Customer comparison tool
- Predictive timeline visualization
- Email notification system

## License

This project is for demonstration purposes.

## Support

For questions or issues, please contact your development team.
