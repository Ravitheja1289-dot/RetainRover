import { Lightbulb, TrendingUp, AlertTriangle, Target } from 'lucide-react';

export const InsightsTab = () => {
  const insights = [
    {
      icon: AlertTriangle,
      title: 'High Risk Regions',
      description: 'Northeast and Midwest regions show significantly higher churn rates (62.7% and 58%) compared to South (31%).',
      recommendation: 'Implement targeted retention campaigns in high-risk regions with personalized offers.',
      impact: 'High',
      color: 'red'
    },
    {
      icon: TrendingUp,
      title: 'Claims Count Driver',
      description: 'Customers with 5+ claims in the last year have an 85% churn probability. This is the strongest predictor.',
      recommendation: 'Proactive outreach to customers after their 3rd claim to address concerns and improve satisfaction.',
      impact: 'Critical',
      color: 'orange'
    },
    {
      icon: Target,
      title: 'Tenure Sweet Spot',
      description: 'Customers with 3-7 years tenure show highest churn risk. Long-term customers (15+ years) are very loyal.',
      recommendation: 'Create loyalty milestones and rewards program targeting 3-7 year tenure segment.',
      impact: 'Medium',
      color: 'blue'
    },
    {
      icon: Lightbulb,
      title: 'Engagement Matters',
      description: 'Customers with no interactions in 60+ days are 3x more likely to churn.',
      recommendation: 'Implement automated check-ins and value-add communications for inactive customers.',
      impact: 'High',
      color: 'green'
    }
  ];

  const getColorClasses = (color: string) => {
    const colors: Record<string, { bg: string; text: string; icon: string }> = {
      red: {
        bg: 'bg-red-50 dark:bg-red-900/20',
        text: 'text-red-900 dark:text-red-100',
        icon: 'text-red-600 dark:text-red-400'
      },
      orange: {
        bg: 'bg-orange-50 dark:bg-orange-900/20',
        text: 'text-orange-900 dark:text-orange-100',
        icon: 'text-orange-600 dark:text-orange-400'
      },
      blue: {
        bg: 'bg-blue-50 dark:bg-blue-900/20',
        text: 'text-blue-900 dark:text-blue-100',
        icon: 'text-blue-600 dark:text-blue-400'
      },
      green: {
        bg: 'bg-green-50 dark:bg-green-900/20',
        text: 'text-green-900 dark:text-green-100',
        icon: 'text-green-600 dark:text-green-400'
      }
    };
    return colors[color] || colors.blue;
  };

  const getImpactBadge = (impact: string) => {
    const styles: Record<string, string> = {
      Critical: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
      High: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
      Medium: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
    };
    return styles[impact] || styles.Medium;
  };

  return (
    <div className="p-6 space-y-6">
      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg p-6 text-white">
        <h2 className="text-2xl font-bold mb-2">AI-Generated Insights</h2>
        <p className="text-blue-100">
          Key findings and actionable recommendations based on churn prediction analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {insights.map((insight, index) => {
          const Icon = insight.icon;
          const colors = getColorClasses(insight.color);
          return (
            <div
              key={index}
              className={`rounded-lg border-2 border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-all ${colors.bg}`}
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 rounded-lg bg-white dark:bg-gray-800 ${colors.icon}`}>
                  <Icon className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className={`text-lg font-semibold ${colors.text}`}>
                      {insight.title}
                    </h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getImpactBadge(insight.impact)}`}>
                      {insight.impact}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
                    {insight.description}
                  </p>
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                    <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                      Recommendation
                    </p>
                    <p className="text-sm text-gray-900 dark:text-white">
                      {insight.recommendation}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Overall Churn Prevention Strategy
        </h3>
        <div className="space-y-4">
          <div className="flex items-start space-x-3">
            <div className="bg-blue-100 dark:bg-blue-900/30 rounded-full p-1 mt-0.5">
              <div className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full"></div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                Prioritize high-risk customers (70%+ churn probability)
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Focus retention efforts on customers most likely to leave for maximum ROI
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="bg-blue-100 dark:bg-blue-900/30 rounded-full p-1 mt-0.5">
              <div className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full"></div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                Address root causes (claims experience, engagement)
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Improve claims process and maintain regular touchpoints with customers
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="bg-blue-100 dark:bg-blue-900/30 rounded-full p-1 mt-0.5">
              <div className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full"></div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                Regional customization of retention strategies
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Tailor approaches based on regional differences in churn patterns
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
