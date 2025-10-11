import { Link } from 'react-router-dom';
import { Shield, BarChart3, Map, Lightbulb, ArrowRight, CheckCircle } from 'lucide-react';

export const LandingPage = () => {
  const features = [
    {
      icon: Shield,
      title: 'AI-Powered Predictions',
      description: 'Advanced machine learning models analyze customer data to predict churn probability with high accuracy.'
    },
    {
      icon: BarChart3,
      title: 'Feature Importance Analysis',
      description: 'Understand which factors most influence customer retention decisions with SHAP value explanations.'
    },
    {
      icon: Map,
      title: 'Regional Insights',
      description: 'Identify geographic patterns and regional differences in churn rates for targeted interventions.'
    },
    {
      icon: Lightbulb,
      title: 'Actionable Recommendations',
      description: 'Get specific, data-driven suggestions for retaining high-risk customers and improving satisfaction.'
    }
  ];

  const stats = [
    { label: 'Prediction Accuracy', value: '94%' },
    { label: 'Customers Analyzed', value: '12K+' },
    { label: 'Regions Covered', value: '4' },
    { label: 'Features Tracked', value: '8' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b border-slate-200/70 bg-white/80 backdrop-blur-xl dark:border-slate-800/70 dark:bg-slate-900/80">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 p-2 shadow-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <h1 className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-2xl font-bold text-transparent">
                InsuraSense
              </h1>
            </div>
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 px-6 py-2 font-semibold text-white shadow-lg shadow-blue-600/30 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl"
            >
              <span>View Dashboard</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="px-4 pb-16 pt-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl text-center">
          <h1 className="mb-6 text-5xl font-bold text-slate-900 dark:text-white md:text-6xl">
            Predict & Prevent
            <span className="block bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
              Customer Churn
            </span>
          </h1>
          <p className="mx-auto mb-8 max-w-3xl text-xl text-slate-600 dark:text-slate-300">
            Leverage AI-powered insights to identify at-risk customers, understand churn drivers,
            and implement targeted retention strategies that save costs and improve satisfaction.
          </p>
          <div className="flex flex-col justify-center gap-4 sm:flex-row">
            <Link
              to="/dashboard"
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 px-8 py-4 text-lg font-semibold text-white shadow-xl shadow-blue-600/30 transition-all duration-200 hover:-translate-y-1 hover:shadow-2xl"
            >
              <span>Explore Dashboard</span>
              <ArrowRight className="h-5 w-5" />
            </Link>
            <Link
              to="/learn-more"
              className="rounded-xl border-2 border-slate-300 px-8 py-4 text-lg font-semibold text-slate-700 transition-all duration-200 hover:-translate-y-1 hover:border-blue-500 hover:text-blue-600 hover:shadow-lg dark:border-slate-600 dark:text-slate-300 dark:hover:border-blue-400 dark:hover:text-blue-400"
            >
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-white/50 py-16 backdrop-blur dark:bg-slate-900/50">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="mb-2 text-4xl font-bold text-slate-900 dark:text-white">
                  {stat.value}
                </div>
                <div className="text-slate-600 dark:text-slate-400">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="mb-16 text-center">
            <h2 className="mb-4 text-4xl font-bold text-slate-900 dark:text-white">
              Powerful Features for Smarter Retention
            </h2>
            <p className="mx-auto max-w-3xl text-xl text-slate-600 dark:text-slate-300">
              Our comprehensive platform provides everything you need to understand and prevent customer churn.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="rounded-3xl border border-slate-200/70 bg-white/80 p-8 shadow-lg backdrop-blur transition-all duration-200 hover:-translate-y-1 hover:shadow-2xl dark:border-slate-800/70 dark:bg-slate-900/80"
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-600 p-3 shadow-lg">
                      <Icon className="h-8 w-8 text-white" />
                    </div>
                    <div>
                      <h3 className="mb-3 text-xl font-semibold text-slate-900 dark:text-white">
                        {feature.title}
                      </h3>
                      <p className="leading-relaxed text-slate-600 dark:text-slate-300">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-blue-600 to-cyan-600 py-20">
        <div className="mx-auto max-w-4xl px-4 text-center sm:px-6 lg:px-8">
          <h2 className="mb-4 text-4xl font-bold text-white">
            Ready to Reduce Churn by 30%?
          </h2>
          <p className="mb-8 text-xl text-blue-100">
            Start analyzing your customer data today and implement data-driven retention strategies.
          </p>
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-2 rounded-xl bg-white px-8 py-4 text-lg font-semibold text-blue-600 shadow-xl transition-all duration-200 hover:-translate-y-1 hover:bg-slate-50 hover:shadow-2xl"
          >
            <span>Get Started Now</span>
            <ArrowRight className="h-5 w-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 py-12 text-white">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center justify-between md:flex-row">
            <div className="mb-4 flex items-center gap-3 md:mb-0">
              <div className="rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 p-2 shadow-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold">InsuraSense</span>
            </div>
            <div className="text-sm text-slate-400">
              © 2024 InsuraSense. AI-powered churn prediction for insurance.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};
