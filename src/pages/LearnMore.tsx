import { Link } from 'react-router-dom';
import { Shield, Brain, BarChart3, Map, Lightbulb, Users, TrendingUp, CheckCircle, ArrowRight, HelpCircle, ChevronDown } from 'lucide-react';
import { useState } from 'react';

export const LearnMore = () => {
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  const toggleFaq = (index: number) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  const faqs = [
    {
      question: "How accurate are the churn predictions?",
      answer: "Our AI models achieve 94% accuracy in predicting customer churn, based on extensive training with historical data from thousands of insurance customers."
    },
    {
      question: "What data do you need to get started?",
      answer: "We analyze customer demographics, policy details, claims history, interaction logs, and payment patterns. No sensitive personal information is required."
    },
    {
      question: "How often should I check the dashboard?",
      answer: "We recommend weekly reviews of high-risk customers and monthly analysis of trends. The system updates in real-time as new data becomes available."
    },
    {
      question: "Can I export the insights and reports?",
      answer: "Yes, all charts, predictions, and recommendations can be exported as PDF reports or CSV data for integration with your existing systems."
    },
    {
      question: "Is my customer data secure?",
      answer: "Absolutely. We use enterprise-grade encryption, comply with GDPR and HIPAA standards, and never share or sell customer data to third parties."
    }
  ];

  const benefits = [
    {
      icon: TrendingUp,
      title: "Reduce Churn by 30%",
      description: "Targeted interventions based on AI predictions can significantly reduce customer loss rates."
    },
    {
      icon: Users,
      title: "Improve Customer Satisfaction",
      description: "Proactive engagement prevents issues before they lead to dissatisfaction and churn."
    },
    {
      icon: BarChart3,
      title: "Data-Driven Decisions",
      description: "Make retention strategies based on proven analytics rather than guesswork."
    },
    {
      icon: Lightbulb,
      title: "Cost-Effective Retention",
      description: "Focus resources on customers most likely to leave, maximizing ROI on retention efforts."
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:from-gray-950 dark:via-gray-900 dark:to-gray-950">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-200/70 bg-white/80 backdrop-blur-xl dark:border-slate-800/70 dark:bg-slate-900/80">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <Link to="/" className="flex items-center gap-3 transition-opacity hover:opacity-80">
              <div className="rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 p-2 shadow-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <h1 className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-2xl font-bold text-transparent">
                InsuraSense
              </h1>
            </Link>
            <div className="flex items-center gap-4">
              <Link
                to="/learn-more"
                className="font-medium text-slate-600 transition-colors hover:text-blue-600 dark:text-slate-400 dark:hover:text-blue-400"
              >
                Learn More
              </Link>
              <Link
                to="/about-us"
                className="font-medium text-slate-600 transition-colors hover:text-blue-600 dark:text-slate-400 dark:hover:text-blue-400"
              >
                About Us
              </Link>
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 px-4 py-2 font-semibold text-white shadow-lg shadow-blue-600/30 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl"
              >
                View Dashboard
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-5xl font-bold mb-6">
            Understanding InsuraSense
          </h1>
          <p className="text-xl text-blue-100 max-w-3xl mx-auto">
            Discover how our AI-powered platform helps insurance companies predict and prevent customer churn
            through advanced analytics and actionable insights.
          </p>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              How InsuraSense Works
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Our platform combines machine learning with insurance expertise to deliver accurate churn predictions.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 dark:bg-blue-900/30 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                Data Analysis
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Our AI analyzes customer behavior patterns, claims history, engagement metrics, and policy details
                to identify risk factors.
              </p>
            </div>

            <div className="text-center">
              <div className="bg-green-100 dark:bg-green-900/30 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <BarChart3 className="w-8 h-8 text-green-600 dark:text-green-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                Risk Scoring
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Advanced algorithms calculate churn probability scores and explain which factors contribute
                most to each prediction.
              </p>
            </div>

            <div className="text-center">
              <div className="bg-purple-100 dark:bg-purple-900/30 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Lightbulb className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                Actionable Insights
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Receive personalized recommendations for retaining high-risk customers, from outreach strategies
                to policy adjustments.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 bg-gray-100 dark:bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Key Features
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Comprehensive tools designed specifically for insurance churn prevention.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
              <div className="flex items-start space-x-4">
                <Shield className="w-8 h-8 text-blue-600 dark:text-blue-400 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    Real-time Risk Monitoring
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Continuous monitoring of customer behavior with instant alerts for high-risk situations.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
              <div className="flex items-start space-x-4">
                <Map className="w-8 h-8 text-green-600 dark:text-green-400 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    Regional Analysis
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Understand geographic patterns and customize retention strategies by region.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
              <div className="flex items-start space-x-4">
                <BarChart3 className="w-8 h-8 text-purple-600 dark:text-purple-400 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    SHAP Explanations
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Transparent AI with feature importance explanations for every prediction.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
              <div className="flex items-start space-x-4">
                <Users className="w-8 h-8 text-orange-600 dark:text-orange-400 mt-1" />
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                    Customer Segmentation
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Automatically segment customers by risk level for targeted interventions.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Proven Benefits
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Real results from insurance companies using InsuraSense.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {benefits.map((benefit, index) => {
              const Icon = benefit.icon;
              return (
                <div key={index} className="text-center">
                  <div className="bg-gradient-to-br from-blue-500 to-cyan-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                    {benefit.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    {benefit.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="py-20 bg-gray-100 dark:bg-gray-900">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Frequently Asked Questions
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300">
              Everything you need to know about InsuraSense.
            </p>
          </div>

          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <div key={index} className="bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                <button
                  onClick={() => toggleFaq(index)}
                  className="w-full px-6 py-4 text-left flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <HelpCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    <span className="font-medium text-gray-900 dark:text-white">
                      {faq.question}
                    </span>
                  </div>
                  <ChevronDown
                    className={`w-5 h-5 text-gray-500 transition-transform ${
                      openFaq === index ? 'rotate-180' : ''
                    }`}
                  />
                </button>
                {openFaq === index && (
                  <div className="px-6 pb-4">
                    <p className="text-gray-600 dark:text-gray-300">
                      {faq.answer}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-cyan-600">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl font-bold text-white mb-4">
            Ready to Transform Your Retention Strategy?
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            Join leading insurance companies using AI to prevent customer churn and improve satisfaction.
          </p>
          <Link
            to="/dashboard"
            className="bg-white text-blue-600 px-8 py-4 rounded-lg font-semibold text-lg hover:bg-gray-50 transition-all duration-200 inline-flex items-center space-x-2"
          >
            <span>Start Your Free Trial</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-4 md:mb-0">
              <div className="bg-gradient-to-br from-blue-500 to-cyan-600 p-2 rounded-lg">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold">InsuraSense</span>
            </div>
            <div className="text-gray-400 text-sm">
              © 2024 InsuraSense. AI-powered churn prediction for insurance.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};
