import { Lightbulb, TrendingUp, AlertTriangle, Target, Sparkles, Brain } from 'lucide-react';

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
    const colors: Record<string, { bg: string; text: string; icon: string; border: string }> = {
      red: {
        bg: 'bg-rose-50/80 dark:bg-rose-900/30',
        text: 'text-rose-900 dark:text-rose-100',
        icon: 'text-rose-600 dark:text-rose-400',
        border: 'border-rose-200/70 dark:border-rose-800/70'
      },
      orange: {
        bg: 'bg-amber-50/80 dark:bg-amber-900/30',
        text: 'text-amber-900 dark:text-amber-100',
        icon: 'text-amber-600 dark:text-amber-400',
        border: 'border-amber-200/70 dark:border-amber-800/70'
      },
      blue: {
        bg: 'bg-blue-50/80 dark:bg-blue-900/30',
        text: 'text-blue-900 dark:text-blue-100',
        icon: 'text-blue-600 dark:text-blue-400',
        border: 'border-blue-200/70 dark:border-blue-800/70'
      },
      green: {
        bg: 'bg-emerald-50/80 dark:bg-emerald-900/30',
        text: 'text-emerald-900 dark:text-emerald-100',
        icon: 'text-emerald-600 dark:text-emerald-400',
        border: 'border-emerald-200/70 dark:border-emerald-800/70'
      }
    };
    return colors[color] || colors.blue;
  };

  const getImpactBadge = (impact: string) => {
    const styles: Record<string, string> = {
      Critical: 'bg-rose-500/10 text-rose-700 dark:bg-rose-500/20 dark:text-rose-300 ring-1 ring-rose-500/30',
      High: 'bg-amber-500/10 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300 ring-1 ring-amber-500/30',
      Medium: 'bg-blue-500/10 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300 ring-1 ring-blue-500/30'
    };
    return styles[impact] || styles.Medium;
  };

  return (
    <div className="space-y-8 p-6">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl border border-blue-500/20 bg-gradient-to-br from-blue-600/10 via-cyan-500/10 to-transparent p-8 shadow-lg transition-shadow hover:shadow-2xl dark:border-blue-400/20 dark:from-blue-500/15 dark:via-sky-500/10">
        <div className="absolute -top-24 -right-12 h-64 w-64 rounded-full bg-blue-500/30 blur-3xl animate-pulse-slow" aria-hidden="true"></div>
        <div className="absolute -bottom-20 -left-10 h-64 w-64 rounded-full bg-cyan-500/20 blur-3xl animate-pulse-slow" aria-hidden="true"></div>
        <div className="relative space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full bg-white/70 px-4 py-2 text-sm font-medium text-blue-700 shadow-sm ring-1 ring-blue-500/20 backdrop-blur dark:bg-slate-900/70 dark:text-blue-200">
            <Brain className="h-4 w-4" />
            AI-powered strategic insights
          </div>
          <div className="space-y-3">
            <h2 className="text-3xl font-semibold text-slate-900 dark:text-white sm:text-4xl">
              Transform data patterns into retention playbooks
            </h2>
            <p className="text-base text-slate-600 dark:text-slate-300 sm:max-w-3xl">
              Key findings and actionable recommendations based on churn prediction analysis. Each insight is backed by AI-driven pattern recognition and designed to guide your next best action.
            </p>
          </div>
          <div className="flex flex-wrap gap-3 text-sm text-slate-600 dark:text-slate-300">
            <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
              <Sparkles className="h-4 w-4 text-blue-500" />
              {insights.length} strategic insights
            </span>
            <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
              <Target className="h-4 w-4 text-blue-500" />
              Actionable recommendations
            </span>
          </div>
        </div>
      </div>

      {/* Insights Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {insights.map((insight, index) => {
          const Icon = insight.icon;
          const colors = getColorClasses(insight.color);
          return (
            <div
              key={index}
              className={`group relative overflow-hidden rounded-3xl border ${colors.border} ${colors.bg} p-6 shadow-lg backdrop-blur transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl`}
            >
              <div className="relative flex items-start gap-4">
                <div className={`flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-2xl bg-white/80 shadow-lg ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/80 dark:ring-slate-800/70 ${colors.icon}`}>
                  <Icon className="h-7 w-7" />
                </div>
                <div className="flex-1 space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <h3 className={`text-lg font-semibold ${colors.text}`}>
                      {insight.title}
                    </h3>
                    <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${getImpactBadge(insight.impact)}`}>
                      {insight.impact}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-300">
                    {insight.description}
                  </p>
                  <div className="rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
                    <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      Recommended Action
                    </p>
                    <p className="text-sm leading-relaxed text-slate-900 dark:text-white">
                      {insight.recommendation}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Strategy Section */}
      <div className="rounded-3xl border border-slate-200/70 bg-white/80 p-8 shadow-lg backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/80">
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-lg">
            <Target className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
              Overall Churn Prevention Strategy
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Integrated approach for maximum retention impact
            </p>
          </div>
        </div>
        <div className="space-y-4">
          <div className="flex items-start gap-4 rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
            <div className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl bg-blue-500/10 ring-1 ring-blue-500/30">
              <div className="h-2 w-2 rounded-full bg-blue-600 dark:bg-blue-400"></div>
            </div>
            <div className="flex-1">
              <p className="font-semibold text-slate-900 dark:text-white">
                Prioritize high-risk customers (70%+ churn probability)
              </p>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
                Focus retention efforts on customers most likely to leave for maximum ROI
              </p>
            </div>
          </div>
          <div className="flex items-start gap-4 rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
            <div className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl bg-blue-500/10 ring-1 ring-blue-500/30">
              <div className="h-2 w-2 rounded-full bg-blue-600 dark:bg-blue-400"></div>
            </div>
            <div className="flex-1">
              <p className="font-semibold text-slate-900 dark:text-white">
                Address root causes (claims experience, engagement)
              </p>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
                Improve claims process and maintain regular touchpoints with customers
              </p>
            </div>
          </div>
          <div className="flex items-start gap-4 rounded-2xl border border-slate-200/70 bg-white/70 p-4 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
            <div className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl bg-blue-500/10 ring-1 ring-blue-500/30">
              <div className="h-2 w-2 rounded-full bg-blue-600 dark:bg-blue-400"></div>
            </div>
            <div className="flex-1">
              <p className="font-semibold text-slate-900 dark:text-white">
                Regional customization of retention strategies
              </p>
              <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
                Tailor approaches based on regional differences in churn patterns
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
