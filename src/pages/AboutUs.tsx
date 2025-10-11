import { Link } from 'react-router-dom';
import { Shield, Users, Mail, Github, Linkedin } from 'lucide-react';

export const AboutUs = () => {
  const teamMembers = [
    {
      name: 'S.Rithvik Reddy',
      role:  'UI/UX Designer',
      bio: 'Passionate about AI and machine learning solutions for business challenges.',
      avatar: '🎨'
    },
    {
      name: 'S.Ravi Teja Reddy',
      role: 'Lead Developer',
      bio: 'Creating intuitive and beautiful user experiences for complex applications.',
      avatar: '👨‍💻'
    },
    {
      name: 'S.Durga Prasad',
      role: 'Backend Engineer',
      bio: 'Building robust and scalable systems to power AI-driven platforms.',
      avatar: '⚙️'
    },
    {
      name: 'T.Sai Anjan Kumar',
      role: 'Data Scientist',
      bio: 'Transforming data into actionable insights for predictive analytics.',
      avatar: '📊'
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
            Meet Our Team
          </h1>
          <p className="text-xl text-blue-100 max-w-3xl mx-auto">
            The passionate minds behind InsuraSense, dedicated to revolutionizing insurance
            with AI-powered churn prediction and customer insights.
          </p>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Our Team
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              A diverse group of experts combining AI, design, engineering, and data science
              to build the future of insurance analytics.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
            {teamMembers.map((member, index) => (
              <div key={index} className="rounded-3xl border border-slate-200/70 bg-white/80 p-6 text-center shadow-lg backdrop-blur transition-all duration-200 hover:-translate-y-1 hover:shadow-2xl dark:border-slate-800/70 dark:bg-slate-900/80">
                <div className="mb-4 text-6xl">{member.avatar}</div>
                <h3 className="mb-2 text-xl font-semibold text-slate-900 dark:text-white">
                  {member.name}
                </h3>
                <p className="mb-3 font-semibold text-blue-600 dark:text-blue-400">
                  {member.role}
                </p>
                <p className="text-sm text-slate-600 dark:text-slate-300">
                  {member.bio}
                </p>
                <div className="mt-4 flex justify-center gap-3">
                  <a
                    href={`mailto:${member.name.toLowerCase().replace(/\s+/g, '.')}@insurasense.com`}
                    className="rounded-xl p-2 text-slate-400 transition-all hover:bg-blue-50 hover:text-blue-600 dark:hover:bg-blue-900/30 dark:hover:text-blue-400"
                    title="Email"
                  >
                    <Mail className="h-5 w-5" />
                  </a>
                  <a
                    href={`https://github.com/${member.name.toLowerCase().replace(/\s+/g, '')}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="rounded-xl p-2 text-slate-400 transition-all hover:bg-blue-50 hover:text-blue-600 dark:hover:bg-blue-900/30 dark:hover:text-blue-400"
                    title="GitHub"
                  >
                    <Github className="h-5 w-5" />
                  </a>
                  <a
                    href={`https://linkedin.com/in/${member.name.toLowerCase().replace(/\s+/g, '-')}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="rounded-xl p-2 text-slate-400 transition-all hover:bg-blue-50 hover:text-blue-600 dark:hover:bg-blue-900/30 dark:hover:text-blue-400"
                    title="LinkedIn"
                  >
                    <Linkedin className="h-5 w-5" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-20 bg-gray-100 dark:bg-gray-900">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-6">
            Our Mission
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
            We're on a mission to empower insurance companies with cutting-edge AI technology
            that prevents customer churn, improves retention rates, and drives sustainable growth.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <Users className="w-12 h-12 text-blue-600 dark:text-blue-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                Customer-Centric
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Putting customers first by predicting and preventing dissatisfaction before it leads to churn.
              </p>
            </div>
            <div>
              <Shield className="w-12 h-12 text-green-600 dark:text-green-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                Trust & Security
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Building solutions with enterprise-grade security and compliance standards.
              </p>
            </div>
            <div>
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white font-bold text-lg">AI</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                Innovation
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Leveraging the latest AI advancements to solve real-world insurance challenges.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-cyan-600">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <h2 className="text-4xl font-bold text-white mb-4">
            Join Us in Transforming Insurance
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            Experience the power of AI-driven churn prediction. Start your journey today.
          </p>
          <Link
            to="/dashboard"
            className="bg-white text-blue-600 px-8 py-4 rounded-lg font-semibold text-lg hover:bg-gray-50 transition-all duration-200 inline-flex items-center space-x-2"
          >
            <span>Get Started</span>
            <Shield className="w-5 h-5" />
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
