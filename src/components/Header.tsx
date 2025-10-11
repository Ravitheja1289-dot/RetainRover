import { Link } from 'react-router-dom';
import { Moon, Sun, Shield } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

export const Header = () => {
  const { isDark, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-50 border-b border-slate-200/70 bg-white/80 backdrop-blur-xl dark:border-slate-800/70 dark:bg-slate-900/80">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link to="/dashboard" className="flex items-center gap-3 transition-opacity hover:opacity-80">
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
            <button
              onClick={toggleTheme}
              className="rounded-xl bg-slate-100/80 p-2 transition-all hover:bg-slate-200/80 hover:shadow-lg dark:bg-slate-800/80 dark:hover:bg-slate-700/80"
              aria-label="Toggle theme"
            >
              {isDark ? (
                <Sun className="h-5 w-5 text-yellow-500" />
              ) : (
                <Moon className="h-5 w-5 text-slate-700" />
              )}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
