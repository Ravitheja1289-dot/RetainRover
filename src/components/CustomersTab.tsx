import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  ChevronDown,
  Filter,
  LayoutGrid,
  List,
  Mail,
  MessageCircle,
  PhoneCall,
  Search,
  Sparkles,
  TrendingUp,
  Upload,
  Users2
} from 'lucide-react';
import Papa from 'papaparse';
import { api, Customer } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';

type RiskFilter = 'all' | 'low' | 'medium' | 'high';
type SortOption = 'risk' | 'tenure' | 'name';
type ViewMode = 'cards' | 'table';

type RiskMeta = {
  label: string;
  color: string;
  bg: string;
  icon: typeof CheckCircle;
  accent: string;
};

const riskPalette: Record<'low' | 'medium' | 'high', { solid: string; soft: string }> = {
  low: { solid: '#10b981', soft: 'rgba(16, 185, 129, 0.12)' },
  medium: { solid: '#f59e0b', soft: 'rgba(245, 158, 11, 0.12)' },
  high: { solid: '#ef4444', soft: 'rgba(239, 68, 68, 0.12)' }
};

const formatCurrency = (value: number) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
  }).format(value);

export const CustomersTab = () => {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all');
  const [sortOption, setSortOption] = useState<SortOption>('risk');
  const [viewMode, setViewMode] = useState<ViewMode>('cards');
  const navigate = useNavigate();

  const fetchCustomers = async () => {
    try {
      const data = await api.getCustomers();
      setCustomers(data);
    } catch (error) {
      console.error('Failed to fetch customers:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomers();
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv') {
      Papa.parse(file, {
        header: true,
        complete: (results) => {
          const data = results.data as any[];
          const parsedCustomers: Customer[] = data.map((row, index) => ({
            id: parseInt(row.id) || index + 1,
            name: row.name || '',
            region: row.region || '',
            churnProbability: parseFloat(row.churnProbability) || 0,
            age: parseInt(row.age) || 0,
            tenure: parseInt(row.tenure) || 0,
            premiumAmount: parseFloat(row.premiumAmount) || 0,
            claimsCount: parseInt(row.claimsCount) || 0,
            lastInteraction: row.lastInteraction || ''
          }));
          api.setCustomers(parsedCustomers);
          fetchCustomers();
        },
        error: (error) => {
          console.error('Error parsing CSV:', error);
          alert('Error parsing CSV file. Please check the format.');
        }
      });
    } else {
      alert('Please select a valid CSV file.');
    }
  };

  const getRiskLevel = (probability: number): RiskMeta => {
    if (probability < 30) {
      return {
        label: 'Low',
        color: 'text-emerald-600 dark:text-emerald-400',
        bg: 'bg-emerald-50/80 dark:bg-emerald-900/30',
        icon: CheckCircle,
        accent: 'low'
      };
    }
    if (probability < 60) {
      return {
        label: 'Medium',
        color: 'text-amber-600 dark:text-amber-400',
        bg: 'bg-amber-50/80 dark:bg-amber-900/30',
        icon: AlertCircle,
        accent: 'medium'
      };
    }
    return {
      label: 'High',
      color: 'text-rose-600 dark:text-rose-400',
      bg: 'bg-rose-50/80 dark:bg-rose-900/30',
      icon: AlertTriangle,
      accent: 'high'
    };
  };

  const heroCustomer = useMemo(() => {
    if (!customers.length) return null;
    return [...customers].sort((a, b) => b.churnProbability - a.churnProbability)[0];
  }, [customers]);

  const averageProbability = useMemo(() => {
    if (!customers.length) return 0;
    const total = customers.reduce((sum, customer) => sum + (customer.churnProbability || 0), 0);
    return Math.round(total / customers.length);
  }, [customers]);

  const riskBreakdown = useMemo(() => {
    const base = { low: 0, medium: 0, high: 0 };
    return customers.reduce((acc, customer) => {
      if (customer.churnProbability < 30) acc.low += 1;
      else if (customer.churnProbability < 60) acc.medium += 1;
      else acc.high += 1;
      return acc;
    }, base);
  }, [customers]);

  const filteredCustomers = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase();
    const bySearch = customers.filter((customer) => {
      if (!normalizedSearch) return true;
      const haystack = `${customer.name} ${customer.region} ${customer.id}`.toLowerCase();
      return haystack.includes(normalizedSearch);
    });

    const byRisk = bySearch.filter((customer) => {
      if (riskFilter === 'all') return true;
      if (riskFilter === 'low') return customer.churnProbability < 30;
      if (riskFilter === 'medium') return customer.churnProbability >= 30 && customer.churnProbability < 60;
      return customer.churnProbability >= 60;
    });

    const sorted = [...byRisk].sort((a, b) => {
      switch (sortOption) {
        case 'tenure':
          return b.tenure - a.tenure;
        case 'name':
          return a.name.localeCompare(b.name);
        case 'risk':
        default:
          return b.churnProbability - a.churnProbability;
      }
    });

    return sorted;
  }, [customers, riskFilter, searchTerm, sortOption]);

  if (loading) return <LoadingSpinner />;

  return (
    <div className="space-y-8 p-6">
      <div className="relative overflow-hidden rounded-3xl border border-blue-500/20 bg-gradient-to-br from-blue-600/10 via-cyan-500/10 to-transparent p-8 shadow-lg transition-shadow hover:shadow-2xl dark:border-blue-400/20 dark:from-blue-500/15 dark:via-sky-500/10">
        <div className="absolute -top-24 -right-12 h-64 w-64 rounded-full bg-blue-500/30 blur-3xl animate-pulse-slow" aria-hidden="true"></div>
        <div className="absolute -bottom-20 -left-10 h-64 w-64 rounded-full bg-cyan-500/20 blur-3xl animate-pulse-slow" aria-hidden="true"></div>
        <div className="relative grid gap-8 lg:grid-cols-[1.2fr_1fr]">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full bg-white/70 px-4 py-2 text-sm font-medium text-blue-700 shadow-sm ring-1 ring-blue-500/20 backdrop-blur dark:bg-slate-900/70 dark:text-blue-200">
              <Sparkles className="h-4 w-4" />
              AI-assisted retention cockpit
            </div>
            <div className="space-y-3">
              <h2 className="text-3xl font-semibold text-slate-900 dark:text-white sm:text-4xl">
                Turn churn signals into personalised rescue playbooks
              </h2>
              <p className="text-base text-slate-600 dark:text-slate-300 sm:max-w-xl">
                Instantly segment, prioritise, and action your customer list with live risk insights, proactive nudges, and guided next-best steps inspired by world-class v0 design templates.
              </p>
            </div>
            <div className="flex flex-wrap gap-3 text-sm text-slate-600 dark:text-slate-300">
              <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
                <TrendingUp className="h-4 w-4 text-blue-500" />
                Average churn probability {averageProbability}%
              </span>
              <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
                <Users2 className="h-4 w-4 text-blue-500" />
                {customers.length} active profiles
              </span>
              <span className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-2 ring-1 ring-slate-200/70 backdrop-blur dark:bg-slate-900/70 dark:ring-slate-700/70">
                <Sparkles className="h-4 w-4 text-blue-500" />
                {riskBreakdown.high} high-risk alerts
              </span>
            </div>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => setViewMode('cards')}
                className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-600/30 transition-transform hover:-translate-y-0.5 hover:bg-blue-500"
              >
                Launch interactive workspace
                <ArrowRight className="h-4 w-4" />
              </button>
              <label className="group inline-flex cursor-pointer items-center gap-2 rounded-xl border border-blue-500/40 bg-white/60 px-4 py-3 text-sm font-medium text-blue-600 transition-all hover:-translate-y-0.5 hover:border-blue-500 hover:text-blue-700 hover:shadow-lg dark:border-blue-400/40 dark:bg-slate-900/70 dark:text-blue-300 dark:hover:border-blue-300 dark:hover:text-blue-200">
                <Upload className="h-4 w-4 transition-transform group-hover:-translate-y-0.5" />
                Upload CSV
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="absolute inset-0 h-full w-full opacity-0"
                />
              </label>
            </div>
          </div>

          <div className="relative overflow-hidden rounded-3xl bg-white/80 p-6 shadow-xl ring-1 ring-slate-200/80 backdrop-blur dark:bg-slate-900/80 dark:ring-slate-800/80">
            <div className="absolute -right-6 -top-6 h-32 w-32 rounded-full bg-blue-500/20 blur-2xl" aria-hidden="true"></div>
            <div className="relative space-y-4">
              <p className="text-sm font-medium uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">Spotlight</p>
              {heroCustomer ? (
                <div className="space-y-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Highest churn probability</p>
                      <h3 className="text-2xl font-semibold text-slate-900 dark:text-white">{heroCustomer.name}</h3>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Customer #{heroCustomer.id}</p>
                    </div>
                    <button
                      onClick={() => navigate(`/customer/${heroCustomer.id}`)}
                      className="inline-flex items-center gap-2 rounded-full bg-blue-600/10 px-4 py-2 text-xs font-semibold text-blue-600 transition-colors hover:bg-blue-600/20 dark:bg-blue-500/10 dark:text-blue-300"
                    >
                      View playbook
                      <ArrowRight className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-center">
                    {[heroCustomer.churnProbability, heroCustomer.tenure, heroCustomer.claimsCount].map((value, index) => (
                      <div key={index} className="rounded-2xl border border-slate-200/70 bg-white/70 p-3 backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
                        <p className="text-xs uppercase tracking-widest text-slate-500 dark:text-slate-400">
                          {index === 0 ? 'Churn risk' : index === 1 ? 'Tenure' : 'Claims'}
                        </p>
                        <p className="mt-2 text-lg font-semibold text-slate-900 dark:text-white">
                          {index === 0 ? `${value}%` : index === 1 ? `${value} yrs` : value}
                        </p>
                      </div>
                    ))}
                  </div>
                  <div className="space-y-2 rounded-2xl border border-slate-200/70 bg-gradient-to-r from-blue-500/10 via-cyan-500/10 to-transparent p-4 dark:border-slate-800/70">
                    <p className="text-sm font-medium text-slate-900 dark:text-white">Recommended nudge</p>
                    <ul className="space-y-2 text-sm text-slate-600 dark:text-slate-300">
                      <li className="flex items-center gap-2">
                        <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                        Offer loyalty discount tied to tenure extension
                      </li>
                      <li className="flex items-center gap-2">
                        <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                        Schedule personalised call with retention specialist
                      </li>
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                  Upload a CSV to populate customers
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
        <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-sm backdrop-blur transition hover:-translate-y-0.5 hover:shadow-lg dark:border-slate-800/70 dark:bg-slate-900/70">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Total customers</p>
          <p className="mt-3 text-3xl font-semibold text-slate-900 dark:text-white">{customers.length}</p>
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">Synced from latest data refresh</p>
        </div>
        <div className="rounded-2xl border border-emerald-500/30 bg-emerald-50/70 p-5 shadow-sm backdrop-blur transition hover:-translate-y-0.5 hover:shadow-lg dark:border-emerald-500/30 dark:bg-emerald-900/30">
          <p className="text-xs font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-300">Low risk cohort</p>
          <p className="mt-3 text-3xl font-semibold text-emerald-600 dark:text-emerald-300">{riskBreakdown.low}</p>
          <p className="mt-2 text-xs text-emerald-700/70 dark:text-emerald-200/80">Retain with periodic nurture</p>
        </div>
        <div className="rounded-2xl border border-amber-500/30 bg-amber-50/70 p-5 shadow-sm backdrop-blur transition hover:-translate-y-0.5 hover:shadow-lg dark:border-amber-500/30 dark:bg-amber-900/30">
          <p className="text-xs font-semibold uppercase tracking-wider text-amber-700 dark:text-amber-300">Medium risk</p>
          <p className="mt-3 text-3xl font-semibold text-amber-600 dark:text-amber-300">{riskBreakdown.medium}</p>
          <p className="mt-2 text-xs text-amber-700/70 dark:text-amber-200/80">Prime for targeted campaigns</p>
        </div>
        <div className="rounded-2xl border border-rose-500/30 bg-rose-50/70 p-5 shadow-sm backdrop-blur transition hover:-translate-y-0.5 hover:shadow-lg dark:border-rose-500/30 dark:bg-rose-900/30">
          <p className="text-xs font-semibold uppercase tracking-wider text-rose-700 dark:text-rose-300">High risk</p>
          <p className="mt-3 text-3xl font-semibold text-rose-600 dark:text-rose-300">{riskBreakdown.high}</p>
          <p className="mt-2 text-xs text-rose-700/70 dark:text-rose-200/80">Immediate intervention suggested</p>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200/70 bg-white/80 p-5 shadow-sm backdrop-blur dark:border-slate-800/70 dark:bg-slate-900/70">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-1 items-center gap-3 rounded-xl border border-slate-200/80 bg-white/70 px-4 py-3 text-sm shadow-sm dark:border-slate-700/70 dark:bg-slate-900/70">
            <Search className="h-4 w-4 text-slate-400" />
            <input
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Search by name, region, or ID"
              className="w-full bg-transparent text-slate-700 placeholder:text-slate-400 focus:outline-none dark:text-slate-200"
            />
          </div>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="flex flex-wrap items-center gap-2">
              {(
                [
                  { id: 'all', label: 'All' },
                  { id: 'low', label: 'Low' },
                  { id: 'medium', label: 'Medium' },
                  { id: 'high', label: 'High' }
                ] as { id: RiskFilter; label: string }[]
              ).map((chip) => (
                <button
                  key={chip.id}
                  onClick={() => setRiskFilter(chip.id)}
                  className={`inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-semibold transition-all ${
                    riskFilter === chip.id
                      ? 'border-blue-500 bg-blue-600 text-white shadow-sm'
                      : 'border-slate-200/80 bg-white/60 text-slate-500 hover:border-blue-500/60 hover:text-blue-600 dark:border-slate-700/80 dark:bg-slate-900/80 dark:text-slate-300'
                  }`}
                >
                  {chip.label} risk
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2 rounded-xl border border-slate-200/80 bg-white/70 px-3 py-2 text-xs font-semibold text-slate-500 shadow-sm dark:border-slate-700/80 dark:bg-slate-900/80 dark:text-slate-300">
              <Filter className="h-4 w-4 text-blue-500" />
              <span>Sort</span>
              <div className="relative">
                <select
                  value={sortOption}
                  onChange={(event) => setSortOption(event.target.value as SortOption)}
                  className="appearance-none bg-transparent pl-1 pr-6 text-slate-600 focus:outline-none dark:text-slate-200"
                >
                  <option value="risk">Highest risk</option>
                  <option value="tenure">Longest tenure</option>
                  <option value="name">Alphabetical</option>
                </select>
                <ChevronDown className="pointer-events-none absolute right-0 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
              </div>
            </div>

            <div className="inline-flex rounded-xl border border-slate-200/80 bg-white/70 p-1 text-xs shadow-sm dark:border-slate-700/80 dark:bg-slate-900/80">
              <button
                onClick={() => setViewMode('cards')}
                className={`flex items-center gap-1 rounded-lg px-3 py-1.5 transition ${
                  viewMode === 'cards'
                    ? 'bg-blue-600 text-white shadow'
                    : 'text-slate-500 hover:text-blue-600 dark:text-slate-300'
                }`}
              >
                <LayoutGrid className="h-4 w-4" /> Cards
              </button>
              <button
                onClick={() => setViewMode('table')}
                className={`flex items-center gap-1 rounded-lg px-3 py-1.5 transition ${
                  viewMode === 'table'
                    ? 'bg-blue-600 text-white shadow'
                    : 'text-slate-500 hover:text-blue-600 dark:text-slate-300'
                }`}
              >
                <List className="h-4 w-4" /> Table
              </button>
            </div>
          </div>
        </div>
      </div>

      {filteredCustomers.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 rounded-3xl border border-slate-200/80 bg-white/70 p-12 text-center text-slate-500 dark:border-slate-800/80 dark:bg-slate-900/70 dark:text-slate-300">
          <Sparkles className="h-8 w-8 text-blue-500" />
          <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">No customers match your filters (yet!)</h3>
          <p className="max-w-md text-sm text-slate-500 dark:text-slate-300">
            Try adjusting the risk chips or reset your search to explore the complete cohort.
          </p>
        </div>
      ) : viewMode === 'cards' ? (
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          {filteredCustomers.map((customer, index) => {
            const risk = getRiskLevel(customer.churnProbability);
            const RiskIcon = risk.icon;
            const accent = riskPalette[risk.accent as keyof typeof riskPalette];
            const initial = customer.name ? customer.name.charAt(0).toUpperCase() : '?';

            return (
              <div
                key={customer.id}
                className="group relative overflow-hidden rounded-3xl border border-transparent bg-white/90 p-6 shadow-lg ring-1 ring-slate-200/70 transition-all duration-300 hover:-translate-y-2 hover:border-blue-500/40 hover:shadow-2xl hover:ring-blue-200 dark:bg-slate-900/80 dark:ring-slate-800/70"
                style={{ animationDelay: `${index * 40}ms` }}
              >
                <div className="absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" style={{
                  background: `linear-gradient(135deg, ${accent.solid}22, transparent 55%)`
                }} aria-hidden="true"></div>
                <div className="relative grid gap-6 lg:grid-cols-5">
                  <div className="lg:col-span-2 flex flex-col gap-4">
                    <div className="flex items-start gap-4">
                      <div className="relative flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 text-2xl font-semibold text-white shadow-lg">
                        {initial}
                        <span className="absolute inset-0 rounded-2xl border border-white/30" aria-hidden="true"></span>
                      </div>
                      <div>
                        <button
                          onClick={() => navigate(`/customer/${customer.id}`)}
                          className="text-left"
                        >
                          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                            {customer.name || 'Unnamed customer'}
                          </h3>
                          <p className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">
                            ID {customer.id} · {customer.region}
                          </p>
                        </button>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold ${risk.bg} ${risk.color}`}>
                        <RiskIcon className="h-4 w-4" />
                        {risk.label} risk
                      </span>
                      <span className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500">
                        {customer.churnProbability}% probability
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="rounded-2xl border border-slate-200/80 bg-white/70 p-3 dark:border-slate-700/80 dark:bg-slate-900/70">
                        <p className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">Tenure</p>
                        <p className="mt-1 text-base font-semibold text-slate-900 dark:text-white">{customer.tenure} yrs</p>
                      </div>
                      <div className="rounded-2xl border border-slate-200/80 bg-white/70 p-3 dark:border-slate-700/80 dark:bg-slate-900/70">
                        <p className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">Premium</p>
                        <p className="mt-1 text-base font-semibold text-slate-900 dark:text-white">{formatCurrency(customer.premiumAmount)}</p>
                      </div>
                    </div>
                  </div>

                  <div className="lg:col-span-2 flex flex-col justify-between gap-4">
                    <div className="relative flex items-center gap-4 rounded-3xl border border-slate-200/80 bg-white/80 p-4 dark:border-slate-700/80 dark:bg-slate-900/70">
                      <div
                        className="flex h-16 w-16 items-center justify-center rounded-full text-lg font-semibold text-white shadow-inner"
                        style={{
                          background: `conic-gradient(${accent.solid} ${customer.churnProbability * 3.6}deg, rgba(15, 23, 42, 0.08) 0deg)`
                        }}
                      >
                        <span className="text-base font-semibold text-slate-900 dark:text-white">
                          {customer.churnProbability}%
                        </span>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm font-semibold text-slate-900 dark:text-white">Risk trajectory</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          Last interaction {customer.lastInteraction ? new Date(customer.lastInteraction).toLocaleDateString('en-US', {
                            month: 'short',
                            day: 'numeric'
                          }) : 'n/a'}
                        </p>
                        <div className="h-1.5 overflow-hidden rounded-full bg-slate-200/80 dark:bg-slate-700/80">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(customer.churnProbability, 100)}%`,
                              background: `linear-gradient(90deg, ${accent.solid}, ${accent.soft})`
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-3xl border border-slate-200/80 bg-white/80 p-4 text-sm dark:border-slate-700/80 dark:bg-slate-900/70">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Next best actions</p>
                      <ul className="mt-2 space-y-2 text-slate-600 dark:text-slate-300">
                        <li className="flex items-center gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                          Trigger personalised retention offer
                        </li>
                        <li className="flex items-center gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                          Follow-up within 3 business days
                        </li>
                      </ul>
                    </div>
                  </div>

                  <div className="lg:col-span-1 flex flex-col justify-between gap-3">
                    <div className="rounded-3xl border border-slate-200/80 bg-white/70 p-4 dark:border-slate-700/80 dark:bg-slate-900/70">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Engagement</p>
                      <div className="mt-3 space-y-2 text-xs text-slate-500 dark:text-slate-400">
                        <p>Claims filed: <span className="font-semibold text-slate-800 dark:text-slate-200">{customer.claimsCount}</span></p>
                        <p>Age: <span className="font-semibold text-slate-800 dark:text-slate-200">{customer.age}</span></p>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs font-semibold text-blue-600 dark:text-blue-300">
                      <button
                        onClick={() => navigate(`/customer/${customer.id}`)}
                        className="flex items-center justify-center gap-1 rounded-xl border border-blue-500/30 bg-blue-500/10 px-2 py-2 transition hover:bg-blue-500/20"
                      >
                        <PhoneCall className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => navigate(`/customer/${customer.id}`)}
                        className="flex items-center justify-center gap-1 rounded-xl border border-blue-500/30 bg-blue-500/10 px-2 py-2 transition hover:bg-blue-500/20"
                      >
                        <Mail className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => navigate(`/customer/${customer.id}`)}
                        className="flex items-center justify-center gap-1 rounded-xl border border-blue-500/30 bg-blue-500/10 px-2 py-2 transition hover:bg-blue-500/20"
                      >
                        <MessageCircle className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="overflow-hidden rounded-3xl border border-slate-200/80 bg-white/90 shadow-sm ring-1 ring-slate-100/80 dark:border-slate-800/80 dark:bg-slate-900/70 dark:ring-slate-800/80">
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-slate-50/80 text-xs uppercase tracking-wider text-slate-500 ring-1 ring-slate-200/80 dark:bg-slate-900/70 dark:text-slate-400 dark:ring-slate-800/80">
                <tr>
                  <th className="px-6 py-4">Customer</th>
                  <th className="px-6 py-4">Region</th>
                  <th className="px-6 py-4">Risk</th>
                  <th className="px-6 py-4">Probability</th>
                  <th className="px-6 py-4">Tenure</th>
                  <th className="px-6 py-4">Premium</th>
                  <th className="px-6 py-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100/80 text-sm dark:divide-slate-800/80">
                {filteredCustomers.map((customer) => {
                  const risk = getRiskLevel(customer.churnProbability);
                  const RiskIcon = risk.icon;
                  return (
                    <tr
                      key={customer.id}
                      className="transition hover:bg-blue-50/40 hover:shadow-inner dark:hover:bg-blue-500/10"
                    >
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 text-base font-semibold text-white">
                            {customer.name ? customer.name.charAt(0).toUpperCase() : '?'}
                          </div>
                          <div>
                            <p className="font-semibold text-slate-900 dark:text-white">{customer.name || 'Unnamed customer'}</p>
                            <p className="text-xs text-slate-500 dark:text-slate-400">ID {customer.id}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-slate-600 dark:text-slate-300">{customer.region}</td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${risk.bg} ${risk.color}`}>
                          <RiskIcon className="h-3.5 w-3.5" />
                          {risk.label}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-200/80 dark:bg-slate-700/80">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${Math.min(customer.churnProbability, 100)}%`,
                                background: `linear-gradient(90deg, rgba(59,130,246,0.9), rgba(14,165,233,0.8))`
                              }}
                            ></div>
                          </div>
                          <span className="font-semibold text-slate-700 dark:text-slate-200">{customer.churnProbability}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-slate-600 dark:text-slate-300">{customer.tenure} yrs</td>
                      <td className="px-6 py-4 text-slate-600 dark:text-slate-300">{formatCurrency(customer.premiumAmount)}</td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={() => navigate(`/customer/${customer.id}`)}
                          className="inline-flex items-center gap-2 rounded-full border border-blue-500/40 px-3 py-1.5 text-xs font-semibold text-blue-600 transition hover:bg-blue-500/10 dark:text-blue-300"
                        >
                          View plan
                          <ArrowRight className="h-3.5 w-3.5" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};