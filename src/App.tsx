import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { LandingPage } from './pages/LandingPage';
import { LearnMore } from './pages/LearnMore';
import { AboutUs } from './pages/AboutUs';
import { Dashboard } from './pages/Dashboard';
import { CustomerDetail } from './pages/CustomerDetail';

function App() {
  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/learn-more" element={<LearnMore />} />
          <Route path="/about-us" element={<AboutUs />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/customer/:id" element={<CustomerDetail />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
