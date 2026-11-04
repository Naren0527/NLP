import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppLayout } from './components/layout/AppLayout';
import { OverviewPage } from './pages/OverviewPage';
import { LiveDemoPage } from './pages/LiveDemoPage';
import { BenchmarkPage } from './pages/BenchmarkPage';
import { CalibrationPage } from './pages/CalibrationPage';
import { ConsistencyPage } from './pages/ConsistencyPage';
import { StrategiesPage } from './pages/StrategiesPage';
import { ReportsPage } from './pages/ReportsPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<OverviewPage />} />
          <Route path="live-demo" element={<LiveDemoPage />} />
          <Route path="benchmark" element={<BenchmarkPage />} />
          <Route path="calibration" element={<CalibrationPage />} />
          <Route path="consistency" element={<ConsistencyPage />} />
          <Route path="strategies" element={<StrategiesPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
