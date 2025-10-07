import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './Navbar';
import PredictPage from './PredictPage';
import Species from './AllSpecies';

function App() {
  const [isServerOnline, setIsServerOnline] = useState(true);

  const checkServerHealth = async () => {
    try {
      const response = await fetch("http://localhost:5000/");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      setIsServerOnline(true);
    } catch (error) {
      console.error("Server is offline:", error);
      setIsServerOnline(false);
    }
  };

  useEffect(() => {
    checkServerHealth();
    const intervalId = setInterval(checkServerHealth, 10000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <Router>
      <Navbar isServerOnline={isServerOnline} />
      <Routes>
        <Route path="/" element={<PredictPage />} />
        <Route path="/species" element={<Species />} />
      </Routes>
    </Router>
  );
}

export default App;