import Layout from "./Layout.jsx";

import Home from "./Home";

import Annotator from "./Annotator";

import About from "./About";

import Lexicon from "./Lexicon";

import Segments from "./Segments";

import CameraSettings from "./CameraSettings";
import CameraTest from "./CameraTest";
import Analysis from "./Analysis";
import Troubleshoot from "./Troubleshoot";

import ASLLex from "./ASLLex";
import AnalyticsDashboard from "./AnalyticsDashboard";
import InvestorPresentation from "./InvestorPresentation";

import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';

const PAGES = {
    
    Home: Home,
    
    Annotator: Annotator,
    
    About: About,
    
    Lexicon: Lexicon,
    
    Segments: Segments,
    
    CameraSettings: CameraSettings,
    
    CameraTest: CameraTest,
    
    Analysis: Analysis,
    
    Troubleshoot: Troubleshoot,
    
    ASLLex: ASLLex,
    AnalyticsDashboard: AnalyticsDashboard,
    InvestorPresentation: InvestorPresentation,
    
}

function _getCurrentPage(url) {
    if (url.endsWith('/')) {
        url = url.slice(0, -1);
    }
    let urlLastPart = url.split('/').pop();
    if (urlLastPart.includes('?')) {
        urlLastPart = urlLastPart.split('?')[0];
    }

    const pageName = Object.keys(PAGES).find(page => page.toLowerCase() === urlLastPart.toLowerCase());
    return pageName || Object.keys(PAGES)[0];
}

// Create a wrapper component that uses useLocation inside the Router context
function PagesContent() {
    const location = useLocation();
    const currentPage = _getCurrentPage(location.pathname);
    
    return (
        <Layout currentPageName={currentPage}>
            <Routes>            
                
                    <Route path="/" element={<Home />} />
                
                
                <Route path="/Home" element={<Home />} />
                
                <Route path="/Annotator" element={<Annotator />} />
                
                <Route path="/About" element={<About />} />
                
                <Route path="/Lexicon" element={<Lexicon />} />
                
                <Route path="/Segments" element={<Segments />} />
                
                <Route path="/CameraSettings" element={<CameraSettings />} />
                
                <Route path="/CameraTest" element={<CameraTest />} />
                
                <Route path="/Analysis" element={<Analysis />} />
                
                <Route path="/Troubleshoot" element={<Troubleshoot />} />
                
                <Route path="/ASLLex" element={<ASLLex />} />
                
                <Route path="/AnalyticsDashboard" element={<AnalyticsDashboard />} />
                
                <Route path="/InvestorPresentation" element={<InvestorPresentation />} />
                
            </Routes>
        </Layout>
    );
}

export default function Pages() {
    return (
        <Router>
            <PagesContent />
        </Router>
    );
}