import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import logo from '../logo.svg';

const navLinks = [
  { name: 'Home', to: '/Home' },
  { name: 'Segments', to: '/Segments' },
  { name: 'About', to: '/About' },
];

export default function Header() {
  const location = useLocation();
  return (
    <header className="w-full bg-white shadow-sm sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-8 flex items-center justify-between h-16">
        <div className="flex items-center gap-3">
          <img src={logo} alt="Spokhand Logo" className="h-8 w-8" />
          <span className="font-bold text-xl tracking-tight text-gray-900">SPOKHAND SIGNCUT</span>
        </div>
        <nav className="flex items-center gap-6">
          {navLinks.map(link => (
            <Link
              key={link.name}
              to={link.to}
              className={`text-sm font-medium transition-colors duration-200 px-2 py-1 rounded hover:bg-indigo-50 hover:text-indigo-700 ${location.pathname.includes(link.to) ? 'text-indigo-600' : 'text-gray-700'}`}
            >
              {link.name}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
} 