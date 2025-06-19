
import React from 'react';
import { Link } from 'react-router-dom';
import { createPageUrl } from '@/utils';
import { Film, Home, Info, Library, ListVideo } from 'lucide-react'; // Added ListVideo

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <header className="bg-white border-b shadow-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <Link to={createPageUrl("Home")} className="flex items-center gap-2">
            <div className="bg-indigo-600 text-white p-2 rounded-lg">
              <Film className="h-5 w-5" />
            </div>
            <h1 className="text-xl font-semibold text-gray-900">SPOKHAND SIGNCUT</h1>
          </Link>
          
          <nav className="flex items-center gap-3 sm:gap-4">
            <Link 
              to={createPageUrl("Home")} 
              className="flex items-center gap-1 text-gray-600 hover:text-indigo-600 text-sm font-medium p-2 rounded-md hover:bg-gray-100"
            >
              <Home className="h-4 w-4 sm:mr-1" />
              <span className="hidden sm:inline">Home</span>
            </Link>
            <Link 
              to={createPageUrl("Segments")} 
              className="flex items-center gap-1 text-gray-600 hover:text-indigo-600 text-sm font-medium p-2 rounded-md hover:bg-gray-100"
            >
              <ListVideo className="h-4 w-4 sm:mr-1" /> {/* Changed Icon and Text */}
              <span className="hidden sm:inline">Segments</span>
            </Link>
            <Link 
              to={createPageUrl("About")} 
              className="flex items-center gap-1 text-gray-600 hover:text-indigo-600 text-sm font-medium p-2 rounded-md hover:bg-gray-100"
            >
              <Info className="h-4 w-4 sm:mr-1" />
              <span className="hidden sm:inline">About</span>
            </Link>
          </nav>
        </div>
      </header>
      
      <main className="container mx-auto px-4 py-6 flex-grow">
        {children}
      </main>
    </div>
  );
}
