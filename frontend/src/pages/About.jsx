import React from 'react';
import { ArrowRight, Video, Clock, FileJson, Users, Monitor, Keyboard } from 'lucide-react';

export default function About() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">About the Sign Language Annotation Tool</h1>
      
      <section className="mb-10">
        <p className="text-lg text-gray-700 mb-6">
          This annotation tool was designed specifically for researchers, linguists, and educators 
          working with sign languages. It allows for precise segmentation and categorization of 
          sign language videos, creating standardized annotations that can be used for further analysis.
        </p>
        
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Key Features</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-xl shadow-sm flex items-start gap-4">
            <div className="bg-indigo-100 p-3 rounded-lg">
              <Video className="h-6 w-6 text-indigo-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-1">Video Segmentation</h3>
              <p className="text-gray-600 text-sm">
                Precisely mark segments in sign language videos with a simple interface
              </p>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm flex items-start gap-4">
            <div className="bg-green-100 p-3 rounded-lg">
              <Clock className="h-6 w-6 text-green-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-1">Exact Timecodes</h3>
              <p className="text-gray-600 text-sm">
                Automatically generate precise timestamps for each annotation
              </p>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm flex items-start gap-4">
            <div className="bg-amber-100 p-3 rounded-lg">
              <FileJson className="h-6 w-6 text-amber-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-1">JSON Export</h3>
              <p className="text-gray-600 text-sm">
                Export standardized JSON data for use in research and analysis
              </p>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-xl shadow-sm flex items-start gap-4">
            <div className="bg-blue-100 p-3 rounded-lg">
              <Keyboard className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-1">Efficient Controls</h3>
              <p className="text-gray-600 text-sm">
                Use keyboard shortcuts for fast, efficient annotation workflows
              </p>
            </div>
          </div>
        </div>
      </section>
      
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Use Cases</h2>
        
        <div className="bg-white p-6 rounded-xl shadow-md mb-6">
          <h3 className="font-semibold text-gray-900 mb-3">Research</h3>
          <p className="text-gray-600 mb-3">
            Perfect for linguistic analysis, studying sign language phonology, morphology, 
            and syntax. The tool helps researchers identify patterns and structures in sign 
            languages with precision.
          </p>
          <ul className="list-disc list-inside text-gray-600 space-y-1">
            <li>Document regional sign language variations</li>
            <li>Analyze prosodic features in signing</li>
            <li>Study transitional movements between signs</li>
          </ul>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-md mb-6">
          <h3 className="font-semibold text-gray-900 mb-3">Education</h3>
          <p className="text-gray-600 mb-3">
            Teaching sign language requires clear segmentation of individual signs. 
            This tool helps educators create precise learning materials and analyze 
            student sign production.
          </p>
          <ul className="list-disc list-inside text-gray-600 space-y-1">
            <li>Create sign language learning resources</li>
            <li>Provide detailed feedback for learners</li>
            <li>Document diverse signing styles for teaching</li>
          </ul>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-md">
          <h3 className="font-semibold text-gray-900 mb-3">Machine Learning</h3>
          <p className="text-gray-600 mb-3">
            Creating training data for sign language recognition systems requires 
            precisely segmented and annotated videos. Our tool creates high-quality 
            datasets for AI development.
          </p>
          <ul className="list-disc list-inside text-gray-600 space-y-1">
            <li>Generate training data for sign language recognition</li>
            <li>Create validation datasets for machine learning models</li>
            <li>Support the development of sign language translation technology</li>
          </ul>
        </div>
      </section>
      
      <section className="mb-10">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Technical Information</h2>
        
        <div className="bg-white p-6 rounded-xl shadow-md">
          <p className="text-gray-600 mb-4">
            The Sign Language Annotation Tool is built with modern web technologies and designed 
            to be fast, responsive, and easy to use. The annotations are stored securely and 
            exported in a standardized JSON format for compatibility with other research tools.
          </p>
          
          <h3 className="font-semibold text-gray-900 mb-2">JSON Output Format</h3>
          <pre className="bg-gray-50 p-4 rounded-md text-xs overflow-x-auto font-mono mb-4">
{`[
  {
    "segment_type": "SIGN_UNIT",
    "start_time": "00:00:19",
    "duration": "03",
    "description": "The video shows a young woman signing in ASL."
  },
  {
    "segment_type": "PAUSE",
    "start_time": "00:00:22",
    "duration": "01",
    "description": "The woman pauses in neutral position."
  }
]`}
          </pre>
          
          <p className="text-gray-600">
            This standardized format makes it easy to incorporate your annotations into 
            other research tools and databases.
          </p>
        </div>
      </section>
    </div>
  );
}