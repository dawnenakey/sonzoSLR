import React, { useState } from 'react';
import { 
  Video, 
  FileText, 
  Users, 
  Briefcase, 
  Play, 
  Pause, 
  Square,
  Download,
  Upload,
  Settings,
  BarChart3,
  Target,
  Award
} from 'lucide-react';
import ResumeFeedback from '../components/ResumeFeedback';

export default function JobCoaching() {
  const [activeSession, setActiveSession] = useState(null);
  const [sessionType, setSessionType] = useState('interview');
  const [isRecording, setIsRecording] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [currentView, setCurrentView] = useState('sessions'); // sessions, resume

  const sessionTypes = [
    {
      id: 'interview',
      title: 'Interview Practice',
      description: 'Practice common interview questions in ASL',
      icon: Briefcase,
      color: 'bg-blue-500',
      features: ['Mock interviews', 'Confidence scoring', 'Body language feedback']
    },
    {
      id: 'resume',
      title: 'Resume Workshop',
      description: 'Learn to present your resume effectively in ASL',
      icon: FileText,
      color: 'bg-green-500',
      features: ['Resume presentation', 'Key achievements', 'Professional vocabulary']
    },
    {
      id: 'communication',
      title: 'Workplace Communication',
      description: 'Master workplace communication in ASL',
      icon: Users,
      color: 'bg-purple-500',
      features: ['Team meetings', 'Client interactions', 'Professional phrases']
    },
    {
      id: 'networking',
      title: 'Networking Skills',
      description: 'Build networking skills for career growth',
      icon: Target,
      color: 'bg-orange-500',
      features: ['Elevator pitch', 'Professional introductions', 'Follow-up strategies']
    }
  ];

  const startSession = (type) => {
    setSessionType(type);
    setActiveSession(type);
    setIsRecording(false);
    setFeedback(null);
  };

  const handleRecording = () => {
    setIsRecording(!isRecording);
    // TODO: Implement actual recording logic
  };

  const stopSession = () => {
    setIsRecording(false);
    setActiveSession(null);
    // TODO: Generate feedback
    setFeedback({
      confidence: 85,
      clarity: 90,
      bodyLanguage: 88,
      suggestions: [
        'Great eye contact maintained throughout',
        'Consider slowing down slightly for complex phrases',
        'Excellent use of facial expressions'
      ]
    });
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">ASL Job Coaching</h1>
        <p className="text-gray-600">
          Master professional communication in American Sign Language with AI-powered feedback
        </p>
        
        {/* Navigation Tabs */}
        <div className="flex space-x-1 mt-6 p-1 bg-gray-100 rounded-lg">
          <button
            onClick={() => setCurrentView('sessions')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              currentView === 'sessions'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Practice Sessions
          </button>
          <button
            onClick={() => setCurrentView('resume')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              currentView === 'resume'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Resume Feedback
          </button>
        </div>
      </div>

      {currentView === 'sessions' && (
        <>
          {!activeSession ? (
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {sessionTypes.map((session) => {
                const IconComponent = session.icon;
                return (
                  <div
                    key={session.id}
                    className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => startSession(session.id)}
                  >
                    <div className={`w-12 h-12 rounded-lg ${session.color} flex items-center justify-center mb-4`}>
                      <IconComponent className="h-6 w-6 text-white" />
                    </div>
                    <h3 className="font-semibold text-gray-900 mb-2">{session.title}</h3>
                    <p className="text-gray-600 text-sm mb-4">{session.description}</p>
                    <ul className="space-y-1">
                      {session.features.map((feature, index) => (
                        <li key={index} className="text-xs text-gray-500 flex items-center">
                          <div className="w-1 h-1 bg-gray-400 rounded-full mr-2"></div>
                          {feature}
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })}
            </div>
          ) : (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">
                {sessionTypes.find(s => s.id === sessionType)?.title}
              </h2>
              <p className="text-gray-600">
                {sessionTypes.find(s => s.id === sessionType)?.description}
              </p>
            </div>
            <button
              onClick={() => setActiveSession(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              ← Back to Sessions
            </button>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-2">Session Controls</h3>
                <div className="flex space-x-2">
                  <button
                    onClick={handleRecording}
                    className={`px-4 py-2 rounded-lg flex items-center space-x-2 ${
                      isRecording 
                        ? 'bg-red-500 text-white hover:bg-red-600' 
                        : 'bg-green-500 text-white hover:bg-green-600'
                    }`}
                  >
                    {isRecording ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    <span>{isRecording ? 'Pause' : 'Start'} Recording</span>
                  </button>
                  <button
                    onClick={stopSession}
                    className="px-4 py-2 rounded-lg bg-gray-500 text-white hover:bg-gray-600 flex items-center space-x-2"
                  >
                    <Square className="h-4 w-4" />
                    <span>End Session</span>
                  </button>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-2">Session Tips</h3>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Maintain good lighting for clear video</li>
                  <li>• Position yourself at arm's length from camera</li>
                  <li>• Use natural facial expressions</li>
                  <li>• Practice at a comfortable pace</li>
                </ul>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-2">Live Feedback</h3>
                {isRecording ? (
                  <div className="text-center py-8">
                    <div className="w-16 h-16 bg-red-500 rounded-full mx-auto mb-4 flex items-center justify-center">
                      <div className="w-4 h-4 bg-white rounded-full animate-pulse"></div>
                    </div>
                    <p className="text-gray-600">Recording in progress...</p>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Video className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">Ready to start recording</p>
                  </div>
                )}
              </div>

              {feedback && (
                <div className="bg-blue-50 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 mb-2">Session Results</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Confidence:</span>
                      <span className="font-semibold">{feedback.confidence}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Clarity:</span>
                      <span className="font-semibold">{feedback.clarity}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Body Language:</span>
                      <span className="font-semibold">{feedback.bodyLanguage}%</span>
                    </div>
                  </div>
                  <div className="mt-4">
                    <h4 className="font-medium text-gray-900 mb-2">Suggestions:</h4>
                    <ul className="space-y-1">
                      {feedback.suggestions.map((suggestion, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-start">
                          <div className="w-1 h-1 bg-blue-400 rounded-full mr-2 mt-2"></div>
                          {suggestion}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>
            </div>
          )}
        </>
      )}

      {currentView === 'resume' && (
        <ResumeFeedback />
      )}

      {currentView === 'sessions' && (
        <div className="mt-8 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Progress Dashboard</h2>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Sessions Completed</p>
                  <p className="text-2xl font-bold text-blue-600">12</p>
                </div>
                <Award className="h-8 w-8 text-blue-500" />
              </div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Average Score</p>
                  <p className="text-2xl font-bold text-green-600">87%</p>
                </div>
                <BarChart3 className="h-8 w-8 text-green-500" />
              </div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Hours Practiced</p>
                  <p className="text-2xl font-bold text-purple-600">8.5</p>
                </div>
                <Target className="h-8 w-8 text-purple-500" />
              </div>
            </div>
            <div className="bg-orange-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Skills Mastered</p>
                  <p className="text-2xl font-bold text-orange-600">6</p>
                </div>
                <Settings className="h-8 w-8 text-orange-500" />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 