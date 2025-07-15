import React, { useState } from 'react';
import { 
  FileText, 
  Upload, 
  Download, 
  CheckCircle, 
  AlertCircle, 
  Info,
  Star,
  Eye,
  Edit,
  Save,
  X
} from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Progress } from './ui/progress';

export default function ResumeFeedback() {
  const [resumeFile, setResumeFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [showPreview, setShowPreview] = useState(false);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf' || file.type.includes('text')) {
      setResumeFile(file);
      setFeedback(null);
    }
  };

  const analyzeResume = async () => {
    if (!resumeFile) return;
    
    setIsAnalyzing(true);
    
    // Simulate analysis delay
    setTimeout(() => {
      const mockFeedback = generateResumeFeedback();
      setFeedback(mockFeedback);
      setIsAnalyzing(false);
    }, 2000);
  };

  const generateResumeFeedback = () => {
    return {
      overallScore: 78,
      sections: {
        contact: { score: 95, issues: [], suggestions: [] },
        summary: { score: 70, issues: ['Missing ASL-specific skills'], suggestions: ['Add "Fluent in American Sign Language" to summary'] },
        experience: { score: 80, issues: ['Could highlight accessibility work'], suggestions: ['Emphasize any experience with Deaf community or accessibility projects'] },
        skills: { score: 65, issues: ['Missing ASL proficiency'], suggestions: ['Add "American Sign Language (Fluent)" to skills section'] },
        education: { score: 90, issues: [], suggestions: [] }
      },
      aslSpecificSuggestions: [
        'Include ASL fluency prominently in your summary',
        'Add any experience working with Deaf community',
        'Mention accessibility and inclusion experience',
        'Highlight communication skills that transfer to ASL contexts',
        'Consider adding a "Languages" section with ASL listed first'
      ],
      generalSuggestions: [
        'Use action verbs to start bullet points',
        'Quantify achievements where possible',
        'Keep bullet points concise and impactful',
        'Ensure consistent formatting throughout',
        'Proofread for grammar and spelling'
      ],
      accessibilityTips: [
        'Consider how your resume reads when translated to ASL',
        'Use clear, direct language that translates well',
        'Avoid idioms that don\'t translate to ASL',
        'Emphasize visual and spatial skills',
        'Highlight any experience with visual communication'
      ]
    };
  };

  const getScoreColor = (score) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 80) return 'text-blue-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadge = (score) => {
    if (score >= 90) return <Badge className="bg-green-100 text-green-800">Excellent</Badge>;
    if (score >= 80) return <Badge className="bg-blue-100 text-blue-800">Good</Badge>;
    if (score >= 70) return <Badge className="bg-yellow-100 text-yellow-800">Fair</Badge>;
    return <Badge className="bg-red-100 text-red-800">Needs Work</Badge>;
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <FileText className="h-6 w-6 text-blue-600" />
            <span>Resume Feedback for ASL Users</span>
          </CardTitle>
          <CardDescription>
            Upload your resume for AI-powered feedback specifically tailored for American Sign Language users
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* File Upload */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              <p className="text-sm text-gray-600 mb-4">
                Upload your resume (PDF, DOC, or TXT)
              </p>
              <input
                type="file"
                accept=".pdf,.doc,.docx,.txt"
                onChange={handleFileUpload}
                className="hidden"
                id="resume-upload"
              />
              <label htmlFor="resume-upload">
                <Button asChild>
                  <span>Choose File</span>
                </Button>
              </label>
              {resumeFile && (
                <div className="mt-4 flex items-center justify-center space-x-2">
                  <FileText className="h-4 w-4 text-green-600" />
                  <span className="text-sm font-medium">{resumeFile.name}</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setResumeFile(null)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>

            {/* Analyze Button */}
            {resumeFile && (
              <div className="flex justify-center">
                <Button
                  onClick={analyzeResume}
                  disabled={isAnalyzing}
                  className="flex items-center space-x-2"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Eye className="h-4 w-4" />
                      <span>Analyze Resume</span>
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Feedback Results */}
      {feedback && (
        <div className="space-y-6">
          {/* Overall Score */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Overall Resume Score</span>
                {getScoreBadge(feedback.overallScore)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center space-x-4">
                  <div className="flex-1">
                    <Progress value={feedback.overallScore} className="h-3" />
                  </div>
                  <span className={`text-2xl font-bold ${getScoreColor(feedback.overallScore)}`}>
                    {feedback.overallScore}/100
                  </span>
                </div>
                <p className="text-sm text-gray-600">
                  Your resume is {feedback.overallScore >= 80 ? 'strong' : feedback.overallScore >= 70 ? 'good' : 'needs improvement'} for ASL users. 
                  Focus on the suggestions below to make it even better.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Section Scores */}
          <Card>
            <CardHeader>
              <CardTitle>Section Analysis</CardTitle>
              <CardDescription>
                Detailed feedback for each resume section
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {Object.entries(feedback.sections).map(([section, data]) => (
                  <div key={section} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold capitalize">{section}</h3>
                      <div className="flex items-center space-x-2">
                        <span className={`text-lg font-bold ${getScoreColor(data.score)}`}>
                          {data.score}/100
                        </span>
                        {getScoreBadge(data.score)}
                      </div>
                    </div>
                    <Progress value={data.score} className="h-2 mb-3" />
                    {data.issues.length > 0 && (
                      <div className="mb-2">
                        <h4 className="text-sm font-medium text-red-600 mb-1">Issues:</h4>
                        <ul className="text-sm text-red-600 space-y-1">
                          {data.issues.map((issue, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                              <span>{issue}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {data.suggestions.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium text-blue-600 mb-1">Suggestions:</h4>
                        <ul className="text-sm text-blue-600 space-y-1">
                          {data.suggestions.map((suggestion, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                              <span>{suggestion}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* ASL-Specific Suggestions */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Star className="h-5 w-5 text-yellow-600" />
                <span>ASL-Specific Recommendations</span>
              </CardTitle>
              <CardDescription>
                Tailored suggestions for American Sign Language users
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {feedback.aslSpecificSuggestions.map((suggestion, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                    <Info className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span className="text-sm">{suggestion}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Accessibility Tips */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Eye className="h-5 w-5 text-purple-600" />
                <span>Accessibility & Translation Tips</span>
              </CardTitle>
              <CardDescription>
                How your resume translates to ASL contexts
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {feedback.accessibilityTips.map((tip, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-purple-50 rounded-lg">
                    <Edit className="h-5 w-5 text-purple-600 mt-0.5 flex-shrink-0" />
                    <span className="text-sm">{tip}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* General Suggestions */}
          <Card>
            <CardHeader>
              <CardTitle>General Resume Tips</CardTitle>
              <CardDescription>
                Standard resume improvement suggestions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {feedback.generalSuggestions.map((suggestion, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <span className="text-sm">{suggestion}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="flex justify-center space-x-4">
            <Button variant="outline" className="flex items-center space-x-2">
              <Download className="h-4 w-4" />
              <span>Download Feedback Report</span>
            </Button>
            <Button className="flex items-center space-x-2">
              <Save className="h-4 w-4" />
              <span>Save for Later</span>
            </Button>
          </div>
        </div>
      )}
    </div>
  );
} 