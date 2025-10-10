import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  Users, 
  DollarSign, 
  Brain, 
  Target, 
  Award, 
  BarChart3, 
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Download,
  Share2,
  Zap,
  Globe,
  Shield,
  Clock,
  CheckCircle,
  Star,
  ArrowRight,
  PieChart,
  LineChart
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/components/ui/use-toast';

export default function InvestorPresentation() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [dashboardData, setDashboardData] = useState(null);
  const { toast } = useToast();

  useEffect(() => {
    fetchDashboardData();
  }, []);

  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentSlide((prev) => (prev + 1) % slides.length);
      }, 8000); // 8 seconds per slide
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/api/analytics/dashboard');
      if (response.ok) {
        const data = await response.json();
        setDashboardData(data);
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    }
  };

  const slides = [
    {
      id: 'title',
      title: 'SpokHand SLR',
      subtitle: 'Revolutionary Sign Language Recognition Platform',
      content: (
        <div className="text-center space-y-8">
          <div className="space-y-4">
            <h1 className="text-6xl font-bold text-gray-900">SpokHand SLR</h1>
            <p className="text-2xl text-gray-600">Revolutionary Sign Language Recognition Platform</p>
            <p className="text-lg text-gray-500 max-w-3xl mx-auto">
              Advanced AI-powered sign language recognition with comprehensive analytics, 
              lexicon management, and real-time processing capabilities.
            </p>
          </div>
          <div className="flex justify-center gap-4">
            <Badge variant="outline" className="text-lg px-4 py-2">
              <Brain className="h-5 w-5 mr-2" />
              AI-Powered
            </Badge>
            <Badge variant="outline" className="text-lg px-4 py-2">
              <Globe className="h-5 w-5 mr-2" />
              Scalable
            </Badge>
            <Badge variant="outline" className="text-lg px-4 py-2">
              <Shield className="h-5 w-5 mr-2" />
              Enterprise-Ready
            </Badge>
          </div>
        </div>
      )
    },
    {
      id: 'problem',
      title: 'The Problem',
      subtitle: 'Addressing a Critical Communication Gap',
      content: (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h3 className="text-2xl font-bold text-red-600">Communication Barriers</h3>
              <ul className="space-y-3 text-lg">
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <span>70M+ deaf and hard-of-hearing individuals globally</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <span>Limited real-time sign language recognition</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <span>High cost of human interpreters</span>
                </li>
                <li className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-red-500 rounded-full mt-2"></div>
                  <span>Inconsistent accuracy in existing solutions</span>
                </li>
              </ul>
            </div>
            <div className="space-y-4">
              <h3 className="text-2xl font-bold text-blue-600">Market Opportunity</h3>
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-3xl font-bold text-blue-600">$2.8B</p>
                  <p className="text-sm text-blue-800">Assistive Technology Market</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-3xl font-bold text-green-600">23.5%</p>
                  <p className="text-sm text-green-800">Annual Growth Rate</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="text-3xl font-bold text-purple-600">$850</p>
                  <p className="text-sm text-purple-800">Average Customer LTV</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'solution',
      title: 'Our Solution',
      subtitle: 'Advanced AI-Powered Sign Language Recognition',
      content: (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="border-2 border-blue-200">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-blue-600">
                  <Brain className="h-6 w-6" />
                  AI Recognition
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Two-stage architecture
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    I3D + Hand shape analysis
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    LLM disambiguation
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    89.3% accuracy rate
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="border-2 border-green-200">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-green-600">
                  <BarChart3 className="h-6 w-6" />
                  Analytics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Real-time metrics
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Performance tracking
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Business intelligence
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    ROI analytics
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="border-2 border-purple-200">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-purple-600">
                  <Users className="h-6 w-6" />
                  Management
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Lexicon management
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    User analytics
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Batch operations
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Export/import
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      )
    },
    {
      id: 'technology',
      title: 'Technology Stack',
      subtitle: 'Cutting-Edge AI and Cloud Infrastructure',
      content: (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-900">AI Architecture</h3>
              <div className="space-y-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-600">Sign Spotting Stage</h4>
                  <p className="text-sm text-gray-600">I3D spatiotemporal features + Hand shape analysis</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-600">Disambiguation Stage</h4>
                  <p className="text-sm text-gray-600">LLM-powered context-aware disambiguation</p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-600">Fusion Strategies</h4>
                  <p className="text-sm text-gray-600">Late, Intermediate, and Full Ensemble approaches</p>
                </div>
              </div>
            </div>
            
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-900">Infrastructure</h3>
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-600">AWS Cloud</h4>
                  <p className="text-sm text-gray-600">DynamoDB, S3, Lambda, API Gateway</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-600">Frontend</h4>
                  <p className="text-sm text-gray-600">React, TypeScript, Tailwind CSS</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-purple-600">AI Models</h4>
                  <p className="text-sm text-gray-600">PyTorch, MediaPipe, Custom Models</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'metrics',
      title: 'Key Metrics',
      subtitle: 'Proven Performance and Growth',
      content: (
        <div className="space-y-8">
          {dashboardData ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-600">
                  {dashboardData.system_metrics.total_users.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">Total Users</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600">
                  ${dashboardData.business_metrics.monthly_recurring_revenue.toLocaleString()}
                </div>
                <div className="text-sm text-gray-600">Monthly Revenue</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-purple-600">
                  {dashboardData.ai_performance.overall_accuracy.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">AI Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-amber-600">
                  {dashboardData.business_metrics.growth_rate.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">Growth Rate</div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-2 text-gray-600">Loading metrics...</p>
            </div>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">340%</div>
                <div className="text-sm text-gray-600">ROI</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">99.9%</div>
                <div className="text-sm text-gray-600">Uptime</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-6 text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">4.6/5</div>
                <div className="text-sm text-gray-600">User Satisfaction</div>
              </CardContent>
            </Card>
          </div>
        </div>
      )
    },
    {
      id: 'market',
      title: 'Market Opportunity',
      subtitle: 'Large and Growing Market',
      content: (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-900">Target Markets</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                  <span className="font-medium">Healthcare</span>
                  <span className="text-2xl font-bold text-blue-600">$850M</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg">
                  <span className="font-medium">Education</span>
                  <span className="text-2xl font-bold text-green-600">$650M</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-purple-50 rounded-lg">
                  <span className="font-medium">Enterprise</span>
                  <span className="text-2xl font-bold text-purple-600">$1.2B</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-amber-50 rounded-lg">
                  <span className="font-medium">Consumer</span>
                  <span className="text-2xl font-bold text-amber-600">$1.1B</span>
                </div>
              </div>
            </div>
            
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-900">Competitive Advantage</h3>
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <Star className="h-5 w-5 text-yellow-500" />
                  <span>Superior accuracy (89.3% vs 65% industry average)</span>
                </div>
                <div className="flex items-center gap-3">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  <span>Real-time processing capabilities</span>
                </div>
                <div className="flex items-center gap-3">
                  <Shield className="h-5 w-5 text-yellow-500" />
                  <span>Enterprise-grade security</span>
                </div>
                <div className="flex items-center gap-3">
                  <Globe className="h-5 w-5 text-yellow-500" />
                  <span>Scalable cloud architecture</span>
                </div>
                <div className="flex items-center gap-3">
                  <BarChart3 className="h-5 w-5 text-yellow-500" />
                  <span>Comprehensive analytics</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'business',
      title: 'Business Model',
      subtitle: 'Sustainable Revenue Streams',
      content: (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="border-2 border-blue-200">
              <CardHeader>
                <CardTitle className="text-blue-600">SaaS Subscriptions</CardTitle>
                <CardDescription>Recurring monthly revenue</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">$15K</div>
                  <div className="text-sm text-gray-600">Monthly Recurring Revenue</div>
                  <div className="text-sm text-green-600">+23.5% growth</div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-2 border-green-200">
              <CardHeader>
                <CardTitle className="text-green-600">API Usage</CardTitle>
                <CardDescription>Pay-per-analysis model</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">$0.15</div>
                  <div className="text-sm text-gray-600">Per Analysis</div>
                  <div className="text-sm text-green-600">High margin</div>
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-2 border-purple-200">
              <CardHeader>
                <CardTitle className="text-purple-600">Enterprise</CardTitle>
                <CardDescription>Custom solutions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">$850</div>
                  <div className="text-sm text-gray-600">Customer LTV</div>
                  <div className="text-sm text-green-600">5.2% churn</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )
    },
    {
      id: 'roadmap',
      title: 'Roadmap',
      subtitle: 'Future Development Plans',
      content: (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h3 className="text-2xl font-bold text-gray-900">Q1 2024</h3>
              <ul className="space-y-2">
                <li className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Core platform launch</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>AI model optimization</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Analytics dashboard</span>
                </li>
              </ul>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-2xl font-bold text-gray-900">Q2 2024</h3>
              <ul className="space-y-2">
                <li className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-blue-500" />
                  <span>Mobile app development</span>
                </li>
                <li className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-blue-500" />
                  <span>Multi-language support</span>
                </li>
                <li className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-blue-500" />
                  <span>Enterprise features</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'investment',
      title: 'Investment Opportunity',
      subtitle: 'Join the Future of Communication',
      content: (
        <div className="text-center space-y-8">
          <div className="space-y-4">
            <h2 className="text-4xl font-bold text-gray-900">Seeking $2M Series A</h2>
            <p className="text-xl text-gray-600">To accelerate growth and market expansion</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="space-y-4">
              <h3 className="text-2xl font-bold text-gray-900">Use of Funds</h3>
              <div className="space-y-3 text-left">
                <div className="flex justify-between">
                  <span>Product Development</span>
                  <span className="font-bold">40%</span>
                </div>
                <div className="flex justify-between">
                  <span>Sales & Marketing</span>
                  <span className="font-bold">35%</span>
                </div>
                <div className="flex justify-between">
                  <span>Team Expansion</span>
                  <span className="font-bold">20%</span>
                </div>
                <div className="flex justify-between">
                  <span>Operations</span>
                  <span className="font-bold">5%</span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-2xl font-bold text-gray-900">Expected Returns</h3>
              <div className="space-y-3">
                <div className="bg-green-50 p-4 rounded-lg">
                  <div className="text-3xl font-bold text-green-600">10x</div>
                  <div className="text-sm text-green-800">Projected ROI in 3 years</div>
                </div>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="text-3xl font-bold text-blue-600">$50M</div>
                  <div className="text-sm text-blue-800">Target valuation</div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex justify-center gap-4">
            <Button size="lg" className="bg-blue-600 hover:bg-blue-700">
              <Download className="h-5 w-5 mr-2" />
              Download Pitch Deck
            </Button>
            <Button size="lg" variant="outline">
              <Share2 className="h-5 w-5 mr-2" />
              Share Presentation
            </Button>
          </div>
        </div>
      )
    }
  ];

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const goToSlide = (index) => {
    setCurrentSlide(index);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-xl font-bold text-gray-900">Investor Presentation</h1>
              <Badge variant="outline">Live Demo</Badge>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={togglePlayPause} variant="outline" size="sm">
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button onClick={prevSlide} variant="outline" size="sm">
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button onClick={nextSlide} variant="outline" size="sm">
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Slide Counter */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 py-2">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              Slide {currentSlide + 1} of {slides.length}
            </div>
            <div className="flex gap-1">
              {slides.map((_, index) => (
                <button
                  key={index}
                  onClick={() => goToSlide(index)}
                  className={`w-2 h-2 rounded-full ${
                    index === currentSlide ? 'bg-blue-600' : 'bg-gray-300'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-8">
        <Card className="min-h-[600px]">
          <CardContent className="p-12">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold text-gray-900 mb-2">
                {slides[currentSlide].title}
              </h1>
              <p className="text-xl text-gray-600">
                {slides[currentSlide].subtitle}
              </p>
            </div>
            
            <div className="min-h-[400px] flex items-center justify-center">
              {slides[currentSlide].content}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Footer */}
      <div className="bg-white border-t mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              SpokHand SLR - Advanced Sign Language Recognition Platform
            </div>
            <div className="flex items-center gap-4">
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
              <Button variant="outline" size="sm">
                <Share2 className="h-4 w-4 mr-2" />
                Share
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
