import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  Video, 
  Brain, 
  Database, 
  DollarSign, 
  Activity,
  Download,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
  Zap,
  Target,
  Award,
  PieChart,
  LineChart,
  BarChart,
  Calendar,
  Filter,
  Settings
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/components/ui/use-toast';

export default function AnalyticsDashboard() {
  const [dashboardData, setDashboardData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState('30d');
  const [selectedMetric, setSelectedMetric] = useState('all');
  const [lastUpdated, setLastUpdated] = useState(null);
  const { toast } = useToast();

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 300000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, [selectedPeriod]);

  const fetchDashboardData = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('/api/analytics/dashboard');
      if (!response.ok) throw new Error('Failed to fetch dashboard data');
      
      const data = await response.json();
      setDashboardData(data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load analytics data"
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportReport = async (reportType = 'executive', format = 'json') => {
    try {
      const response = await fetch(`/api/analytics/export-report?type=${reportType}&format=${format}`);
      if (!response.ok) throw new Error('Export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `spokhand_analytics_${reportType}_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast({
        title: "Export Complete",
        description: "Analytics report exported successfully"
      });
    } catch (error) {
      console.error('Export failed:', error);
      toast({
        variant: "destructive",
        title: "Export Failed",
        description: "Could not export analytics report"
      });
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  };

  const formatCurrency = (num) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(num);
  };

  const formatPercentage = (num) => {
    return `${num.toFixed(1)}%`;
  };

  const getTrendIcon = (trend) => {
    if (trend.direction === 'increasing') return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (trend.direction === 'decreasing') return <TrendingUp className="h-4 w-4 text-red-500 rotate-180" />;
    return <Activity className="h-4 w-4 text-gray-500" />;
  };

  const getTrendColor = (trend) => {
    if (trend.direction === 'increasing') return 'text-green-600';
    if (trend.direction === 'decreasing') return 'text-red-600';
    return 'text-gray-600';
  };

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-gray-500" />
          <span className="ml-2 text-gray-500">Loading analytics dashboard...</span>
        </div>
      </div>
    );
  }

  if (!dashboardData) {
    return (
      <div className="max-w-7xl mx-auto p-6">
        <div className="text-center py-12">
          <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Unable to load analytics</h3>
          <p className="text-gray-500 mb-4">There was an error loading the analytics dashboard.</p>
          <Button onClick={fetchDashboardData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  const { system_metrics, user_engagement, business_metrics, ai_performance, data_quality, trends } = dashboardData;

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <BarChart3 className="h-8 w-8 text-indigo-600" />
              Analytics Dashboard
            </h1>
            <p className="text-gray-600 mt-2">
              Comprehensive insights and performance metrics for SpokHand SLR
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="7d">7 Days</SelectItem>
                <SelectItem value="30d">30 Days</SelectItem>
                <SelectItem value="90d">90 Days</SelectItem>
                <SelectItem value="1y">1 Year</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={fetchDashboardData} variant="outline" size="sm">
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button onClick={() => handleExportReport('executive', 'json')} variant="outline" size="sm">
              <Download className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {lastUpdated && (
          <p className="text-sm text-gray-500">
            Last updated: {lastUpdated.toLocaleString()}
          </p>
        )}
      </div>

      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Users</p>
                <p className="text-2xl font-bold text-gray-900">{formatNumber(system_metrics.total_users)}</p>
                <div className="flex items-center mt-1">
                  {getTrendIcon(trends.trends.users)}
                  <span className={`text-sm ml-1 ${getTrendColor(trends.trends.users)}`}>
                    {formatPercentage(trends.trends.users.percentage)}
                  </span>
                </div>
              </div>
              <div className="bg-blue-100 p-3 rounded-lg">
                <Users className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Monthly Revenue</p>
                <p className="text-2xl font-bold text-gray-900">{formatCurrency(business_metrics.monthly_recurring_revenue)}</p>
                <div className="flex items-center mt-1">
                  {getTrendIcon(trends.trends.revenue)}
                  <span className={`text-sm ml-1 ${getTrendColor(trends.trends.revenue)}`}>
                    {formatPercentage(trends.trends.revenue.percentage)}
                  </span>
                </div>
              </div>
              <div className="bg-green-100 p-3 rounded-lg">
                <DollarSign className="h-6 w-6 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">AI Accuracy</p>
                <p className="text-2xl font-bold text-gray-900">{formatPercentage(ai_performance.overall_accuracy)}</p>
                <div className="flex items-center mt-1">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm ml-1 text-green-600">Excellent</span>
                </div>
              </div>
              <div className="bg-purple-100 p-3 rounded-lg">
                <Brain className="h-6 w-6 text-purple-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">System Uptime</p>
                <p className="text-2xl font-bold text-gray-900">{formatPercentage(system_metrics.system_uptime)}</p>
                <div className="flex items-center mt-1">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm ml-1 text-green-600">Stable</span>
                </div>
              </div>
              <div className="bg-amber-100 p-3 rounded-lg">
                <Activity className="h-6 w-6 text-amber-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Metrics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* System Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Performance
            </CardTitle>
            <CardDescription>Key system metrics and health indicators</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Active Users (30d)</p>
                <p className="text-xl font-semibold">{formatNumber(system_metrics.active_users_30d)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Videos</p>
                <p className="text-xl font-semibold">{formatNumber(system_metrics.total_videos)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Annotations</p>
                <p className="text-xl font-semibold">{formatNumber(system_metrics.total_annotations)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">AI Analyses</p>
                <p className="text-xl font-semibold">{formatNumber(system_metrics.ai_analysis_count)}</p>
              </div>
            </div>
            <div className="pt-4 border-t">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-600">Success Rate</span>
                <span className="text-sm font-medium">{formatPercentage(system_metrics.success_rate)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full" 
                  style={{ width: `${system_metrics.success_rate}%` }}
                ></div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Business Metrics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              Business Intelligence
            </CardTitle>
            <CardDescription>Revenue, growth, and business KPIs</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Total Revenue</p>
                <p className="text-xl font-semibold">{formatCurrency(business_metrics.total_revenue)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Growth Rate</p>
                <p className="text-xl font-semibold text-green-600">{formatPercentage(business_metrics.growth_rate)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Customer LTV</p>
                <p className="text-xl font-semibold">{formatCurrency(business_metrics.customer_lifetime_value)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">ROI</p>
                <p className="text-xl font-semibold text-green-600">{formatPercentage(business_metrics.roi_percentage)}</p>
              </div>
            </div>
            <div className="pt-4 border-t">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Churn Rate</span>
                <span className="text-sm font-medium text-red-600">{formatPercentage(business_metrics.churn_rate)}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* AI Performance & User Engagement */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* AI Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              AI Performance
            </CardTitle>
            <CardDescription>Model accuracy and processing metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Sign Detection</span>
                <span className="text-sm font-medium">{formatPercentage(ai_performance.sign_detection_accuracy)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full" 
                  style={{ width: `${ai_performance.sign_detection_accuracy}%` }}
                ></div>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Handshape Classification</span>
                <span className="text-sm font-medium">{formatPercentage(ai_performance.handshape_classification_accuracy)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-purple-500 h-2 rounded-full" 
                  style={{ width: `${ai_performance.handshape_classification_accuracy}%` }}
                ></div>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Disambiguation</span>
                <span className="text-sm font-medium">{formatPercentage(ai_performance.disambiguation_accuracy)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full" 
                  style={{ width: `${ai_performance.disambiguation_accuracy}%` }}
                ></div>
              </div>
            </div>
            
            <div className="pt-4 border-t">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">Processing Time</p>
                  <p className="font-medium">{ai_performance.processing_time_per_video.toFixed(2)}s/video</p>
                </div>
                <div>
                  <p className="text-gray-600">Vocabulary Coverage</p>
                  <p className="font-medium">{formatPercentage(ai_performance.vocabulary_coverage)}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* User Engagement */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              User Engagement
            </CardTitle>
            <CardDescription>User behavior and engagement metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Daily Active Users</p>
                <p className="text-xl font-semibold">{formatNumber(user_engagement.daily_active_users)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Weekly Active Users</p>
                <p className="text-xl font-semibold">{formatNumber(user_engagement.weekly_active_users)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Session Duration</p>
                <p className="text-xl font-semibold">{user_engagement.average_session_duration.toFixed(1)}m</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Satisfaction Score</p>
                <p className="text-xl font-semibold">{user_engagement.user_satisfaction_score.toFixed(1)}/5</p>
              </div>
            </div>
            
            <div className="pt-4 border-t">
              <p className="text-sm font-medium text-gray-900 mb-3">Feature Adoption</p>
              <div className="space-y-2">
                {Object.entries(user_engagement.feature_adoption_rate).map(([feature, rate]) => (
                  <div key={feature} className="flex justify-between items-center">
                    <span className="text-sm text-gray-600 capitalize">{feature.replace('_', ' ')}</span>
                    <span className="text-sm font-medium">{formatPercentage(rate)}</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Quality & Export Options */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Quality */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Data Quality
            </CardTitle>
            <CardDescription>Data validation and quality metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Quality Score</p>
                <p className="text-xl font-semibold">{formatPercentage(data_quality.quality_score)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Validation Rate</p>
                <p className="text-xl font-semibold">{formatPercentage(data_quality.lexicon_validation_rate)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Data Completeness</p>
                <p className="text-xl font-semibold">{formatPercentage(data_quality.data_completeness)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Schema Compliance</p>
                <p className="text-xl font-semibold">{formatPercentage(data_quality.schema_compliance)}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Export & Actions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Download className="h-5 w-5" />
              Reports & Export
            </CardTitle>
            <CardDescription>Generate and download analytics reports</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <Button 
                onClick={() => handleExportReport('executive', 'json')} 
                className="w-full justify-start"
                variant="outline"
              >
                <Download className="h-4 w-4 mr-2" />
                Executive Summary (JSON)
              </Button>
              <Button 
                onClick={() => handleExportReport('detailed', 'csv')} 
                className="w-full justify-start"
                variant="outline"
              >
                <Download className="h-4 w-4 mr-2" />
                Detailed Report (CSV)
              </Button>
              <Button 
                onClick={() => handleExportReport('technical', 'json')} 
                className="w-full justify-start"
                variant="outline"
              >
                <Download className="h-4 w-4 mr-2" />
                Technical Metrics (JSON)
              </Button>
            </div>
            
            <div className="pt-4 border-t">
              <p className="text-sm text-gray-600 mb-2">Quick Actions</p>
              <div className="flex gap-2">
                <Button size="sm" variant="outline">
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Refresh
                </Button>
                <Button size="sm" variant="outline">
                  <Settings className="h-4 w-4 mr-1" />
                  Settings
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
