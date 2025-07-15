import React, { useState, useEffect } from 'react';
import { Brain, Search, Zap, Play, Pause, RotateCcw, Download, Upload, Settings } from 'lucide-react';

export default function AdvancedSignSpotting({ videoId, onAnnotationsGenerated }) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [detectedSigns, setDetectedSigns] = useState([]);
  const [disambiguationResults, setDisambiguationResults] = useState([]);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [fusionStrategy, setFusionStrategy] = useState('late_fusion');
  const [vocabularySize, setVocabularySize] = useState(1000);
  const [beamWidth, setBeamWidth] = useState(5);
  const [alpha, setAlpha] = useState(0.9);

  // Processing stages based on the paper's two-stage architecture
  const processingStages = [
    'Extracting I3D spatiotemporal features (1024-dim)...',
    'Detecting left and right hands with MediaPipe...',
    'Extracting hand shape features (2048-dim each)...',
    'Performing dictionary-based matching (1000 vocab)...',
    `Applying ${fusionStrategy} feature fusion...`,
    'Running LLM disambiguation with beam search...',
    'Generating final annotations...'
  ];

  const simulateAdvancedSignSpotting = async () => {
    setIsProcessing(true);
    setDetectedSigns([]);
    setDisambiguationResults([]);
    setConfidence(0);

    // Simulate the two-stage process described in the paper
    for (let i = 0; i < processingStages.length; i++) {
      setProcessingStage(processingStages[i]);
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Simulate confidence building
      setConfidence((i + 1) * 0.15);
    }

    // Mock results based on the paper's approach
    const mockDetectedSigns = [
      {
        id: 1,
        startTime: 1.2,
        endTime: 2.1,
        sign: 'HELLO',
        confidence: 0.85,
        handShape: 'B',
        location: 'neutral space',
        features: {
          i3d: [0.23, 0.45, 0.67, 0.89, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12],
          lh: [0.34, 0.56, 0.78, 0.90, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12],
          rh: [0.45, 0.67, 0.89, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12, 0.34]
        }
      },
      {
        id: 2,
        startTime: 2.5,
        endTime: 3.8,
        sign: 'WORLD',
        confidence: 0.78,
        handShape: 'W',
        location: 'neutral space',
        features: {
          i3d: [0.45, 0.67, 0.89, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12, 0.34],
          lh: [0.56, 0.78, 0.90, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12, 0.34],
          rh: [0.67, 0.89, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12, 0.34, 0.56]
        }
      },
      {
        id: 3,
        startTime: 4.2,
        endTime: 5.5,
        sign: 'TOGETHER',
        confidence: 0.92,
        handShape: 'C',
        location: 'neutral space',
        features: {
          i3d: [0.67, 0.89, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12, 0.34, 0.56],
          lh: [0.78, 0.90, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12, 0.34, 0.56],
          rh: [0.89, 0.12, 0.34, 0.56, 0.78, 0.90, 0.12, 0.34, 0.56, 0.78]
        }
      }
    ];

    const mockDisambiguationResults = [
      {
        originalSign: 'HELLO',
        alternatives: ['HELLO', 'HI', 'GREETING'],
        llmScore: 0.95,
        context: 'Beginning of sentence',
        finalChoice: 'HELLO',
        beamSearchPath: ['HELLO', 'WORLD', 'TOGETHER']
      },
      {
        originalSign: 'WORLD',
        alternatives: ['WORLD', 'EARTH', 'PLANET'],
        llmScore: 0.87,
        context: 'Following "HELLO"',
        finalChoice: 'WORLD',
        beamSearchPath: ['HELLO', 'WORLD', 'TOGETHER']
      },
      {
        originalSign: 'TOGETHER',
        alternatives: ['TOGETHER', 'UNITED', 'COMBINED'],
        llmScore: 0.93,
        context: 'End of phrase',
        finalChoice: 'TOGETHER',
        beamSearchPath: ['HELLO', 'WORLD', 'TOGETHER']
      }
    ];

    setDetectedSigns(mockDetectedSigns);
    setDisambiguationResults(mockDisambiguationResults);
    setIsProcessing(false);
    setProcessingStage('');

    // Generate annotations for the parent component
    const annotations = mockDetectedSigns.map(sign => ({
      id: sign.id,
      videoId,
      startTime: sign.startTime,
      endTime: sign.endTime,
      label: sign.sign,
      confidence: sign.confidence,
      handShape: sign.handShape,
      location: sign.location,
      notes: `AI-detected using ${fusionStrategy} fusion. LLM disambiguation applied with beam width ${beamWidth}.`,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }));

    onAnnotationsGenerated(annotations);
  };

  const getFusionStrategyDescription = (strategy) => {
    switch (strategy) {
      case 'late_fusion':
        return 'S_Late = α * S_I3D + (1-α) * S_RH (Equation 4)';
      case 'intermediate_fusion':
        return 'F_Mid = F_I3D ⊕ F_RH ⊕ F_LH ∈ R^5120';
      case 'full_ensemble':
        return 'S_Ensemble = α * S_Mid + (1-α) * S_I3D (Equation 5)';
      default:
        return '';
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-purple-100 p-2 rounded-lg">
            <Brain className="h-6 w-6 text-purple-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Advanced Sign Spotting</h3>
            <p className="text-sm text-gray-600">LLM-powered sign detection and disambiguation</p>
          </div>
        </div>
        
        <button
          onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
          className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
        >
          <Settings className="h-4 w-4" />
          {showAdvancedOptions ? 'Hide' : 'Show'} Advanced Options
        </button>
      </div>

      {showAdvancedOptions && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-900 mb-3">Advanced Configuration</h4>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Feature Fusion Strategy
              </label>
              <select 
                value={fusionStrategy}
                onChange={(e) => setFusionStrategy(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              >
                <option value="late_fusion">Late Fusion (Best Performance)</option>
                <option value="intermediate_fusion">Intermediate Fusion</option>
                <option value="full_ensemble">Full Ensemble</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {getFusionStrategyDescription(fusionStrategy)}
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dictionary Size
              </label>
              <select 
                value={vocabularySize}
                onChange={(e) => setVocabularySize(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              >
                <option value={1000}>1,000 signs (Paper Default)</option>
                <option value={1500}>1,500 signs</option>
                <option value={2000}>2,000 signs</option>
                <option value={4373}>4,373 signs (Maximum)</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                LLM Beam Search Width
              </label>
              <input
                type="number"
                min="1"
                max="50"
                value={beamWidth}
                onChange={(e) => setBeamWidth(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Fusion Weight (α)
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={alpha}
                onChange={(e) => setAlpha(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-gray-500 mt-1">α = {alpha}</p>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <h5 className="font-medium text-blue-900 mb-2">Paper Implementation Details</h5>
            <ul className="text-xs text-blue-800 space-y-1">
              <li>• I3D Features: 1024-dimensional spatiotemporal features</li>
              <li>• Hand Features: 2048-dimensional each (LH + RH)</li>
              <li>• Dictionary: 1000 BSL vocabulary items</li>
              <li>• Late Fusion: α = 0.9 (optimal from paper)</li>
              <li>• Beam Width: 5 (default from paper)</li>
            </ul>
          </div>
        </div>
      )}

      <div className="mb-6">
        <button
          onClick={simulateAdvancedSignSpotting}
          disabled={isProcessing}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isProcessing ? (
            <>
              <Pause className="h-4 w-4" />
              Processing...
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              Start Advanced Analysis
            </>
          )}
        </button>
      </div>

      {isProcessing && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Processing Progress</span>
            <span className="text-sm text-gray-500">{Math.round(confidence * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-purple-600 h-2 rounded-full transition-all duration-500"
              style={{ width: `${confidence * 100}%` }}
            ></div>
          </div>
          <p className="text-sm text-gray-600 mt-2">{processingStage}</p>
        </div>
      )}

      {detectedSigns.length > 0 && (
        <div className="space-y-4">
          <h4 className="font-medium text-gray-900">Detected Signs</h4>
          <div className="space-y-3">
            {detectedSigns.map((sign) => (
              <div key={sign.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{sign.sign}</span>
                  <span className="text-sm text-gray-500">
                    {sign.startTime}s - {sign.endTime}s
                  </span>
                </div>
                <div className="grid grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Confidence:</span>
                    <span className="ml-1 font-medium">{Math.round(sign.confidence * 100)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Hand Shape:</span>
                    <span className="ml-1 font-medium">{sign.handShape}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Location:</span>
                    <span className="ml-1 font-medium">{sign.location}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Features:</span>
                    <span className="ml-1 font-medium">I3D+LH+RH</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {disambiguationResults.length > 0 && (
        <div className="mt-6">
          <h4 className="font-medium text-gray-900 mb-3">LLM Disambiguation Results</h4>
          <div className="space-y-3">
            {disambiguationResults.map((result, index) => (
              <div key={index} className="border border-blue-200 rounded-lg p-4 bg-blue-50">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{result.originalSign}</span>
                  <span className="text-sm text-blue-600">
                    LLM Score: {Math.round(result.llmScore * 100)}%
                  </span>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <span className="font-medium">Context:</span> {result.context}
                </div>
                <div className="text-sm">
                  <span className="text-gray-600">Alternatives:</span>
                  <span className="ml-1 font-medium">{result.alternatives.join(', ')}</span>
                </div>
                <div className="text-sm mt-1">
                  <span className="text-gray-600">Final Choice:</span>
                  <span className="ml-1 font-medium text-green-600">{result.finalChoice}</span>
                </div>
                <div className="text-sm mt-1">
                  <span className="text-gray-600">Beam Path:</span>
                  <span className="ml-1 font-medium text-purple-600">{result.beamSearchPath.join(' → ')}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {detectedSigns.length > 0 && (
        <div className="mt-6 flex gap-3">
          <button className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200">
            <Download className="h-4 w-4" />
            Export Results
          </button>
          <button className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200">
            <Upload className="h-4 w-4" />
            Save to Database
          </button>
        </div>
      )}
    </div>
  );
} 