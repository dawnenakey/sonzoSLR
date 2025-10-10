"""
ASLLRP Integration Module for Sentence-Level ASL Recognition

This module handles integration with Boston University's ASLLRP dataset
for training sentence-level ASL recognition models.

Author: SpokHand SLR Team
"""

import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ASLLRPSentence:
    """Represents a sentence-level ASL annotation from ASLLRP."""
    video_id: str
    sentence_id: str
    glosses: List[str]  # List of sign glosses in order
    timestamps: List[Tuple[float, float]]  # Start/end times for each sign
    signer_id: str
    sentence_type: str  # conversation, narrative, etc.
    metadata: Dict[str, any]

@dataclass
class ASLLRPVideo:
    """Represents a video file from ASLLRP with its annotations."""
    video_path: str
    video_id: str
    duration: float
    resolution: Tuple[int, int]
    frame_rate: float
    sentences: List[ASLLRPSentence]
    metadata: Dict[str, any]

class ASLLRPDataLoader:
    """Loads and processes ASLLRP dataset for sentence-level training."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.videos = []
        self.vocabulary = set()
        
    def load_dataset(self) -> List[ASLLRPVideo]:
        """Load the complete ASLLRP dataset."""
        logger.info(f"Loading ASLLRP dataset from {self.data_root}")
        
        # Load video files
        video_files = self._find_video_files()
        
        # Load annotations
        annotation_files = self._find_annotation_files()
        
        # Process each video with its annotations
        for video_file in video_files:
            video_id = self._extract_video_id(video_file)
            annotation_file = self._find_annotation_for_video(video_id, annotation_files)
            
            if annotation_file:
                video_data = self._process_video_with_annotations(video_file, annotation_file)
                self.videos.append(video_data)
                
        logger.info(f"Loaded {len(self.videos)} videos with sentence-level annotations")
        return self.videos
    
    def _find_video_files(self) -> List[Path]:
        """Find all video files in the dataset."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.data_root.rglob(f'*{ext}'))
            
        return video_files
    
    def _find_annotation_files(self) -> List[Path]:
        """Find all annotation files (XML, JSON, CSV)."""
        annotation_extensions = ['.xml', '.json', '.csv', '.ssp']
        annotation_files = []
        
        for ext in annotation_extensions:
            annotation_files.extend(self.data_root.rglob(f'*{ext}'))
            
        return annotation_files
    
    def _extract_video_id(self, video_path: Path) -> str:
        """Extract video ID from file path."""
        return video_path.stem
    
    def _find_annotation_for_video(self, video_id: str, annotation_files: List[Path]) -> Optional[Path]:
        """Find annotation file that corresponds to a video."""
        for ann_file in annotation_files:
            if video_id in ann_file.name:
                return ann_file
        return None
    
    def _process_video_with_annotations(self, video_path: Path, annotation_path: Path) -> ASLLRPVideo:
        """Process a video file with its annotations."""
        # Get video metadata
        video_metadata = self._get_video_metadata(video_path)
        
        # Parse annotations
        sentences = self._parse_annotations(annotation_path)
        
        # Build vocabulary
        for sentence in sentences:
            self.vocabulary.update(sentence.glosses)
        
        return ASLLRPVideo(
            video_path=str(video_path),
            video_id=self._extract_video_id(video_path),
            duration=video_metadata.get('duration', 0.0),
            resolution=video_metadata.get('resolution', (1920, 1080)),
            frame_rate=video_metadata.get('frame_rate', 30.0),
            sentences=sentences,
            metadata=video_metadata
        )
    
    def _get_video_metadata(self, video_path: Path) -> Dict[str, any]:
        """Extract metadata from video file."""
        # This would use OpenCV or similar to get video properties
        # For now, return default values
        return {
            'duration': 0.0,  # Would extract from video
            'resolution': (1920, 1080),
            'frame_rate': 30.0,
            'file_size': video_path.stat().st_size
        }
    
    def _parse_annotations(self, annotation_path: Path) -> List[ASLLRPSentence]:
        """Parse annotation file to extract sentence-level data."""
        sentences = []
        
        if annotation_path.suffix == '.xml':
            sentences = self._parse_xml_annotations(annotation_path)
        elif annotation_path.suffix == '.json':
            sentences = self._parse_json_annotations(annotation_path)
        elif annotation_path.suffix == '.csv':
            sentences = self._parse_csv_annotations(annotation_path)
        
        return sentences
    
    def _parse_xml_annotations(self, xml_path: Path) -> List[ASLLRPSentence]:
        """Parse XML annotation file (SignStream format)."""
        sentences = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Parse SignStream XML structure
            for sentence_elem in root.findall('.//sentence'):
                sentence_id = sentence_elem.get('id', '')
                glosses = []
                timestamps = []
                
                for sign_elem in sentence_elem.findall('.//sign'):
                    gloss = sign_elem.get('gloss', '')
                    start_time = float(sign_elem.get('start', 0.0))
                    end_time = float(sign_elem.get('end', 0.0))
                    
                    glosses.append(gloss)
                    timestamps.append((start_time, end_time))
                
                if glosses:  # Only add non-empty sentences
                    sentence = ASLLRPSentence(
                        video_id=self._extract_video_id(xml_path),
                        sentence_id=sentence_id,
                        glosses=glosses,
                        timestamps=timestamps,
                        signer_id=sentence_elem.get('signer', ''),
                        sentence_type=sentence_elem.get('type', 'conversation'),
                        metadata={}
                    )
                    sentences.append(sentence)
                    
        except ET.ParseError as e:
            logger.error(f"Error parsing XML file {xml_path}: {e}")
            
        return sentences
    
    def _parse_json_annotations(self, json_path: Path) -> List[ASLLRPSentence]:
        """Parse JSON annotation file."""
        sentences = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse JSON structure (format depends on ASLLRP export)
            for sentence_data in data.get('sentences', []):
                sentence = ASLLRPSentence(
                    video_id=data.get('video_id', ''),
                    sentence_id=sentence_data.get('id', ''),
                    glosses=sentence_data.get('glosses', []),
                    timestamps=sentence_data.get('timestamps', []),
                    signer_id=sentence_data.get('signer', ''),
                    sentence_type=sentence_data.get('type', 'conversation'),
                    metadata=sentence_data.get('metadata', {})
                )
                sentences.append(sentence)
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing JSON file {json_path}: {e}")
            
        return sentences
    
    def _parse_csv_annotations(self, csv_path: Path) -> List[ASLLRPSentence]:
        """Parse CSV annotation file."""
        sentences = []
        
        try:
            df = pd.read_csv(csv_path)
            
            # Group by sentence_id to create sentence objects
            for sentence_id, group in df.groupby('sentence_id'):
                glosses = group['gloss'].tolist()
                timestamps = list(zip(group['start_time'], group['end_time']))
                
                sentence = ASLLRPSentence(
                    video_id=group['video_id'].iloc[0],
                    sentence_id=sentence_id,
                    glosses=glosses,
                    timestamps=timestamps,
                    signer_id=group['signer_id'].iloc[0],
                    sentence_type=group['sentence_type'].iloc[0],
                    metadata={}
                )
                sentences.append(sentence)
                
        except Exception as e:
            logger.error(f"Error parsing CSV file {csv_path}: {e}")
            
        return sentences
    
    def get_vocabulary(self) -> List[str]:
        """Get the complete vocabulary from the dataset."""
        return sorted(list(self.vocabulary))
    
    def get_sentence_statistics(self) -> Dict[str, any]:
        """Get statistics about the sentence-level data."""
        if not self.videos:
            return {}
        
        total_sentences = sum(len(video.sentences) for video in self.videos)
        sentence_lengths = []
        
        for video in self.videos:
            for sentence in video.sentences:
                sentence_lengths.append(len(sentence.glosses))
        
        return {
            'total_videos': len(self.videos),
            'total_sentences': total_sentences,
            'vocabulary_size': len(self.vocabulary),
            'avg_sentence_length': np.mean(sentence_lengths),
            'max_sentence_length': max(sentence_lengths),
            'min_sentence_length': min(sentence_lengths),
            'sentence_types': list(set(s.sentence_type for v in self.videos for s in v.sentences))
        }

class ASLLRPDatasetConverter:
    """Converts ASLLRP data to training format."""
    
    def __init__(self, asllrp_data: List[ASLLRPVideo]):
        self.asllrp_data = asllrp_data
        self.vocabulary = self._build_vocabulary()
        self.gloss_to_idx = {gloss: idx for idx, gloss in enumerate(self.vocabulary)}
        self.idx_to_gloss = {idx: gloss for gloss, idx in self.gloss_to_idx.items()}
    
    def _build_vocabulary(self) -> List[str]:
        """Build vocabulary from all sentences."""
        vocabulary = set()
        for video in self.asllrp_data:
            for sentence in video.sentences:
                vocabulary.update(sentence.glosses)
        
        # Add special tokens
        vocabulary.add('<PAD>')
        vocabulary.add('<SOS>')  # Start of sentence
        vocabulary.add('<EOS>')  # End of sentence
        vocabulary.add('<UNK>')  # Unknown
        
        return sorted(list(vocabulary))
    
    def convert_to_training_format(self) -> List[Dict[str, any]]:
        """Convert ASLLRP data to training format."""
        training_data = []
        
        for video in self.asllrp_data:
            for sentence in video.sentences:
                # Convert glosses to indices
                gloss_indices = [self.gloss_to_idx.get(gloss, self.gloss_to_idx['<UNK>']) 
                               for gloss in sentence.glosses]
                
                # Add special tokens
                input_sequence = [self.gloss_to_idx['<SOS>']] + gloss_indices
                target_sequence = gloss_indices + [self.gloss_to_idx['<EOS>']]
                
                training_data.append({
                    'video_path': video.video_path,
                    'video_id': video.video_id,
                    'sentence_id': sentence.sentence_id,
                    'input_sequence': input_sequence,
                    'target_sequence': target_sequence,
                    'timestamps': sentence.timestamps,
                    'signer_id': sentence.signer_id,
                    'sentence_type': sentence.sentence_type,
                    'metadata': sentence.metadata
                })
        
        return training_data
    
    def save_training_data(self, output_path: str):
        """Save converted training data to file."""
        training_data = self.convert_to_training_format()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocabulary': self.vocabulary,
                'gloss_to_idx': self.gloss_to_idx,
                'idx_to_gloss': self.idx_to_gloss,
                'training_data': training_data,
                'statistics': {
                    'total_sentences': len(training_data),
                    'vocabulary_size': len(self.vocabulary),
                    'max_sequence_length': max(len(item['input_sequence']) for item in training_data)
                }
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved training data to {output_path}")

# Example usage
if __name__ == "__main__":
    # Load ASLLRP dataset
    loader = ASLLRPDataLoader("/path/to/asllrp/data")
    videos = loader.load_dataset()
    
    # Print statistics
    stats = loader.get_sentence_statistics()
    print("ASLLRP Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Convert to training format
    converter = ASLLRPDatasetConverter(videos)
    converter.save_training_data("asllrp_training_data.json")
    
    print(f"Vocabulary size: {len(converter.vocabulary)}")
    print(f"First 10 vocabulary items: {converter.vocabulary[:10]}")
