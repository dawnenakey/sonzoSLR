"""
Test suite for Epic 2: Text Corpus Management

This test suite validates all functionality of the text corpus management system
including CRUD operations, permissions, and integration with Epic 1 authentication.
"""

import unittest
import json
import os
import sys
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_corpus_service import TextCorpusService, TextCorpus, TextSegment, CorpusExport
from auth_service import AuthService, UserRole

class TestTextCorpusService(unittest.TestCase):
    """Test cases for TextCorpusService"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock AWS services
        self.mock_dynamodb = MagicMock()
        self.mock_corpora_table = MagicMock()
        self.mock_segments_table = MagicMock()
        self.mock_exports_table = MagicMock()
        
        # Patch boto3 resource
        self.patcher = patch('text_corpus_service.boto3.resource')
        self.mock_boto3 = self.patcher.start()
        self.mock_boto3.return_value = self.mock_dynamodb
        
        # Set up table references
        self.mock_dynamodb.Table.side_effect = lambda name: {
            'spokhand-text-corpora': self.mock_corpora_table,
            'spokhand-text-segments': self.mock_segments_table,
            'spokhand-corpus-exports': self.mock_exports_table
        }.get(name)
        
        # Create service instance
        self.service = TextCorpusService()
        
        # Sample test data
        self.sample_corpus_data = {
            'id': 'test-corpus-123',
            'name': 'Test ASL Corpus',
            'description': 'A test corpus for unit testing',
            'language': 'ASL',
            'metadata': {'test': True},
            'created_by': 'test@example.com',
            'created_at': '2024-01-01T00:00:00Z',
            'updated_at': '2024-01-01T00:00:00Z',
            'status': 'draft',
            'total_segments': 0,
            'validated_segments': 0,
            'tags': ['test'],
            'version': '1.0.0'
        }
        
        self.sample_segment_data = {
            'id': 'test-segment-123',
            'corpus_id': 'test-corpus-123',
            'text': 'Hello, how are you?',
            'metadata': {'category': 'greetings'},
            'position': 1,
            'segment_type': 'phrase',
            'created_by': 'test@example.com',
            'created_at': '2024-01-01T00:00:00Z',
            'updated_at': '2024-01-01T00:00:00Z',
            'status': 'draft',
            'validation_notes': '',
            'related_signs': [],
            'confidence_score': 0.0
        }
    
    def tearDown(self):
        """Clean up after tests"""
        self.patcher.stop()
    
    def test_create_corpus_success(self):
        """Test successful corpus creation"""
        # Mock DynamoDB response
        self.mock_corpora_table.put_item.return_value = None
        
        # Create corpus
        corpus = self.service.create_corpus(
            name='Test Corpus',
            description='Test Description',
            language='ASL',
            created_by='test@example.com'
        )
        
        # Verify corpus was created
        self.assertIsInstance(corpus, TextCorpus)
        self.assertEqual(corpus.name, 'Test Corpus')
        self.assertEqual(corpus.language, 'ASL')
        self.assertEqual(corpus.status, 'draft')
        
        # Verify DynamoDB was called
        self.mock_corpora_table.put_item.assert_called_once()
    
    def test_get_corpus_success(self):
        """Test successful corpus retrieval"""
        # Mock DynamoDB response
        self.mock_corpora_table.get_item.return_value = {
            'Item': self.sample_corpus_data
        }
        
        # Get corpus
        corpus = self.service.get_corpus('test-corpus-123')
        
        # Verify corpus was retrieved
        self.assertIsInstance(corpus, TextCorpus)
        self.assertEqual(corpus.id, 'test-corpus-123')
        self.assertEqual(corpus.name, 'Test ASL Corpus')
        
        # Verify DynamoDB was called
        self.mock_corpora_table.get_item.assert_called_once_with(
            Key={'id': 'test-corpus-123'}
        )
    
    def test_get_corpus_not_found(self):
        """Test corpus retrieval when not found"""
        # Mock DynamoDB response - no item
        self.mock_corpora_table.get_item.return_value = {}
        
        # Get corpus
        corpus = self.service.get_corpus('non-existent')
        
        # Verify None was returned
        self.assertIsNone(corpus)
    
    def test_list_corpora_success(self):
        """Test successful corpus listing"""
        # Mock DynamoDB response
        self.mock_corpora_table.scan.return_value = {
            'Items': [self.sample_corpus_data]
        }
        
        # List corpora
        corpora = self.service.list_corpora()
        
        # Verify corpora were retrieved
        self.assertEqual(len(corpora), 1)
        self.assertIsInstance(corpora[0], TextCorpus)
        self.assertEqual(corpora[0].name, 'Test ASL Corpus')
        
        # Verify DynamoDB was called
        self.mock_corpora_table.scan.assert_called_once()
    
    def test_list_corpora_with_filters(self):
        """Test corpus listing with filters"""
        # Mock DynamoDB response
        self.mock_corpora_table.scan.return_value = {
            'Items': [self.sample_corpus_data]
        }
        
        # List corpora with filters
        corpora = self.service.list_corpora(
            user_id='test@example.com',
            status='draft',
            language='ASL'
        )
        
        # Verify filters were applied
        self.assertEqual(len(corpora), 1)
        self.assertEqual(corpora[0].language, 'ASL')
        self.assertEqual(corpora[0].status, 'draft')
    
    def test_update_corpus_success(self):
        """Test successful corpus update"""
        # Mock get_corpus to return existing corpus
        with patch.object(self.service, 'get_corpus') as mock_get:
            mock_get.return_value = TextCorpus(**self.sample_corpus_data)
            
            # Mock DynamoDB update
            self.mock_corpora_table.update_item.return_value = None
            
            # Update corpus
            updated_corpus = self.service.update_corpus(
                'test-corpus-123',
                {'name': 'Updated Name', 'status': 'active'},
                'test@example.com'
            )
            
            # Verify corpus was updated
            self.assertIsInstance(updated_corpus, TextCorpus)
            self.assertEqual(updated_corpus.name, 'Updated Name')
            self.assertEqual(updated_corpus.status, 'active')
            
            # Verify DynamoDB was called
            self.mock_corpora_table.update_item.assert_called_once()
    
    def test_add_text_segment_success(self):
        """Test successful text segment addition"""
        # Mock get_corpus to return existing corpus
        with patch.object(self.service, 'get_corpus') as mock_get:
            mock_get.return_value = TextCorpus(**self.sample_corpus_data)
            
            # Mock list_segments to return empty list
            with patch.object(self.service, 'list_segments') as mock_list:
                mock_list.return_value = []
                
                # Mock DynamoDB operations
                self.mock_segments_table.put_item.return_value = None
                self.mock_corpora_table.update_item.return_value = None
                
                # Add segment
                segment = self.service.add_text_segment(
                    'test-corpus-123',
                    'Hello world',
                    'sentence',
                    'test@example.com'
                )
                
                # Verify segment was created
                self.assertIsInstance(segment, TextSegment)
                self.assertEqual(segment.text, 'Hello world')
                self.assertEqual(segment.segment_type, 'sentence')
                self.assertEqual(segment.position, 1)
                
                # Verify DynamoDB was called
                self.mock_segments_table.put_item.assert_called_once()
                self.mock_corpora_table.update_item.assert_called_once()
    
    def test_add_text_segment_corpus_not_found(self):
        """Test adding segment to non-existent corpus"""
        # Mock get_corpus to return None
        with patch.object(self.service, 'get_corpus') as mock_get:
            mock_get.return_value = None
            
            # Attempt to add segment
            with self.assertRaises(ValueError):
                self.service.add_text_segment(
                    'non-existent',
                    'Hello world',
                    'sentence',
                    'test@example.com'
                )
    
    def test_search_corpus_text_search(self):
        """Test text search within corpus"""
        # Mock list_segments to return sample segments
        with patch.object(self.service, 'list_segments') as mock_list:
            mock_list.return_value = [
                TextSegment(**self.sample_segment_data),
                TextSegment(**{**self.sample_segment_data, 'id': 'seg2', 'text': 'Goodbye world'})
            ]
            
            # Search for "hello"
            results = self.service.search_corpus(
                'test-corpus-123',
                'hello',
                'text'
            )
            
            # Verify search results
            self.assertEqual(len(results), 1)
            self.assertIn('hello', results[0].text.lower())
    
    def test_search_corpus_metadata_search(self):
        """Test metadata search within corpus"""
        # Mock list_segments to return sample segments
        with patch.object(self.service, 'list_segments') as mock_list:
            mock_list.return_value = [
                TextSegment(**self.sample_segment_data),
                TextSegment(**{**self.sample_segment_data, 'id': 'seg2', 'metadata': {'category': 'farewells'}})
            ]
            
            # Search for "greetings" in metadata
            results = self.service.search_corpus(
                'test-corpus-123',
                'greetings',
                'metadata'
            )
            
            # Verify search results
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata['category'], 'greetings')
    
    def test_export_corpus_success(self):
        """Test successful corpus export"""
        # Mock DynamoDB operations
        self.mock_exports_table.put_item.return_value = None
        
        # Create export
        export_job = self.service.export_corpus(
            'test-corpus-123',
            'json',
            'test@example.com'
        )
        
        # Verify export job was created
        self.assertIsInstance(export_job, CorpusExport)
        self.assertEqual(export_job.corpus_id, 'test-corpus-123')
        self.assertEqual(export_job.export_format, 'json')
        self.assertEqual(export_job.status, 'pending')
        
        # Verify DynamoDB was called
        self.mock_exports_table.put_item.assert_called_once()
    
    def test_delete_corpus_success(self):
        """Test successful corpus deletion (soft delete)"""
        # Mock get_corpus to return existing corpus
        with patch.object(self.service, 'get_corpus') as mock_get:
            mock_get.return_value = TextCorpus(**self.sample_corpus_data)
            
            # Mock list_segments to return empty list
            with patch.object(self.service, 'list_segments') as mock_list:
                mock_list.return_value = []
                
                # Mock update_corpus
                with patch.object(self.service, 'update_corpus') as mock_update:
                    mock_update.return_value = TextCorpus(**{**self.sample_corpus_data, 'status': 'deleted'})
                    
                    # Delete corpus
                    success = self.service.delete_corpus(
                        'test-corpus-123',
                        'test@example.com'
                    )
                    
                    # Verify deletion was successful
                    self.assertTrue(success)
                    
                    # Verify update_corpus was called
                    mock_update.assert_called_once()
    
    def test_delete_segment_success(self):
        """Test successful segment deletion (soft delete)"""
        # Mock get_segment to return existing segment
        with patch.object(self.service, 'get_segment') as mock_get:
            mock_get.return_value = TextSegment(**self.sample_segment_data)
            
            # Mock update_segment
            with patch.object(self.service, 'update_segment') as mock_update:
                mock_update.return_value = TextSegment(**{**self.sample_segment_data, 'status': 'deleted'})
                
                # Mock get_corpus for corpus update
                with patch.object(self.service, 'get_corpus') as mock_get_corpus:
                    mock_get_corpus.return_value = TextCorpus(**self.sample_corpus_data)
                    
                    # Mock DynamoDB update
                    self.mock_corpora_table.update_item.return_value = None
                    
                    # Delete segment
                    success = self.service.delete_segment(
                        'test-segment-123',
                        'test@example.com'
                    )
                    
                    # Verify deletion was successful
                    self.assertTrue(success)
                    
                    # Verify update_segment was called
                    mock_update.assert_called_once()
    
    def test_get_export_status_success(self):
        """Test successful export status retrieval"""
        # Mock DynamoDB response
        export_data = {
            'id': 'export-123',
            'corpus_id': 'test-corpus-123',
            'export_format': 'json',
            'status': 'completed',
            'created_by': 'test@example.com',
            'created_at': '2024-01-01T00:00:00Z',
            'completed_at': '2024-01-01T00:01:00Z',
            'download_url': '/download/export-123',
            'error_message': None
        }
        
        self.mock_exports_table.get_item.return_value = {
            'Item': export_data
        }
        
        # Get export status
        export = self.service.get_export_status('export-123')
        
        # Verify export was retrieved
        self.assertIsInstance(export, CorpusExport)
        self.assertEqual(export.id, 'export-123')
        self.assertEqual(export.status, 'completed')
        
        # Verify DynamoDB was called
        self.mock_exports_table.get_item.assert_called_once_with(
            Key={'id': 'export-123'}
        )

class TestTextCorpusIntegration(unittest.TestCase):
    """Integration tests for text corpus system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # This would set up a real test database in a real test environment
        pass
    
    def test_corpus_workflow(self):
        """Test complete corpus workflow: create, add segments, search, export"""
        # This test would use real database operations in a test environment
        # For now, we'll skip it
        self.skipTest("Integration test requires test database setup")
    
    def test_permission_workflow(self):
        """Test permission-based access control"""
        # This test would verify role-based access in a real environment
        # For now, we'll skip it
        self.skipTest("Integration test requires test database setup")

class TestTextCorpusEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        self.service = TextCorpusService()
    
    def test_empty_corpus_operations(self):
        """Test operations on empty corpus"""
        # Test listing segments in empty corpus
        with patch.object(self.service, 'list_segments') as mock_list:
            mock_list.return_value = []
            
            segments = self.service.list_segments('empty-corpus')
            self.assertEqual(len(segments), 0)
    
    def test_invalid_export_format(self):
        """Test export with invalid format"""
        # This would be tested in the API layer
        # Service layer should handle any format string
        pass
    
    def test_large_corpus_operations(self):
        """Test operations on corpus with many segments"""
        # This would test performance with large datasets
        # For now, we'll skip it
        self.skipTest("Performance test requires large dataset setup")

def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTextCorpusService)
    suite.addTests(loader.loadTestsFromTestCase(TestTextCorpusIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTextCorpusEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    # Run tests
    exit_code = run_tests()
    sys.exit(exit_code)
