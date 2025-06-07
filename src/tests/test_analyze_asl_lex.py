"""
Tests for ASL Lex data analysis functionality.
"""
import pytest
import os
from src.data.analyze_asl_lex import analyze_asl_lex_data

def test_analyze_asl_lex_data_runs(data_dirs):
    """Test that ASL Lex analysis runs without errors."""
    try:
        analyze_asl_lex_data(data_dirs['asl_lex'])
        assert os.path.exists('reports')
        # Check that at least one plot was created
        files = os.listdir('reports')
        assert any(f.endswith('.png') for f in files)
    except Exception as e:
        pytest.fail(f"ASL Lex analysis failed: {str(e)}")

def test_analyze_asl_lex_invalid_path():
    """Test ASL Lex analysis with invalid path."""
    with pytest.raises(Exception):
        analyze_asl_lex_data('nonexistent/path')

@pytest.mark.integration
def test_analyze_asl_lex_output_files(data_dirs):
    """Test that ASL Lex analysis creates expected output files."""
    try:
        analyze_asl_lex_data(data_dirs['asl_lex'])
        
        # Check for specific output files
        expected_files = [
            'handshape_distribution.png',
            'sign_frequency_hist.png',
            'confusion_matrix.png'
        ]
        
        for file in expected_files:
            file_path = os.path.join('reports', file)
            assert os.path.exists(file_path), f"Expected file {file} not found"
    except Exception as e:
        pytest.fail(f"ASL Lex analysis output verification failed: {str(e)}") 