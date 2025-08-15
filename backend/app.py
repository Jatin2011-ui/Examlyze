"""
Examlyze Backend - Flask Application
AI-Powered Exam Paper Analysis Platform
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Flask imports
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# ML and NLP imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import nltk

# File processing imports
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Install with: pip install PyMuPDF")

try:
    from docx import Document
except ImportError:
    print("python-docx not found. Install with: pip install python-docx")

# Configuration
class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'examlyze-dev-key-2024'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
    
    # Create upload folder if it doesn't exist
    def __init__(self):
        Path(self.UPLOAD_FOLDER).mkdir(exist_ok=True)

config = Config()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)
CORS(app, origins=['http://localhost:8080', 'http://127.0.0.1:8080', 'file://'])

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data
def setup_nltk():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

setup_nltk()

# Utility functions
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def extract_text_from_file(file_path: str) -> str:
    """Extract text content from uploaded file"""
    try:
        file_extension = file_path.rsplit('.', 1)[1].lower()
        
        if file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        elif file_extension == 'pdf':
            try:
                doc = fitz.open(file_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except Exception as e:
                logger.error(f"PDF extraction error: {e}")
                return ""
        
        elif file_extension == 'docx':
            try:
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except Exception as e:
                logger.error(f"DOCX extraction error: {e}")
                return ""
                
    except Exception as e:
        logger.error(f"File extraction error: {e}")
        return ""
    
    return ""

# Text processing and analysis
class TextAnalyzer:
    """Advanced text analysis and topic extraction"""
    
    def __init__(self):
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.subjects_keywords = {
            'Mathematics': ['equation', 'calculus', 'algebra', 'geometry', 'derivative', 'integral', 'matrix', 'function'],
            'Physics': ['mechanics', 'thermodynamics', 'wave', 'quantum', 'energy', 'force', 'motion', 'electricity'],
            'Chemistry': ['organic', 'inorganic', 'reaction', 'molecule', 'bond', 'acid', 'base', 'element'],
            'Biology': ['cell', 'dna', 'genetics', 'evolution', 'organism', 'protein', 'enzyme', 'photosynthesis'],
            'Computer Science': ['algorithm', 'programming', 'data structure', 'database', 'software', 'network', 'security'],
            'Statistics': ['probability', 'distribution', 'hypothesis', 'regression', 'correlation', 'variance', 'mean', 'median'],
            'Machine Learning': ['neural network', 'classification', 'regression', 'training', 'model', 'feature', 'prediction'],
            'Data Science': ['analysis', 'visualization', 'mining', 'processing', 'analytics', 'insights', 'trends']
        }
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis"""
        from nltk.tokenize import word_tokenize
        import re
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return processed_tokens
    
    def extract_topics_tfidf(self, text: str, n_topics: int = 8) -> List[Dict]:
        """Extract topics using TF-IDF and clustering"""
        try:
            # Preprocess text
            processed_text = ' '.join(self.preprocess_text(text))
            
            if len(processed_text.split()) < 10:
                return self.extract_topics_keyword_based(text)
            
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            # Split text into sentences for clustering
            sentences = text.split('.')
            if len(sentences) < 3:
                return self.extract_topics_keyword_based(text)
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # K-means clustering
            n_clusters = min(n_topics, len(sentences), tfidf_matrix.shape[0])
            if n_clusters < 2:
                return self.extract_topics_keyword_based(text)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Extract topics from clusters
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for i in range(n_clusters):
                # Get top terms for this cluster
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-5:][::-1]
                top_terms = [feature_names[idx] for idx in top_indices]
                
                # Generate topic name from top terms
                topic_name = self.generate_topic_name(top_terms)
                confidence = float(np.max(cluster_center)) * 100
                
                topics.append({
                    'name': topic_name,
                    'confidence': min(95, max(70, confidence)),
                    'keywords': top_terms[:4],
                    'frequency': int(np.sum(clusters == i))
                })
            
            return sorted(topics, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"TF-IDF topic extraction error: {e}")
            return self.extract_topics_keyword_based(text)
    
    def extract_topics_keyword_based(self, text: str) -> List[Dict]:
        """Fallback: Extract topics based on keyword matching"""
        topics = []
        text_lower = text.lower()
        
        for subject, keywords in self.subjects_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                confidence = min(95, 60 + (matches * 5))
                topics.append({
                    'name': subject,
                    'confidence': confidence,
                    'keywords': [kw for kw in keywords if kw in text_lower][:4],
                    'frequency': matches
                })
        
        # Add some generic topics if none found
        if not topics:
            topics = [
                {
                    'name': 'General Knowledge',
                    'confidence': 75,
                    'keywords': ['knowledge', 'information', 'study', 'learn'],
                    'frequency': 5
                },
                {
                    'name': 'Problem Solving',
                    'confidence': 70,
                    'keywords': ['problem', 'solution', 'solve', 'answer'],
                    'frequency': 3
                }
            ]
        
        return sorted(topics, key=lambda x: x['confidence'], reverse=True)[:8]
    
    def generate_topic_name(self, terms: List[str]) -> str:
        """Generate a meaningful topic name from terms"""
        # Check if terms match known subjects
        for subject, keywords in self.subjects_keywords.items():
            if any(term in keywords for term in terms):
                return subject
        
        # Generate name from top terms
        if len(terms) > 0:
            return ' '.join(terms[:2]).title()
        
        return "General Topic"
    
    def generate_predictions(self, topics: List[Dict]) -> List[Dict]:
        """Generate trend predictions based on topics"""
        predictions = []
        priorities = ['high', 'medium', 'low']
        
        for topic in topics[:5]:  # Top 5 topics
            # Simulate prediction logic
            base_prob = topic['confidence']
            trend_factor = np.random.uniform(0.8, 1.2)
            probability = min(95, int(base_prob * trend_factor))
            
            priority = (
                'high' if probability > 85 else
                'medium' if probability > 75 else
                'low'
            )
            
            reasoning = self.generate_reasoning(topic['name'], probability)
            
            predictions.append({
                'topic': topic['name'],
                'probability': probability,
                'priority': priority,
                'reasoning': reasoning,
                'trend': 'increasing' if trend_factor > 1.0 else 'stable'
            })
        
        return sorted(predictions, key=lambda x: x['probability'], reverse=True)
    
    def generate_reasoning(self, topic: str, probability: int) -> str:
        """Generate reasoning for predictions"""
        high_freq_reasons = [
            f"{topic} appears frequently in recent exam patterns",
            f"Current academic curriculum emphasizes {topic}",
            f"{topic} is a fundamental concept often tested"
        ]
        
        medium_freq_reasons = [
            f"{topic} shows moderate importance in exam trends",
            f"Historical data suggests {topic} has steady relevance",
            f"{topic} concepts are regularly featured in assessments"
        ]
        
        low_freq_reasons = [
            f"{topic} may appear as supplementary content",
            f"Basic {topic} knowledge is occasionally tested",
            f"{topic} serves as foundational material"
        ]
        
        if probability > 85:
            return np.random.choice(high_freq_reasons)
        elif probability > 75:
            return np.random.choice(medium_freq_reasons)
        else:
            return np.random.choice(low_freq_reasons)

# Initialize analyzer
analyzer = TextAnalyzer()

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(config.UPLOAD_FOLDER, unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        logger.info(f"File uploaded: {filename} ({file_size} bytes)")
        
        return jsonify({
            'file_id': unique_filename,
            'filename': filename,
            'size': file_size,
            'status': 'uploaded'
        })
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large'}), 413
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    """Analyze uploaded file"""
    try:
        data = request.json
        file_id = data.get('file_id')
        
        if not file_id:
            return jsonify({'error': 'File ID required'}), 400
        
        file_path = os.path.join(config.UPLOAD_FOLDER, file_id)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Extract text
        text_content = extract_text_from_file(file_path)
        
        if not text_content or len(text_content.strip()) < 50:
            return jsonify({'error': 'Could not extract sufficient text from file'}), 400
        
        # Analyze text
        topics = analyzer.extract_topics_tfidf(text_content)
        predictions = analyzer.generate_predictions(topics)
        
        # Calculate metadata
        word_count = len(text_content.split())
        reading_time = max(1, word_count // 200)  # ~200 words per minute
        
        analysis_result = {
            'analysis_id': str(uuid.uuid4()),
            'file_id': file_id,
            'topics': topics,
            'predictions': predictions,
            'metadata': {
                'word_count': word_count,
                'reading_time': reading_time,
                'analysis_timestamp': datetime.now().isoformat(),
                'text_length': len(text_content)
            }
        }
        
        logger.info(f"Analysis completed for file: {file_id}")
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple files"""
    try:
        data = request.json
        file_ids = data.get('file_ids', [])
        
        if not file_ids:
            return jsonify({'error': 'File IDs required'}), 400
        
        results = []
        
        for file_id in file_ids:
            file_path = os.path.join(config.UPLOAD_FOLDER, file_id)
            
            if os.path.exists(file_path):
                text_content = extract_text_from_file(file_path)
                
                if text_content and len(text_content.strip()) >= 50:
                    topics = analyzer.extract_topics_tfidf(text_content)
                    predictions = analyzer.generate_predictions(topics)
                    
                    word_count = len(text_content.split())
                    reading_time = max(1, word_count // 200)
                    
                    results.append({
                        'file_id': file_id,
                        'topics': topics,
                        'predictions': predictions,
                        'metadata': {
                            'word_count': word_count,
                            'reading_time': reading_time,
                            'analysis_timestamp': datetime.now().isoformat()
                        }
                    })
        
        return jsonify({
            'batch_id': str(uuid.uuid4()),
            'results': results,
            'total_files': len(results)
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({'error': 'Batch analysis failed'}), 500

@app.route('/api/topics/trending', methods=['GET'])
def get_trending_topics():
    """Get trending topics across all analyses"""
    # This would typically query a database
    # For demo, return sample trending topics
    trending_topics = [
        {'name': 'Machine Learning', 'trend_score': 95, 'frequency': 45},
        {'name': 'Data Structures', 'trend_score': 88, 'frequency': 38},
        {'name': 'Algorithms', 'trend_score': 82, 'frequency': 33},
        {'name': 'Statistics', 'trend_score': 79, 'frequency': 29},
        {'name': 'Linear Algebra', 'trend_score': 76, 'frequency': 25}
    ]
    
    return jsonify({
        'trending_topics': trending_topics,
        'last_updated': datetime.now().isoformat()
    })

@app.route('/')
def serve_frontend():
    """Serve the frontend (for development)"""
    return """
    <h1>Examlyze Backend API</h1>
    <p>The Flask backend is running successfully!</p>
    <h3>Available Endpoints:</h3>
    <ul>
        <li><strong>GET</strong> /api/health - Health check</li>
        <li><strong>POST</strong> /api/upload - Upload file</li>
        <li><strong>POST</strong> /api/analyze - Analyze file</li>
        <li><strong>POST</strong> /api/batch-analyze - Batch analyze files</li>
        <li><strong>GET</strong> /api/topics/trending - Get trending topics</li>
    </ul>
    <p>Frontend should be served separately on port 8080</p>
    """

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(RequestEntityTooLarge)
def too_large_error(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    logger.info("Starting Examlyze Backend Server...")
    logger.info(f"Upload folder: {config.UPLOAD_FOLDER}")
    logger.info("Server starting on http://localhost:5000")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,  # Set to False in production
        threaded=True
    )