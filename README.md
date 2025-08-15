# 📚 Examlyze - AI-Powered Exam Analysis Platform

A revolutionary platform that uses artificial intelligence to analyze exam papers, extract topics, and predict trends to optimize student study strategies.

## ✨ Features

- **🚀 Modern UI**: Cutting-edge glassmorphism design with smooth animations
- **🤖 AI Analysis**: Advanced NLP-powered topic extraction and trend prediction
- **📊 Data Visualization**: Interactive charts and visual analytics
- **🎯 Smart Predictions**: ML-based trend forecasting for exam preparation
- **📱 Responsive Design**: Works perfectly on all devices
- **🌓 Dark/Light Themes**: Beautiful theme switching
- **💾 Export Options**: Detailed reports in multiple formats

## 🛠️ Technology Stack

### Frontend
- **HTML5, CSS3, JavaScript ES6+**
- **GSAP** - Professional animations
- **Chart.js** - Data visualization
- **Glassmorphism UI** - Modern design aesthetics
- **CSS Grid & Flexbox** - Responsive layouts

### Backend
- **Python Flask** - Web framework
- **scikit-learn** - Machine learning
- **NLTK** - Natural language processing
- **PyMuPDF** - PDF text extraction
- **python-docx** - Word document processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser
- (Optional) Node.js for development server

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/examlyze.git
cd examlyze
```

### 2. Setup Backend
```bash
# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

The backend will start on `http://localhost:5000`

### 3. Setup Frontend
```bash
# Option 1: Simple Python server
cd frontend
python -m http.server 8080

# Option 2: Using Node.js (if installed)
cd frontend
npx serve . -p 8080

# Option 3: Just open index.html in your browser
# (Some features may be limited due to CORS)
```

The frontend will be available at `http://localhost:8080`

## 📁 Project Structure

```
examlyze/
├── frontend/
│   ├── index.html          # Main HTML file
│   ├── css/
│   │   └── styles.css      # All CSS styles
│   └── js/
│       └── app.js          # Frontend JavaScript
├── backend/
│   ├── app.py              # Flask application
│   ├── requirements.txt    # Python dependencies
│   └── uploads/            # File upload directory (auto-created)
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## 🔧 Configuration

### Backend Configuration
Edit `app.py` to modify:
- Upload file size limits
- Allowed file extensions
- CORS origins
- Database settings (for production)

### Frontend Configuration
Edit `js/app.js` to modify:
- API endpoints
- Animation settings
- Theme preferences
- Chart configurations

## 📖 API Documentation

### Endpoints

#### Health Check
```
GET /api/health
```
Returns server status and version info.

#### Upload File
```
POST /api/upload
Content-Type: multipart/form-data

Body: file (PDF, DOCX, or TXT)
```
Returns file ID for analysis.

#### Analyze File
```
POST /api/analyze
Content-Type: application/json

Body: {
    "file_id": "unique-file-id"
}
```
Returns analysis results with topics and predictions.

#### Batch Analysis
```
POST /api/batch-analyze
Content-Type: application/json

Body: {
    "file_ids": ["id1", "id2", "id3"]
}
```
Analyze multiple files at once.

#### Get Trending Topics
```
GET /api/topics/trending
```
Returns trending topics across all analyses.

## 🎨 Features Showcase

### Modern UI Components
- **Glassmorphism cards** with backdrop blur
- **3D hover effects** and smooth transitions
- **Particle animations** in background
- **Morphing buttons** with ripple effects
- **Advanced progress indicators**

### Analysis Features
- **TF-IDF topic extraction** with clustering
- **Keyword-based fallback** analysis
- **Confidence scoring** for topics
- **Trend predictions** with reasoning
- **Batch processing** capabilities

### User Experience
- **Drag & drop** file uploads
- **Real-time progress** tracking
- **Keyboard shortcuts** (Ctrl+U, Ctrl+E, Ctrl+D)
- **Export options** (PDF, JSON, Markdown)
- **Responsive design** for all devices

## 🔍 Usage Guide

### 1. Upload Files
- Drag and drop exam papers (PDF, DOCX, TXT)
- Or click "Choose Files" button
- Supports multiple file uploads

### 2. Analyze Documents
- Click "Analyze Document" on uploaded files
- Watch real-time progress tracking
- View extracted topics and predictions

### 3. Review Results
- Topics with confidence scores
- Trend predictions with reasoning
- Interactive data visualizations
- Export detailed reports

### 4. Export Results
- **Summary Report**: Markdown format with detailed analysis
- **Raw Data**: JSON format for further processing
- Use floating action buttons or keyboard shortcuts

## 🛡️ Security Notes

- File uploads are validated for type and size
- Uploaded files are stored securely with unique IDs
- No sensitive data is logged or stored permanently
- CORS is configured for development (adjust for production)

## 🚀 Deployment

### Development
- Frontend: Any static file server
- Backend: Flask development server
- Database: In-memory (for demo)

### Production Recommendations
- **Frontend**: Nginx, Apache, or CDN
- **Backend**: Gunicorn + Nginx
- **Database**: PostgreSQL or MongoDB
- **Caching**: Redis
- **Security**: HTTPS, proper CORS, file validation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Future Enhancements

- [ ] User authentication and profiles
- [ ] Historical analysis tracking
- [ ] Advanced ML model training
- [ ] Multi-language support
- [ ] Real-time collaboration
- [ ] Mobile app development
- [ ] Integration with LMS platforms
- [ ] Advanced analytics dashboard

## 🐛 Troubleshooting

### Common Issues

**CORS Errors:**
- Ensure backend is running on port 5000
- Check frontend is served from port 8080
- Verify CORS settings in Flask app

**File Upload Fails:**
- Check file size (max 16MB)
- Verify file format (PDF, DOCX, TXT only)
- Ensure uploads/ directory exists

**Analysis Errors:**
- Verify text extraction from uploaded files
- Check NLTK data is downloaded
- Ensure sufficient text content in documents

### Getting Help
- Check the console for detailed error messages
- Verify all dependencies are installed
- Ensure Python and pip versions are compatible

## 👥 Team

Created with ❤️ by the Examlyze team.

---

**Happy Analyzing! 🚀📊**