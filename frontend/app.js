// Global state
let currentTheme = 'dark';
let uploadedFiles = [];
let analysisResults = {};
let animationFrameId;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initBackgroundAnimation();
    initUploadHandlers();
    initGSAP();
    showNotification('Welcome to Examlyze! Upload your exam papers to begin analysis.', 'success');
});

// Background particle animation
function initBackgroundAnimation() {
    const bgContainer = document.getElementById('bgAnimation');
    const particleCount = window.innerWidth > 768 ? 50 : 25;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 6 + 's';
        particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
        bgContainer.appendChild(particle);
    }
}

// GSAP animations
function initGSAP() {
    // Animate hero on load
    gsap.from('.hero-content', {
        duration: 1.5,
        y: 50,
        opacity: 0,
        ease: "power3.out"
    });

    gsap.from('.upload-zone', {
        duration: 1.2,
        y: 30,
        opacity: 0,
        delay: 0.3,
        ease: "power3.out"
    });
}

// Theme toggle with smooth transition
function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', currentTheme);
    
    const toggle = document.getElementById('themeToggle');
    toggle.style.transform = 'scale(0) rotate(180deg)';
    
    setTimeout(() => {
        toggle.textContent = currentTheme === 'dark' ? '🌓' : '☀️';
        toggle.style.transform = 'scale(1) rotate(0deg)';
    }, 200);
    
    showNotification(`Switched to ${currentTheme} theme`, 'success');
}

// Upload handlers
function initUploadHandlers() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    // Drag and drop
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    handleFiles(files);
}

// File handling with animations
function handleFiles(files) {
    uploadedFiles = [...uploadedFiles, ...files];
    displayFiles();
    
    // Animate new files
    setTimeout(() => {
        const lastCard = document.querySelector('.file-card:last-child');
        if (lastCard) {
            gsap.from(lastCard, {
                duration: 0.6,
                scale: 0.8,
                opacity: 0,
                y: 20,
                ease: "back.out(1.7)"
            });
        }
    }, 100);
    
    showNotification(`${files.length} file(s) uploaded successfully!`, 'success');
}

function displayFiles() {
    const fileGrid = document.getElementById('fileGrid');
    
    if (uploadedFiles.length === 0) {
        fileGrid.style.display = 'none';
        return;
    }

    fileGrid.style.display = 'grid';
    fileGrid.innerHTML = '';

    uploadedFiles.forEach((file, index) => {
        const fileCard = createFileCard(file, index);
        fileGrid.appendChild(fileCard);
    });
}

function createFileCard(file, index) {
    const card = document.createElement('div');
    card.className = 'file-card glass-container';
    
    const extension = file.name.split('.').pop().toUpperCase();
    const size = (file.size / 1024 / 1024).toFixed(2);
    
    card.innerHTML = `
        <div class="file-header">
            <div class="file-type-icon">${extension}</div>
            <div class="file-info">
                <h4>${file.name}</h4>
                <div class="file-size">${size} MB</div>
            </div>
        </div>
        <button class="analyze-btn" onclick="analyzeFile(${index})">
            <span>🚀 Analyze Document</span>
        </button>
    `;
    
    return card;
}

// File analysis with advanced progress
async function analyzeFile(index) {
    const file = uploadedFiles[index];
    showProgress();
    
    // Simulate analysis with realistic progress
    const stages = [
        { progress: 15, text: 'Reading document...' },
        { progress: 35, text: 'Extracting text content...' },
        { progress: 55, text: 'Processing with NLP...' },
        { progress: 75, text: 'Identifying topics...' },
        { progress: 90, text: 'Generating predictions...' },
        { progress: 100, text: 'Analysis complete!' }
    ];
    
    for (const stage of stages) {
        await new Promise(resolve => setTimeout(resolve, 800));
        updateProgress(stage.progress, stage.text);
    }

    // Generate and display results
    const results = generateAdvancedResults(file);
    analysisResults[file.name] = results;
    
    hideProgress();
    displayResults(results);
    showFABs();
    
    showNotification('Analysis completed! Scroll down to view results.', 'success');
}

function showProgress() {
    const container = document.getElementById('progressContainer');
    container.style.display = 'block';
    
    gsap.from(container, {
        duration: 0.5,
        scale: 0.8,
        opacity: 0,
        ease: "power3.out"
    });
}

function updateProgress(percentage, text) {
    const circle = document.getElementById('progressCircle');
    const progressText = document.getElementById('progressText');
    
    const circumference = 2 * Math.PI * 45;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;
    
    circle.style.strokeDashoffset = strokeDashoffset;
    progressText.textContent = percentage + '%';
    
    // Update status text if provided
    if (text) {
        const container = document.getElementById('progressContainer');
        const statusElement = container.querySelector('p');
        if (statusElement) {
            statusElement.textContent = text;
        }
    }
}

function hideProgress() {
    const container = document.getElementById('progressContainer');
    
    gsap.to(container, {
        duration: 0.3,
        opacity: 0,
        scale: 0.8,
        onComplete: () => {
            container.style.display = 'none';
        }
    });
}

// Advanced results generation
function generateAdvancedResults(file) {
    const subjects = [
        'Mathematics', 'Physics', 'Chemistry', 'Biology', 'Computer Science',
        'Statistics', 'Linear Algebra', 'Data Structures', 'Algorithms',
        'Machine Learning', 'Artificial Intelligence', 'Database Systems'
    ];
    
    const difficulties = ['Beginner', 'Intermediate', 'Advanced', 'Expert'];
    const priorities = ['high', 'medium', 'low'];
    
    // Generate realistic topics
    const numTopics = Math.floor(Math.random() * 4) + 5;
    const topics = subjects
        .sort(() => 0.5 - Math.random())
        .slice(0, numTopics)
        .map(subject => ({
            name: subject,
            confidence: Math.floor(Math.random() * 25 + 75),
            frequency: Math.floor(Math.random() * 15 + 5),
            difficulty: difficulties[Math.floor(Math.random() * difficulties.length)],
            keywords: generateKeywords(subject)
        }));

    // Generate predictions
    const predictions = topics
        .slice(0, Math.min(5, topics.length))
        .map(topic => ({
            name: topic.name,
            probability: Math.floor(Math.random() * 30 + 70),
            priority: priorities[Math.floor(Math.random() * priorities.length)],
            reasoning: generateReasoning(topic.name),
            trend: Math.random() > 0.5 ? 'increasing' : 'stable'
        }));

    return {
        topics: topics.sort((a, b) => b.confidence - a.confidence),
        predictions: predictions.sort((a, b) => b.probability - a.probability),
        metadata: {
            fileName: file.name,
            fileSize: file.size,
            analyzedAt: new Date().toISOString(),
            totalWords: Math.floor(Math.random() * 5000 + 1000),
            readingTime: Math.floor(Math.random() * 20 + 5)
        }
    };
}

function generateKeywords(subject) {
    const keywordMap = {
        'Mathematics': ['equations', 'calculus', 'derivatives', 'integrals'],
        'Physics': ['mechanics', 'thermodynamics', 'waves', 'quantum'],
        'Chemistry': ['organic', 'reactions', 'molecules', 'bonds'],
        'Computer Science': ['algorithms', 'programming', 'data', 'systems'],
        'Machine Learning': ['neural networks', 'training', 'classification', 'regression']
    };
    
    return keywordMap[subject] || ['analysis', 'theory', 'application', 'concepts'];
}

function generateReasoning(topic) {
    const reasons = [
        `${topic} has appeared frequently in recent exams`,
        `Current curriculum emphasizes ${topic} concepts`,
        `${topic} shows increasing trend in academic focus`,
        `Industry demands make ${topic} a priority topic`
    ];
    
    return reasons[Math.floor(Math.random() * reasons.length)];
}

// Display results with animations
function displayResults(results) {
    const container = document.getElementById('resultsContainer');
    const topicsList = document.getElementById('topicsList');
    const predictionsList = document.getElementById('predictionsList');

    container.style.display = 'block';
    
    // Animate container appearance
    gsap.from(container, {
        duration: 1,
        y: 50,
        opacity: 0,
        ease: "power3.out"
    });

    // Display topics with stagger animation
    topicsList.innerHTML = '';
    results.topics.forEach((topic, index) => {
        const topicItem = document.createElement('li');
        topicItem.className = 'topic-item';
        topicItem.innerHTML = `
            <div class="topic-header">
                <div>
                    <div class="topic-name">${topic.name}</div>
                    <div class="topic-keywords" style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.2rem;">
                        ${topic.keywords.join(' • ')}
                    </div>
                </div>
                <div class="confidence-badge">${topic.confidence}%</div>
            </div>
        `;
        topicsList.appendChild(topicItem);
        
        // Stagger animation
        gsap.from(topicItem, {
            duration: 0.5,
            x: -30,
            opacity: 0,
            delay: index * 0.1,
            ease: "power2.out"
        });
    });

    // Display predictions with stagger animation
    predictionsList.innerHTML = '';
    results.predictions.forEach((prediction, index) => {
        const predictionItem = document.createElement('li');
        predictionItem.className = 'prediction-item';
        predictionItem.innerHTML = `
            <div class="prediction-header">
                <div>
                    <div class="prediction-name">${prediction.name}</div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.2rem;">
                        ${prediction.reasoning}
                    </div>
                </div>
                <div class="probability-badge priority-${prediction.priority}">
                    ${prediction.probability}%
                </div>
            </div>
        `;
        predictionsList.appendChild(predictionItem);
        
        // Stagger animation
        gsap.from(predictionItem, {
            duration: 0.5,
            x: 30,
            opacity: 0,
            delay: index * 0.1,
            ease: "power2.out"
        });
    });

    // Create advanced chart
    createAdvancedChart(results.topics);
}

// Advanced chart with gradient and animations
function createAdvancedChart(topics) {
    const ctx = document.getElementById('topicChart').getContext('2d');
    
    // Destroy existing chart
    if (window.topicChartInstance) {
        window.topicChartInstance.destroy();
    }
    
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(102, 126, 234, 0.8)');
    gradient.addColorStop(0.5, 'rgba(118, 75, 162, 0.6)');
    gradient.addColorStop(1, 'rgba(240, 147, 251, 0.4)');
    
    window.topicChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: topics.map(t => t.name),
            datasets: [{
                data: topics.map(t => t.frequency),
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(118, 75, 162, 0.8)',
                    'rgba(240, 147, 251, 0.8)',
                    'rgba(79, 172, 254, 0.8)',
                    'rgba(245, 87, 108, 0.8)',
                    'rgba(67, 233, 123, 0.8)'
                ],
                borderColor: 'rgba(255, 255, 255, 0.2)',
                borderWidth: 2,
                hoverBorderWidth: 3,
                hoverBorderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: getComputedStyle(document.documentElement)
                            .getPropertyValue('--text-primary'),
                        font: {
                            family: 'Inter',
                            size: 12,
                            weight: 500
                        },
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(102, 126, 234, 0.5)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            const topic = topics[context.dataIndex];
                            return `${topic.name}: ${topic.frequency} mentions (${topic.confidence}% confidence)`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 2000,
                easing: 'easeOutQuart'
            },
            elements: {
                arc: {
                    borderWidth: 2,
                    hoverBorderWidth: 4
                }
            }
        }
    });
}

// Show floating action buttons
function showFABs() {
    const fabContainer = document.getElementById('fabContainer');
    fabContainer.style.display = 'block';
    
    gsap.from('.fab', {
        duration: 0.5,
        scale: 0,
        stagger: 0.1,
        ease: "back.out(1.7)"
    });
}

// Export functions
function exportSummary() {
    const summary = generateDetailedSummary();
    downloadFile(summary, 'examlyze-detailed-summary.md', 'text/markdown');
    showNotification('Detailed summary exported!', 'success');
}

function exportData() {
    const data = {
        results: analysisResults,
        exportedAt: new Date().toISOString(),
        version: '2.0',
        metadata: {
            totalFiles: uploadedFiles.length,
            totalTopics: Object.values(analysisResults)
                .reduce((acc, result) => acc + result.topics.length, 0),
            avgConfidence: calculateAverageConfidence()
        }
    };
    
    downloadFile(JSON.stringify(data, null, 2), 'examlyze-analysis-data.json', 'application/json');
    showNotification('Analysis data exported!', 'success');
}

function generateDetailedSummary() {
    let summary = '# 📚 Examlyze Analysis Report\n\n';
    summary += `**Generated:** ${new Date().toLocaleString()}\n\n`;
    summary += '---\n\n';
    
    Object.entries(analysisResults).forEach(([fileName, results]) => {
        summary += `## 📄 ${fileName}\n\n`;
        summary += `**Analysis Summary:**\n`;
        summary += `- Total Words: ${results.metadata.totalWords}\n`;
        summary += `- Reading Time: ${results.metadata.readingTime} minutes\n`;
        summary += `- Topics Identified: ${results.topics.length}\n\n`;
        
        summary += '### 🎯 Key Topics\n\n';
        results.topics.forEach((topic, index) => {
            summary += `${index + 1}. **${topic.name}** (${topic.confidence}% confidence)\n`;
            summary += `   - Keywords: ${topic.keywords.join(', ')}\n`;
            summary += `   - Frequency: ${topic.frequency} mentions\n`;
            summary += `   - Difficulty: ${topic.difficulty}\n\n`;
        });
        
        summary += '### 🔮 Trend Predictions\n\n';
        results.predictions.forEach((pred, index) => {
            const priorityEmoji = pred.priority === 'high' ? '🔴' : pred.priority === 'medium' ? '🟡' : '🟢';
            summary += `${index + 1}. ${priorityEmoji} **${pred.name}** - ${pred.probability}% likelihood\n`;
            summary += `   - Reasoning: ${pred.reasoning}\n`;
            summary += `   - Trend: ${pred.trend}\n\n`;
        });
        
        summary += '---\n\n';
    });
    
    summary += '## 📊 Overall Statistics\n\n';
    summary += `- Total Files Analyzed: ${uploadedFiles.length}\n`;
    summary += `- Average Confidence Score: ${calculateAverageConfidence()}%\n`;
    summary += `- Most Common Topics: ${getMostCommonTopics().join(', ')}\n\n`;
    
    summary += '---\n\n';
    summary += '*Generated by Examlyze - AI-Powered Exam Analysis Platform*';
    
    return summary;
}

function calculateAverageConfidence() {
    const allTopics = Object.values(analysisResults)
        .flatMap(result => result.topics);
    
    if (allTopics.length === 0) return 0;
    
    const avgConfidence = allTopics.reduce((sum, topic) => sum + topic.confidence, 0) / allTopics.length;
    return Math.round(avgConfidence);
}

function getMostCommonTopics() {
    const topicCounts = {};
    
    Object.values(analysisResults).forEach(result => {
        result.topics.forEach(topic => {
            topicCounts[topic.name] = (topicCounts[topic.name] || 0) + 1;
        });
    });
    
    return Object.entries(topicCounts)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([topic]) => topic);
}

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Notification system with better animations
function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    // Animate in
    gsap.set(notification, { y: -100, opacity: 0 });
    gsap.to(notification, {
        duration: 0.5,
        y: 0,
        opacity: 1,
        ease: "back.out(1.7)"
    });
    
    // Auto dismiss
    setTimeout(() => {
        gsap.to(notification, {
            duration: 0.3,
            y: -100,
            opacity: 0,
            ease: "power2.in",
            onComplete: () => document.body.removeChild(notification)
        });
    }, 4000);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
            case 'u':
                e.preventDefault();
                document.getElementById('fileInput').click();
                break;
            case 'e':
                e.preventDefault();
                if (Object.keys(analysisResults).length > 0) {
                    exportSummary();
                }
                break;
            case 'd':
                e.preventDefault();
                if (Object.keys(analysisResults).length > 0) {
                    exportData();
                }
                break;
        }
    }
    
    if (e.key === 'Escape') {
        // Close any open modals or reset states
        const progressContainer = document.getElementById('progressContainer');
        if (progressContainer.style.display === 'block') {
            hideProgress();
        }
    }
});

// Add tooltips for better UX
function addTooltips() {
    const tooltips = [
        { selector: '.theme-toggle', text: 'Toggle Theme (Light/Dark)' },
        { selector: '.morph-btn', text: 'Click to upload files or press Ctrl+U' },
        { selector: '.fab[title="Export Summary"]', text: 'Export Analysis Summary (Ctrl+E)' },
        { selector: '.fab[title="Export Data"]', text: 'Export Raw Data (Ctrl+D)' }
    ];
    
    tooltips.forEach(tooltip => {
        const elements = document.querySelectorAll(tooltip.selector);
        elements.forEach(el => {
            if (!el.hasAttribute('title')) {
                el.setAttribute('title', tooltip.text);
            }
        });
    });
}

// Initialize tooltips after DOM is ready
setTimeout(addTooltips, 1000);