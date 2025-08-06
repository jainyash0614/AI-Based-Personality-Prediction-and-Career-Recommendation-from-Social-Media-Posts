# 🤖 AI-Based Personality Prediction and Career Recommendation from Social Media Posts

A comprehensive machine learning application that analyzes personality traits from social media posts and provides personalized career recommendations using the Big Five personality model (OCEAN).

## 🌟 Features

### 🔍 **Reddit Personality Analyzer**
- **Real-time Analysis**: Analyzes Reddit users' posts and comments
- **Sentiment Analysis**: Determines emotional tone using TextBlob
- **Personality Traits**: Measures Big Five traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- **Career Recommendations**: Suggests suitable career paths based on personality
- **Interactive Visualizations**: Charts and graphs for results display

### 🧠 **Advanced ML Personality Predictor**
- **Machine Learning Models**: Random Forest and Naive Bayes classifiers
- **Text Feature Extraction**: TF-IDF vectorization for text analysis
- **Model Training**: Pre-trained models on Big Five and myPersonality datasets
- **React Frontend**: Modern UI with Material-UI components
- **Real-time Predictions**: Instant personality analysis from text input

## 🛠️ Technology Stack

### **Backend**
- **Python 3.7+** - Core programming language
- **Flask** - Web framework
- **scikit-learn** - Machine learning library
- **NLTK** - Natural Language Processing
- **TextBlob** - Sentiment analysis
- **PRAW** - Reddit API wrapper
- **pandas & numpy** - Data manipulation

### **Frontend**
- **React 16.3.2** - UI framework
- **Material-UI** - Component library
- **Chart.js** - Data visualization
- **Webpack** - Module bundler
- **Babel** - JavaScript transpiler

### **Machine Learning**
- **Random Forest** - Regression and classification
- **TF-IDF Vectorizer** - Text feature extraction
- **Naive Bayes** - Text classification
- **Big Five Personality Model** - OCEAN traits

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- Node.js and npm (for React frontend)
- Reddit API credentials

### 1. Clone the Repository
```bash
git clone https://github.com/jainyash0614/AI-Based-Personality-Prediction-and-Career-Recommendation-from-Social-Media-Posts.git
cd AI-Based-Personality-Prediction-and-Career-Recommendation-from-Social-Media-Posts
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Reddit API Credentials
1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Fill in the details:
   - **Name**: Personality Analyzer
   - **App type**: Script
   - **Description**: Reddit personality analysis tool
   - **Redirect URI**: `http://localhost:8080`

4. Create `.env` file in the root directory:
```bash
cp env_template.txt .env
```

5. Edit `.env` with your credentials:
```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=PersonalityAnalyzer/1.0 (by /u/your_username)
```

### 4. Install Frontend Dependencies (Optional)
For the advanced personality predictor:
```bash
cd personality-prediction-from-text
npm install
npm run build
```

## 🚀 Usage

### Simple Reddit Analyzer
```bash
python app.py
```
Visit `http://localhost:5000` and enter a Reddit username to analyze their personality.

### Advanced Personality Predictor
```bash
cd personality-prediction-from-text
python app.py
```
Visit `http://localhost:5000` for the full-featured application.

## 📊 How It Works

### 1. **Data Collection**
- Fetches user's recent posts and comments from Reddit
- Analyzes text content for sentiment and personality indicators

### 2. **Text Processing**
- Tokenizes and cleans text data
- Removes stopwords and performs stemming
- Extracts features using TF-IDF vectorization

### 3. **Personality Analysis**
- **Openness**: Creativity, curiosity, and openness to new experiences
- **Conscientiousness**: Organization, planning, and goal-directed behavior
- **Extraversion**: Social engagement and outgoing nature
- **Agreeableness**: Cooperation, trust, and helpfulness
- **Neuroticism**: Emotional sensitivity and stress response

### 4. **Career Recommendations**
Based on personality traits, suggests suitable career paths:
- **High Openness**: Artists, designers, researchers, entrepreneurs
- **High Conscientiousness**: Accountants, engineers, project managers
- **High Extraversion**: Sales, PR, event planning, HR
- **High Agreeableness**: Healthcare, teaching, social work
- **Low Neuroticism**: Emergency services, military, aviation

## 📈 Features in Detail

### **Sentiment Analysis**
- Analyzes emotional tone of posts
- Provides sentiment scores (-1 to +1)
- Identifies positive, negative, or neutral communication patterns

### **Communication Style Analysis**
- Average words per post
- Community impact (upvote scores)
- Engagement patterns

### **Interactive Visualizations**
- Radar charts for personality traits
- Bar charts for sentiment distribution
- Career recommendation cards

### **Machine Learning Models**
- **Regression Models**: Predict continuous personality scores
- **Classification Models**: Categorize personality types
- **Ensemble Methods**: Combine multiple models for accuracy

## 🎯 Use Cases

### **Personal Development**
- Understand your own personality traits
- Identify areas for personal growth
- Find suitable career paths

### **HR and Recruitment**
- Analyze candidate personality from social media
- Match candidates to job requirements
- Team composition analysis

### **Research and Analytics**
- Social media personality research
- Behavioral pattern analysis
- Psychological studies

## 🔧 Configuration

### Environment Variables
```env
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_app_name/version
```

### Model Parameters
The machine learning models can be customized in `personality-prediction-from-text/model.py`:
- Number of estimators
- Feature selection methods
- Model hyperparameters

## 📁 Project Structure

```
├── app.py                          # Main Reddit analyzer
├── requirements.txt                 # Python dependencies
├── templates/                      # HTML templates
├── env_template.txt                # Environment variables template
├── personality-prediction-from-text/  # Advanced ML module
│   ├── app.py                     # Flask API
│   ├── model.py                   # ML models
│   ├── predict.py                 # Prediction logic
│   ├── data_prep.py              # Data preprocessing
│   ├── static/                    # React frontend
│   ├── data/                      # Training datasets
│   └── webpack.config.js          # Build configuration
└── README.md                      # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Reddit API** for social media data access
- **Big Five Personality Model** for psychological framework
- **scikit-learn** for machine learning capabilities
- **React and Material-UI** for frontend development

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/jainyash0614/AI-Based-Personality-Prediction-and-Career-Recommendation-from-Social-Media-Posts/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 🔮 Future Enhancements

- [ ] Support for Twitter and Facebook analysis
- [ ] Real-time personality tracking over time
- [ ] Advanced career matching algorithms
- [ ] Mobile application
- [ ] API endpoints for third-party integration
- [ ] Multi-language support
- [ ] Enhanced privacy controls

---

**⭐ Star this repository if you find it helpful!**

**🔗 Connect with me: [GitHub Profile](https://github.com/jainyash0614)** 