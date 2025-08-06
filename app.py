from flask import Flask, render_template, request
import praw
from textblob import TextBlob
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Reddit API client using environment variables
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_user_data(username):
    try:
        user = reddit.redditor(username)
        posts_and_comments = []
        
        # Get user's submissions
        for submission in user.submissions.new(limit=50):
            posts_and_comments.append({
                'text': submission.title + " " + (submission.selftext if submission.selftext else ""),
                'type': 'submission',
                'score': submission.score
            })
        
        # Get user's comments
        for comment in user.comments.new(limit=50):
            posts_and_comments.append({
                'text': comment.body,
                'type': 'comment',
                'score': comment.score
            })
        
        return analyze_personality(posts_and_comments)
    except Exception as e:
        return {"error": str(e)}

def analyze_personality(data):
    if not data:
        return {"error": "No data found for this user"}
    
    total_sentiment = 0
    word_count = 0
    total_score = 0
    
    personality_traits = {
        'openness': 0,
        'conscientiousness': 0,
        'extraversion': 0,
        'agreeableness': 0,
        'neuroticism': 0
    }
    
    # Career recommendations based on OCEAN traits
    career_recommendations = {
        'openness': [
            'Artist (Painter, Musician, Writer)',
            'Graphic Designer / UX Designer',
            'Research Scientist',
            'Entrepreneur / Startup Founder',
            'Psychologist / Philosopher',
            'Architect',
            'Filmmaker / Director',
            'Marketing Strategist'
        ],
        'conscientiousness': [
            'Accountant / Auditor',
            'Project Manager',
            'Software Engineer',
            'Surgeon',
            'Lawyer',
            'Civil Servant / Bureaucrat',
            'Engineer (any field)',
            'Data Analyst'
        ],
        'extraversion': [
            'Salesperson',
            'Public Relations Manager',
            'Actor / Performer',
            'Politician',
            'Event Planner',
            'Human Resources Professional',
            'News Anchor / TV Host',
            'Business Executive'
        ],
        'agreeableness': [
            'Nurse / Doctor',
            'Social Worker',
            'Teacher / Professor',
            'Therapist / Counselor',
            'Humanitarian / NGO Worker',
            'Customer Service Rep',
            'Veterinarian',
            'Childcare Worker'
        ],
        'neuroticism': {
            'low': [
                'Emergency Room Doctor',
                'Pilot',
                'Military Officer'
            ],
            'high': [
                'Writer / Poet',
                'Artist',
                'Researcher (in a calm setting)',
                'Librarian',
                'Archivist'
            ]
        }
    }
    
    # Additional keywords for better personality detection
    personality_keywords = {
        'openness': ['new', 'creative', 'idea', 'curious', 'art', 'music', 'culture', 'learn', 'explore', 'discover'],
        'conscientiousness': ['should', 'must', 'plan', 'organize', 'routine', 'responsible', 'duty', 'work', 'goal', 'achieve'],
        'extraversion': ['party', 'friends', 'social', 'together', 'group', 'fun', 'exciting', 'people', 'talk', 'share'],
        'agreeableness': ['help', 'thank', 'appreciate', 'kind', 'care', 'support', 'understand', 'sorry', 'please', 'welcome'],
        'neuroticism': ['worry', 'afraid', 'nervous', 'stress', 'anxiety', 'sad', 'depression', 'angry', 'upset', 'hate']
    }
    
    for item in data:
        sentiment = analyze_sentiment(item['text'])
        total_sentiment += sentiment
        words = len(word_tokenize(item['text']))
        word_count += words
        total_score += item['score']
        
        # Enhanced personality trait analysis
        text = item['text'].lower()
        for trait, keywords in personality_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            personality_traits[trait] += matches
    
    avg_sentiment = total_sentiment / len(data) if data else 0
    avg_words = word_count / len(data) if data else 0
    avg_score = total_score / len(data) if data else 0
    
    # Normalize personality traits
    max_trait = max(personality_traits.values()) if any(personality_traits.values()) else 1
    for trait in personality_traits:
        personality_traits[trait] = (personality_traits[trait] / max_trait) * 100
    
    # Get career recommendations based on highest traits
    sorted_traits = sorted(personality_traits.items(), key=lambda x: x[1], reverse=True)
    top_traits = [trait for trait, score in sorted_traits[:3]]
    
    recommended_careers = []
    for trait in top_traits:
        if trait == 'neuroticism':
            # For neuroticism, recommend based on score
            if personality_traits[trait] > 50:
                recommended_careers.extend(career_recommendations['neuroticism']['high'])
            else:
                recommended_careers.extend(career_recommendations['neuroticism']['low'])
        else:
            recommended_careers.extend(career_recommendations[trait])
    
    # Remove duplicates while preserving order
    recommended_careers = list(dict.fromkeys(recommended_careers))
    
    return {
        'personality_traits': personality_traits,
        'avg_sentiment': avg_sentiment,
        'avg_words': avg_words,
        'avg_score': avg_score,
        'total_analyzed': len(data),
        'recommended_careers': recommended_careers
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        username = request.form['username']
        result = get_user_data(username)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True) 