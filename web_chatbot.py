import streamlit as st
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re
import os
from typing import Optional, List, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="🎬 Movie Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTitle {
        color: #FF6B6B;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .movie-card {
        background-color: #FFF9C4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #FBC02D;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
def load_data():
    """Load movies dataset"""
    df = pd.read_csv('movies.csv')
    df.columns = df.columns.str.lower()
    return df

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'model' not in st.session_state:
    st.session_state.model = None

# Configure Gemini API from .env
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    # Try reading directly from .env file content
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('GEMINI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    break
    except:
        pass

if api_key and api_key.strip() and len(api_key) > 10:
    try:
        genai.configure(api_key=api_key)
        st.session_state.model = genai.GenerativeModel('gemini-2.5-flash')
        st.session_state.api_configured = True
    except Exception as e:
        st.session_state.api_configured = False
else:
    st.session_state.api_configured = False

try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"❌ Error loading movies.csv: {str(e)}")
    data_loaded = False

# Helper functions
def similarity_score(s1: str, s2: str) -> float:
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

def search_movies(query: str, threshold: float = 0.6) -> List[Tuple]:
    """Search movies by similarity"""
    matches = []
    query_lower = query.lower()

    search_fields = ['title', 'genres', 'keywords', 'cast', 'overview']

    for idx, row in df.iterrows():
        for field in search_fields:
            if field not in df.columns or pd.isna(row[field]):
                continue

            value = str(row[field]).lower()

            # Exact substring match (higher priority)
            if query_lower in value:
                matches.append((idx, row, 1.0))
                break
            # Fuzzy match
            elif similarity_score(query_lower, value) >= threshold:
                score = similarity_score(query_lower, value)
                matches.append((idx, row, score))
                break

    # Remove duplicates and sort
    seen = set()
    unique_matches = []
    for idx, row, score in sorted(matches, key=lambda x: x[2], reverse=True):
        if idx not in seen:
            seen.add(idx)
            unique_matches.append((row, score))

    return unique_matches[:5]

def format_movie_info(movie: pd.Series, detailed: bool = True) -> str:
    """Format movie information for display"""
    info = f"**{movie.get('title', 'Unknown')}**"

    if 'release_date' in movie and pd.notna(movie['release_date']):
        year = str(movie['release_date'])[:4]
        info += f" ({year})"

    info += "\n\n"

    if 'vote_average' in movie and pd.notna(movie['vote_average']):
        rating = float(movie['vote_average'])
        stars = "⭐" * int(rating / 2)
        info += f"{stars} **Rating:** {rating}/10\n\n"

    if 'genres' in movie and pd.notna(movie['genres']):
        info += f"**Genres:** {movie['genres']}\n\n"

    if detailed:
        if 'budget' in movie and pd.notna(movie['budget']) and movie['budget'] > 0:
            budget = int(movie['budget'])
            info += f"💰 **Budget:** ${budget:,}\n\n"

        if 'revenue' in movie and pd.notna(movie['revenue']) and movie['revenue'] > 0:
            revenue = int(movie['revenue'])
            info += f"💵 **Revenue:** ${revenue:,}\n\n"

        if 'runtime' in movie and pd.notna(movie['runtime']):
            info += f"⏱️ **Runtime:** {int(movie['runtime'])} minutes\n\n"

        if 'director' in movie and pd.notna(movie['director']):
            info += f"🎥 **Director:** {movie['director']}\n\n"

        if 'overview' in movie and pd.notna(movie['overview']):
            info += f"📝 **Overview:** {str(movie['overview'])[:300]}...\n"

    return info

def call_gemini_api(query: str) -> str:
    """Call Gemini API for general movie knowledge"""
    if not st.session_state.api_configured or st.session_state.model is None:
        return "⚠️ Gemini API not configured. Please add your API key in the sidebar."

    try:
        prompt = f"""You are a helpful movie expert chatbot.
        Answer this movie question concisely and informatively: {query}
        Keep the response to 2-3 paragraphs maximum."""

        response = st.session_state.model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        # Check for quota exceeded error
        if "429" in error_msg or "quota" in error_msg.lower():
            return """⚠️ **API Quota Exceeded**

The free tier Gemini API quota has been exceeded. This resets:
- **Every minute** for request limits
- **Daily** for token limits

Try again in a few moments, or you can:
1. Search the database for specific movies
2. Add billing to your Google Cloud account for higher limits
3. Wait until tomorrow for daily reset"""
        return f"❌ Error: {error_msg[:200]}"

def answer_from_dataset(query: str) -> Optional[str]:
    """Try to answer using the dataset"""
    query_lower = query.lower()

    # Pattern: Highest rated
    if any(word in query_lower for word in ['highest', 'best', 'top rated', 'best rated', 'highest rated']):
        if 'genre' in query_lower:
            genre_match = re.search(r'(action|comedy|drama|thriller|horror|romance|sci-fi|animation|adventure)', query_lower)
            if genre_match:
                genre = genre_match.group(1)
                filtered = df[df['genres'].str.contains(genre, case=False, na=False)]
                if not filtered.empty:
                    best = filtered.nlargest(1, 'vote_average').iloc[0]
                    return format_movie_info(best, True)

        best_movies = df.nlargest(3, 'vote_average')
        response = ""
        for _, movie in best_movies.iterrows():
            response += format_movie_info(movie, True) + "\n---\n"
        return response

    # Pattern: Most expensive
    if any(word in query_lower for word in ['expensive', 'highest budget', 'most costly', 'most expensive', 'budget']):
        if 'budget' in query_lower:
            expensive = df[df['budget'] > 0].nlargest(3, 'budget')
            response = ""
            for _, movie in expensive.iterrows():
                response += format_movie_info(movie, True) + "\n---\n"
            return response

    # Pattern: Longest movies
    if any(word in query_lower for word in ['longest', 'longest runtime', 'longest movie', 'runtime']):
        if any(w in query_lower for w in ['longest', 'runtime']):
            longest = df.nlargest(3, 'runtime')
            response = ""
            for _, movie in longest.iterrows():
                response += format_movie_info(movie, True) + "\n---\n"
            return response

    # Pattern: Direct movie title search (exact or close match)
    matches = search_movies(query, threshold=0.65)
    if matches and similarity_score(query_lower, str(matches[0][0]['title']).lower()) > 0.65:
        response = ""
        for movie, score in matches:
            response += format_movie_info(movie, True) + "\n---\n"
        return response

    # If no strong dataset match found, return None to trigger Gemini API
    return None

def answer_question(query: str) -> str:
    """Main function to answer questions"""
    # Try dataset first for specific patterns
    dataset_answer = answer_from_dataset(query)
    if dataset_answer:
        return dataset_answer

    # Use Gemini for general questions
    if st.session_state.api_configured:
        return call_gemini_api(query)

    return "❓ I couldn't find information about this. Try:\n- Asking about specific movies\n- Questions about ratings, budgets, or runtimes\n- Or configure Gemini API in .env for general movie questions"

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("🔑 Gemini API Status")
    if st.session_state.api_configured:
        st.success("✅ Gemini API Active")
        st.caption("API key found in .env file")
    else:
        st.error("❌ Gemini API Not Active")
        st.caption("Add GEMINI_API_KEY=your_key to .env file")
        st.info("You can still search the movie database without API!")

    st.divider()

    st.subheader("📊 Dataset Info")
    if data_loaded:
        st.metric("Total Movies", len(df))
        st.metric("Total Columns", len(df.columns))

        st.subheader("🎭 Top Genres")
        # Parse genres and count
        all_genres = []
        for genres_str in df['genres'].dropna():
            all_genres.extend(str(genres_str).split())
        if all_genres:
            from collections import Counter
            genre_counts = Counter(all_genres).most_common(5)
            for genre, count in genre_counts:
                st.write(f"• {genre}: {count}")

    st.divider()

    st.subheader("💡 Sample Questions")
    sample_questions = [
        "Tell me about Avatar",
        "What are the highest rated movies?",
        "Show me the most expensive movies",
        "What are the longest movies?",
        "Best action movies"
    ]
    for q in sample_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.title("🎬 Movie Chatbot")
st.markdown("By: Salar & Campos")
st.markdown("*Ask me anything about movies!*")
st.divider()

# Chat display
st.subheader("💬 Chat")
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <b>You:</b><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <b>🤖 Bot:</b><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)

# Input section
st.divider()
st.subheader("Ask a Question")

with st.form(key='chat_form', clear_on_submit=True):
    col1, col2 = st.columns([6, 1])

    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="e.g., Tell me about Avatar, What are the highest rated movies?",
            label_visibility="collapsed"
        )

    with col2:
        submit_button = st.form_submit_button("Send", use_container_width=True)

    if submit_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Get response
        with st.spinner("🤖 Thinking..."):
            response = answer_question(user_input)

        # Add bot response to history
        st.session_state.chat_history.append({
            "role": "bot",
            "content": response
        })

        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <p>🎬 Movie Chatbot v1.0 | Built with Streamlit & Gemini API</p>
    <p>Powered by a dataset of 5000+ movies</p>
</div>
""", unsafe_allow_html=True)
