# 🎬 Movie Chatbot - Complete Guide

A powerful movie chatbot that answers questions about movies using a dataset combined with Google's Gemini API for enhanced responses.

## 📋 Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Local Usage](#local-usage)
5. [Google Colab Usage](#google-colab-usage)
6. [How It Works](#how-it-works)
7. [Example Questions](#example-questions)
8. [API Configuration](#api-configuration)

---

## ✨ Features

- **Movie Dataset Search**: Search through 5000+ movies with intelligent matching
- **Gemini API Integration**: Handle questions beyond the dataset
- **Smart Responses**: Different response types based on query patterns:

  - Highest/lowest rated movies
  - Movie budgets and revenues
  - Movie recommendations by genre
  - Detailed movie information
  - General movie knowledge via Gemini API

- **Easy to Use**: Simple text-based interface
- **Flexible Deployment**: Run locally or on Google Colab

---

## 📦 Prerequisites

### Required

- Python 3.7+
- Google Gemini API key (get free at https://ai.google.dev/)

### Python Libraries

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `google-generativeai` - Gemini API integration

---

## 🚀 Installation & Setup

### Step 1: Get Gemini API Key

1. Visit https://ai.google.dev/
2. Click "Get API Key"
3. Create a new API key (free tier available)
4. Copy your API key

### Step 2: Prepare Files

Required files:

- `movies.csv` - The dataset file (provided)
- `movies_chatbot.py` - The chatbot application
- Your API key

---

## 💻 Local Usage

### Step 1: Install Dependencies

```bash
pip install pandas numpy google-generativeai
```

### Step 2: Set Environment Variable (Optional)

**Windows (CMD):**

```cmd
set GEMINI_API_KEY=your_api_key_here
```

**Windows (PowerShell):**

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**Mac/Linux:**

```bash
export GEMINI_API_KEY=your_api_key_here
```

Or you can provide the API key directly when the script asks for it.

### Step 3: Run the Chatbot

```bash
python movies_chatbot.py
```

You'll be prompted for your Gemini API key if not in environment variables.

### Example Interaction

```
You: Tell me about Avatar
Bot: 🎬 **Avatar** (2009)
     ⭐ Rating: 7.2/10
     🎭 Genres: Action Adventure Fantasy Science Fiction
     📝 Overview: In the 22nd century, a paraplegic Marine...

You: What's the highest rated movie?
Bot: 🏆 **Top Rated Movies:**
     🎬 **The Dark Knight Rises** (2012)
     ⭐ Rating: 7.6/10
     ... (more movies)

You: exit
Bot: 👋 Thank you for chatting! Goodbye!
```

---

## ☁️ Google Colab Usage

### Step 1: Open Google Colab

1. Go to https://colab.research.google.com/
2. Create a new notebook

### Step 2: Upload Dataset

```python
from google.colab import files
files.upload()
```

Upload the `movies.csv` file

### Step 3: Install Requirements

```python
!pip install google-generativeai pandas numpy
```

### Step 4: Copy Colab Code

Copy the entire content from `movies_chatbot_colab.ipynb` into your Colab notebook.

### Step 5: Configure and Run

Uncomment these lines at the bottom of your notebook:

```python
# Uncomment to run in Colab
api_key = input("🔑 Enter your Google Gemini API key: ")
chatbot = MovieChatbot('movies.csv', api_key)
colab_chat_interactive(chatbot)
```

Then run the cell.

### Colab Cells Breakdown

**Cell 1: Setup & Imports**

```python
import subprocess
import sys

# Install packages
packages = ['google-generativeai', 'pandas', 'numpy']
for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
```

**Cell 2: Define Chatbot Class**
(Copy the entire MovieChatbot class from the notebook)

**Cell 3: Upload Dataset & Initialize**

```python
from google.colab import files

# Upload movies.csv
print("Upload movies.csv:")
files.upload()

# Get API key
api_key = input("🔑 Enter your Google Gemini API key: ")

# Initialize chatbot
chatbot = MovieChatbot('movies.csv', api_key)
```

**Cell 4: Start Chatting**

```python
colab_chat_interactive(chatbot)
```

---

## 🧠 How It Works

### Architecture

```
User Query
    ↓
Local Dataset Search
(titles, genres, cast, keywords)
    ↓
Match Found? → YES → Format & Return Response
    ↓ NO
Gemini API Query
(with dataset context)
    ↓
Return AI-Generated Response
```

### Query Processing

1. **Pattern Recognition**: Identifies question type

   - Rating queries (highest/lowest)
   - Budget/revenue queries
   - Runtime queries
   - Movie search queries
   - Genre-specific queries

2. **Dataset Search**: Finds matches using similarity scoring

   - Substring matching (priority)
   - Fuzzy matching (similarity)
   - Multi-field search

3. **Fallback to Gemini**:
   - Handles unknown query types
   - Provides general movie knowledge
   - Contextual responses

### Key Components

| Component                | Purpose                       |
| ------------------------ | ----------------------------- |
| `_similarity_score()`    | Calculate string similarity   |
| `_search_movies()`       | Search dataset with ranking   |
| `_answer_from_dataset()` | Pattern-based dataset queries |
| `_answer_with_gemini()`  | AI-powered responses          |
| `answer_question()`      | Main orchestration method     |

---

## 🎯 Example Questions

### Movie Search

```
"Tell me about Avatar"
"What's Inception about?"
"Find movies with Tom Cruise"
```

### Ratings

```
"What are the highest rated movies?"
"Show me the worst rated films"
"What's the best action movie?"
```

### Budget & Revenue

```
"What are the most expensive movies?"
"Which movies made the most money?"
"Show movies with highest budgets"
```

### Duration

```
"What's the longest movie?"
"Show me short films"
"Movies over 3 hours"
```

### General Knowledge

```
"What makes a good movie?"
"How are movies rated?"
"What's the difference between genres?"
```

---

## 🔑 API Configuration

### Getting Your Gemini API Key

1. **Visit Google AI**: https://ai.google.dev/
2. **Click "Get API Key"**
3. **Sign in with your Google account**
4. **Create API Key**:
   - Select "Create API Key in new project" or existing project
   - Copy the generated key
5. **Store safely** (don't share publicly)

### Setting API Key

#### Option 1: Environment Variable

```bash
# Linux/Mac
export GEMINI_API_KEY=sk-xxxxxxxxxxxxx

# Windows
setx GEMINI_API_KEY sk-xxxxxxxxxxxxx
```

#### Option 2: Interactive Input

The script will prompt you to enter the key if not in environment variables.

#### Option 3: Hardcode (Not Recommended)

```python
chatbot = MovieChatbot('movies.csv', 'your_api_key_here')
```

### API Usage

- **Free Tier**: 60 requests per minute
- **Rate Limits**: Check Google AI Studio dashboard
- **Cost**: Currently free (may change)

---

## 📊 Dataset Information

### Dataset Size

- **Total Movies**: 5000+
- **Fields**: 24 columns including:
  - Title, Release Date, Genres
  - Budget, Revenue, Runtime
  - Vote Average, Vote Count
  - Director, Cast, Crew
  - Overview, Keywords
  - Production Companies, Languages

### Data Quality

- Some fields may be missing (NaN values handled)
- Genres contain multiple values (comma-separated)
- Cast and crew data in JSON format

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'google.generativeai'"

**Solution:**

```bash
pip install google-generativeai
```

### Issue: "API key invalid"

**Solution:**

1. Verify your API key from https://ai.google.dev/
2. Ensure no extra spaces in the key
3. Check if API is enabled in Google Cloud

### Issue: "movies.csv not found"

**Solution:**

1. Place `movies.csv` in the same directory as the script
2. Or provide the full path to the CSV file

### Issue: Slow responses

**Solution:**

- API calls take a few seconds, this is normal
- Ensure internet connection is stable
- Check API rate limits

---

## 📝 Code Structure

### Main Classes

#### MovieChatbot

Main class handling all chatbot functionality

**Key Methods:**

- `__init__(csv_path, api_key)` - Initialize
- `answer_question(query)` - Main query handler
- `_search_movies(query)` - Dataset search
- `_answer_from_dataset(query)` - Pattern-based responses
- `_answer_with_gemini(query)` - AI responses
- `chat()` - Interactive loop

### Usage Pattern

```python
from movies_chatbot import MovieChatbot

# Initialize
chatbot = MovieChatbot('movies.csv', 'your_api_key')

# Single question
response = chatbot.answer_question("What's the best movie?")
print(response)

# Interactive mode
chatbot.chat()
```

---

## 🎓 Learning Resources

### Python Concepts Used

- **Pandas**: Data manipulation and analysis
- **RegEx**: Pattern matching in queries
- **String Similarity**: Fuzzy matching algorithms
- **API Integration**: Gemini API calls
- **Error Handling**: Try-except blocks

### Gemini API Documentation

- https://ai.google.dev/docs
- https://ai.google.dev/tutorials

### Dataset Analysis

- The `_search_movies()` method uses fuzzy matching
- The `_answer_from_dataset()` method uses regex patterns
- Movie information is formatted with emoji indicators

---

## 🚀 Future Enhancements

Potential improvements:

1. **Vector Embeddings**: Better semantic search
2. **Conversation Memory**: Remember previous questions
3. **Advanced Filters**: Complex query combinations
4. **User Preferences**: Personalized recommendations
5. **Database**: Store conversation history
6. **Web Interface**: Flask/Streamlit UI
7. **Multi-language**: Support multiple languages

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 🤝 Support

For issues or questions:

1. Check the troubleshooting section
2. Verify API key validity
3. Ensure CSV file is present
4. Check internet connection
5. Review error messages carefully

---

## ✨ Tips & Tricks

1. **More Specific Queries**: "Best action movies rated above 7.0" works better than just "best movies"
2. **Try Different Phrasings**: If one question doesn't work, rephrase it
3. **Use Movie Names**: Full titles usually work better than partial names
4. **Combine Filters**: "highest budget action movies" works well

---

**Happy Chatting! 🎬🎭✨**
