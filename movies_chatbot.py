"""
Movie Chatbot Application
A chatbot that answers questions about movies using a dataset and Gemini API for general queries.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import re

# Import Gemini API
import google.generativeai as genai

# For semantic search (simple text-based matching)
from difflib import SequenceMatcher


class MovieChatbot:
    """
    A chatbot that answers movie-related questions.
    Uses the movies dataset for specific queries and Gemini API for general queries.
    """

    def __init__(self, csv_path: str, api_key: Optional[str] = None):
        """
        Initialize the chatbot with movie dataset and Gemini API.

        Args:
            csv_path: Path to the movies.csv file
            api_key: Google Gemini API key (if None, will try to read from environment)
        """
        # Load the dataset
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.lower()

        # Initialize Gemini API
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')

        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.gemini_available = True
        else:
            self.gemini_available = False
            print("Warning: Gemini API key not found. General queries will be limited.")

        print(f"✓ Loaded {len(self.df)} movies from dataset")
        print(f"✓ Dataset columns: {', '.join(self.df.columns[:10])}")

    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate similarity score between two strings."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _search_movies(self, query: str, field: str = None, threshold: float = 0.6) -> list:
        """
        Search movies by similarity in dataset.

        Args:
            query: Search query
            field: Specific field to search in (title, genres, keywords, etc.)
            threshold: Minimum similarity score (0-1)

        Returns:
            List of matching movies
        """
        matches = []
        query_lower = query.lower()

        # If no specific field, search in multiple fields
        search_fields = [field] if field else ['title', 'genres', 'keywords', 'cast', 'overview']

        for idx, row in self.df.iterrows():
            for search_field in search_fields:
                if search_field not in self.df.columns:
                    continue

                value = str(row[search_field]).lower()

                # Direct substring match (highest priority)
                if query_lower in value:
                    matches.append((idx, row, 1.0))
                    break
                # Similarity score match
                elif self._similarity_score(query_lower, value) >= threshold:
                    score = self._similarity_score(query_lower, value)
                    matches.append((idx, row, score))
                    break

        # Remove duplicates and sort by score
        seen = set()
        unique_matches = []
        for idx, row, score in sorted(matches, key=lambda x: x[2], reverse=True):
            if idx not in seen:
                seen.add(idx)
                unique_matches.append((row, score))

        return unique_matches[:5]  # Return top 5 matches

    def _format_movie_info(self, movie: pd.Series, include_all: bool = False) -> str:
        """Format movie information for display."""
        info = f"\n🎬 **{movie.get('title', 'Unknown')}**"

        if 'release_date' in movie and pd.notna(movie['release_date']):
            info += f" ({movie['release_date'][:4]})"

        if not include_all:
            # Brief info
            if 'vote_average' in movie and pd.notna(movie['vote_average']):
                info += f"\n⭐ Rating: {movie['vote_average']}/10"

            if 'genres' in movie and pd.notna(movie['genres']):
                info += f"\n🎭 Genres: {movie['genres']}"

            if 'overview' in movie and pd.notna(movie['overview']):
                overview = str(movie['overview'])[:200]
                info += f"\n📝 Overview: {overview}..."
        else:
            # Detailed info
            if 'vote_average' in movie and pd.notna(movie['vote_average']):
                info += f"\n⭐ Rating: {movie['vote_average']}/10 ({movie.get('vote_count', 0)} votes)"

            if 'genres' in movie and pd.notna(movie['genres']):
                info += f"\n🎭 Genres: {movie['genres']}"

            if 'release_date' in movie and pd.notna(movie['release_date']):
                info += f"\n📅 Release Date: {movie['release_date']}"

            if 'runtime' in movie and pd.notna(movie['runtime']):
                info += f"\n⏱️  Runtime: {int(movie['runtime'])} minutes"

            if 'budget' in movie and pd.notna(movie['budget']) and movie['budget'] > 0:
                budget = int(movie['budget'])
                info += f"\n💰 Budget: ${budget:,}"

            if 'revenue' in movie and pd.notna(movie['revenue']) and movie['revenue'] > 0:
                revenue = int(movie['revenue'])
                info += f"\n💵 Revenue: ${revenue:,}"

            if 'overview' in movie and pd.notna(movie['overview']):
                info += f"\n📝 Overview: {movie['overview']}"

            if 'director' in movie and pd.notna(movie['director']):
                info += f"\n🎥 Director: {movie['director']}"

            if 'cast' in movie and pd.notna(movie['cast']):
                cast = str(movie['cast'])[:100]
                info += f"\n👥 Cast: {cast}"

        return info

    def _answer_from_dataset(self, query: str) -> Optional[str]:
        """Try to answer query using the dataset."""
        query_lower = query.lower()

        # Question patterns
        patterns = {
            'highest_rated': r'(highest|best|top rated|best rated)',
            'lowest_rated': r'(lowest|worst|worst rated)',
            'highest_budget': r'(highest|most expensive|highest budget)',
            'longest': r'(longest|longest runtime)',
            'shortest': r'(shortest|shortest runtime)',
            'most_popular': r'(most popular|most voted|highest votes)',
            'by_genre': r'(movies in|movies with|genre)',
            'by_year': r'(year|release date)',
            'director': r'(by|director|directed)',
            'cast': r'(starring|with|actor|actress)',
        }

        # Check highest rated
        if any(re.search(pattern, query_lower) for pattern in [patterns['highest_rated']]):
            if 'genre' in query_lower:
                genre_match = re.search(r'(action|comedy|drama|thriller|horror|romance|sci-fi|animation|adventure|fantasy)', query_lower)
                if genre_match:
                    genre = genre_match.group(1).capitalize()
                    filtered = self.df[self.df['genres'].str.contains(genre, case=False, na=False)]
                    if not filtered.empty:
                        best = filtered.nlargest(1, 'vote_average').iloc[0]
                        return f"The highest-rated {genre} movie is: {self._format_movie_info(best)}"

            best = self.df.nlargest(3, 'vote_average')
            response = "🏆 **Top Rated Movies:**"
            for idx, movie in best.iterrows():
                response += self._format_movie_info(movie)
            return response

        # Check lowest rated
        if any(re.search(pattern, query_lower) for pattern in [patterns['lowest_rated']]):
            worst = self.df.nsmallest(3, 'vote_average')
            response = "📉 **Lowest Rated Movies:**"
            for idx, movie in worst.iterrows():
                response += self._format_movie_info(movie)
            return response

        # Check by budget
        if any(re.search(pattern, query_lower) for pattern in [patterns['highest_budget']]):
            expensive = self.df.nlargest(3, 'budget')
            response = "💸 **Most Expensive Movies:**"
            for idx, movie in expensive.iterrows():
                if movie['budget'] > 0:
                    response += self._format_movie_info(movie)
            return response

        # Check by runtime
        if any(re.search(pattern, query_lower) for pattern in [patterns['longest']]):
            longest = self.df.nlargest(3, 'runtime')
            response = "⏱️  **Longest Movies:**"
            for idx, movie in longest.iterrows():
                response += self._format_movie_info(movie)
            return response

        # Search for specific movie
        if 'movie' in query_lower or 'film' in query_lower or 'about' in query_lower:
            # Extract potential movie name
            movie_name = re.sub(r'(movie|film|about|tell me|info|what|is)', '', query_lower).strip()
            matches = self._search_movies(movie_name)

            if matches:
                response = "🎬 **Found movies matching your query:**"
                for movie, score in matches:
                    response += self._format_movie_info(movie, include_all=True)
                return response

        # Direct movie search
        matches = self._search_movies(query)
        if matches:
            response = "🎬 **Found movies:**"
            for movie, score in matches:
                response += self._format_movie_info(movie)
            return response

        return None

    def _answer_with_gemini(self, query: str, dataset_context: str = "") -> str:
        """Use Gemini API to answer queries outside the dataset."""
        if not self.gemini_available:
            return "I don't have information about this query. Please ask about movies in the database or provide a Gemini API key."

        prompt = f"""You are a helpful movie expert chatbot. Answer the following question about movies.

Database Context (if relevant):
{dataset_context}

User Question: {query}

Provide a helpful and concise answer. If the question is about general movie knowledge, provide general information.
If you found relevant movies in the database context, mention them."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def answer_question(self, query: str) -> str:
        """
        Answer a question about movies.

        Args:
            query: User's question

        Returns:
            Answer to the question
        """
        # First, try to answer from dataset
        dataset_answer = self._answer_from_dataset(query)
        if dataset_answer:
            return dataset_answer

        # If no dataset answer, try Gemini with context
        if self.gemini_available:
            # Provide some context from the dataset
            dataset_info = f"Database contains {len(self.df)} movies with fields: {', '.join(self.df.columns[:8])}"
            return self._answer_with_gemini(query, dataset_info)

        return "I couldn't find information about this in the database. Try asking about movie ratings, budgets, or specific movie titles."

    def chat(self):
        """Interactive chat loop."""
        print("\n" + "="*60)
        print("🎬 Welcome to Movie Chatbot!")
        print("="*60)
        print("\nYou can ask about:")
        print("  • Specific movies (e.g., 'Tell me about Avatar')")
        print("  • Highest/lowest rated movies")
        print("  • Movies by genre, director, or cast")
        print("  • Movie budgets, revenues, and runtimes")
        print("  • General movie questions")
        print("\nType 'exit' to quit\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() == 'exit':
                    print("\n👋 Thank you for chatting! Goodbye!")
                    break
                if not user_input:
                    continue

                response = self.answer_question(user_input)
                print(f"\nBot: {response}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Chatbot closed. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n")


def main():
    """Main function to run the chatbot."""
    # Get API key from environment or user input
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("Note: Gemini API key not found in environment variables.")
        api_key_input = input("Enter your Gemini API key (or press Enter to skip): ").strip()
        if api_key_input:
            api_key = api_key_input

    # Initialize chatbot
    csv_path = "movies.csv"

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return

    chatbot = MovieChatbot(csv_path, api_key)
    chatbot.chat()


if __name__ == "__main__":
    main()
