# 🤖 IndiaSpend AI Chatbot - RAG-Powered News Analysis API

A sophisticated FastAPI-based conversational AI system that provides intelligent question-answering capabilities on IndiaSpend articles using Retrieval Augmented Generation (RAG), vector databases, and multiple LLM integrations.

## 🌟 Features

### 🧠 Intelligent Conversational AI
- **Streaming Responses**: Real-time Server-Sent Events (SSE) for seamless chat experience
- **Thread-based Conversations**: Persistent conversation context with thread management
- **Source Attribution**: Automatic citation of relevant article sources
- **Multi-LLM Support**: Integration with Google Gemini, OpenAI, and other providers

### 📰 Article Management System
- **Daily Article Ingestion**: Automated daily article fetching and storage
- **Custom Date Range Processing**: Bulk article processing for specific time periods
- **Vector Database Storage**: Efficient semantic search using Pinecone
- **Question Generation**: AI-powered question generation from latest articles

### 🔧 Developer Tools
- **URL Processing**: Single URL content extraction and indexing
- **Vector Search**: Direct Pinecone index querying capabilities
- **Link Extraction**: Automated article link discovery from IndiaSpend pages
- **Background Processing**: Non-blocking article processing workflows

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- API keys for: Google (Gemini), Tavily, Pinecone, OpenAI
- IndiaSpend website access

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/indiaspend-ai-chatbot.git
cd indiaspend-ai-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-dotenv langchain-core langchain-pinecone

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Configuration

Create a `.env` file with the following variables:

```bash
GOOGLE_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key  
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Running the Application

```bash
# Start the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Access the API documentation at: `http://127.0.0.1:8000/docs`

## 📍 API Endpoints

### 🗣️ Chatbot Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stream_query` | Stream chatbot responses with real-time source attribution |
| GET | `/query` | Standard chatbot query with complete response |

**Parameters:**
- `question` (required): User question
- `thread_id` (required): Conversation thread identifier

### 📚 Article Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/store_articles` | Store articles for custom date range |
| GET | `/store_daily_articles` | Fetch and store today's articles |
| GET | `/generate_questions` | Generate questions from latest articles |

### 🛠️ Developer Tools

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/process-url` | Process single URL and add to vector store |
| POST | `/query_pinecone` | Direct vector database querying |
| GET | `/extract_links` | Extract article links from IndiaSpend pages |
| GET | `/generate_tags_questions` | Extract links and generate questions |

## 🧪 Sample Requests

### Streaming Chat Query

```bash
curl -X GET "http://127.0.0.1:8000/stream_query?question=What are the latest developments in climate policy?&thread_id=user123"
```

### Store Custom Date Range Articles

```bash
curl -X POST "http://127.0.0.1:8000/store_articles" \
-H "Content-Type: application/json" \
-d '{"from_date": "2025-01-01", "to_date": "2025-01-31"}'
```

### Process Single URL

```bash
curl -X POST "http://127.0.0.1:8000/process-url" \
-H "Content-Type: application/json" \
-d '{"url": "https://www.indiaspend.com/article-url"}'
```

### Query Vector Database

```bash
curl -X POST "http://127.0.0.1:8000/query_pinecone" \
-H "Content-Type: application/json" \
-d '{"query": "climate change policies", "top_k": 5}'
```

## 🏗️ Architecture

### Core Components

- **FastAPI Framework**: High-performance async web framework
- **LangChain Integration**: Advanced LLM orchestration and RAG workflows
- **Vector Store**: Pinecone for semantic similarity search
- **Multi-LLM Support**: Google Gemini Pro, OpenAI GPT models
- **Background Tasks**: Async article processing and indexing

### Data Flow

1. **Article Ingestion**: Daily/custom scraping from IndiaSpend
2. **Text Processing**: Chunking and embedding generation
3. **Vector Storage**: Semantic indexing in Pinecone
4. **Query Processing**: RAG-based retrieval and response generation
5. **Source Attribution**: Automatic citation and relevance ranking

## 🔧 Configuration

### Chatbot Workflow

```python
# Initialize chatbot with custom configuration
mybot = Chatbot()
workflow = mybot()  # Creates LangChain workflow
```

### Vector Store Setup

```python
# Pinecone configuration
PINECONE_INDEX_NAME = "indiaspend-articles"
EMBEDDING_DIMENSION = 1536  # OpenAI ada-002 dimensions
```

### CORS Configuration

```python
# Enable cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Response Examples

### Chat Response

```json
{
  "response": "Based on recent IndiaSpend articles, climate policy developments include...",
  "sources": [
    "https://www.indiaspend.com/climate-policy-2025",
    "https://www.indiaspend.com/environmental-updates"
  ]
}
```

### Article Storage Response

```json
{
  "success": true,
  "message": "Processed 45 articles from 2025-01-01 to 2025-01-31",
  "articles_stored": 45,
  "date_range": "2025-01-01 to 2025-01-31"
}
```

### Generated Questions

```json
{
  "questions": [
    "What are the key findings in the latest climate report?",
    "How has government spending on renewable energy changed?",
    "What are the implications of new environmental policies?"
  ],
  "article_count": 12,
  "generated_on": "2025-08-30T12:47:00Z"
}
```

## 🛠️ Technologies Used

- **Framework**: FastAPI, Uvicorn
- **AI/ML**: LangChain, Google Gemini Pro, OpenAI
- **Vector Database**: Pinecone
- **Web Scraping**: Custom scrapers for IndiaSpend
- **Async Processing**: Python asyncio, Background Tasks
- **Environment Management**: python-dotenv

## 🔐 Security Features

- **API Key Management**: Environment-based configuration
- **CORS Protection**: Configurable origin restrictions  
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Comprehensive exception management

## 🚧 Development Roadmap

- [ ] **Authentication System**: User-based access control
- [ ] **Rate Limiting**: API usage restrictions
- [ ] **Caching Layer**: Redis integration for improved performance
- [ ] **Multi-language Support**: Hindi and other Indian languages
- [ ] **Advanced Analytics**: Usage metrics and conversation insights
- [ ] **Mobile API**: Optimized endpoints for mobile applications

## 📈 Performance Optimizations

- **Streaming Responses**: Reduced perceived latency
- **Background Processing**: Non-blocking article ingestion
- **Vector Similarity Search**: Fast semantic retrieval
- **Async Workflows**: Concurrent request handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Implement changes with proper testing
4. Add documentation for new endpoints
5. Commit changes (`git commit -am 'Add new feature'`)
6. Push to branch (`git push origin feature/enhancement`)
7. Open a Pull Request

### Development Guidelines

- Follow FastAPI best practices
- Add type hints for all functions
- Include docstrings for API endpoints
- Test new endpoints thoroughly
- Update README for new features

## ⚠️ Important Notes

- **Rate Limits**: Be mindful of API quotas for external services
- **Data Privacy**: Ensure compliance with data protection regulations
- **Source Attribution**: Always maintain proper article citations
- **Production Deployment**: Use proper CORS settings and security measures

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



1. **`fastapi`** - Core web framework
2. **`rag`** - Retrieval Augmented Generation architecture  
3. **`conversational-ai`** - Primary use case as chatbot system
