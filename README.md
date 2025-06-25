# MCP PGVector Server
[![smithery badge](https://smithery.ai/badge/@yusufferdogan/mcp-pgvector-server)](https://smithery.ai/server/@yusufferdogan/mcp-pgvector-server)

A Model Context Protocol (MCP) server that provides semantic search capabilities for PostgreSQL databases using pgvector extensions. Supports multiple embedding providers including Azure OpenAI and Hugging Face.

## 🚀 Quick Start

### Installing via Smithery

To install PGVector Semantic Search Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@yusufferdogan/mcp-pgvector-server):

```bash
npx -y @smithery/cli install @yusufferdogan/mcp-pgvector-server --client claude
```

### Using npx (Recommended)

```bash
# With Azure OpenAI embeddings
DATABASE_URL="postgresql://user:pass@localhost:5432/db" \
AZURE_OPENAI_API_KEY="your-key" \
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/" \
npx mcp-pgvector-server

# With Hugging Face embeddings  
DATABASE_URL="postgresql://user:pass@localhost:5432/db" \
HUGGINGFACE_API_KEY="your-hf-token" \
npx mcp-pgvector-server

# Metadata-only mode (no embeddings)
DATABASE_URL="postgresql://user:pass@localhost:5432/db" \
npx mcp-pgvector-server
```

### Local Installation

```bash
# Install globally
npm install -g mcp-pgvector-server

# Run with environment variables
mcp-pgvector-server
```

## 📋 Prerequisites

1. **PostgreSQL with pgvector**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Embedding Provider** (optional - choose one or none):
   - Azure OpenAI Account with text-embedding-ada-002 deployment
   - Hugging Face API token for transformer models

3. **Node.js 18+**

## ⚙️ Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost:5432/db` |

### Embedding Provider Options (choose one or none)

#### Azure OpenAI
| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | `your-api-key` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | `https://your-endpoint.openai.azure.com/` |

#### Hugging Face
| Variable | Description | Example |
|----------|-------------|---------|
| `HUGGINGFACE_API_KEY` | Hugging Face API token | `your-hf-token` |
| `HUGGINGFACE_MODEL` | Model to use (optional) | `sentence-transformers/all-MiniLM-L6-v2` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `auto` | Force provider: auto\|azure\|huggingface\|none |
| `MCP_SERVER_NAME` | `pgvector` | Server name for MCP |
| `MCP_SERVER_VERSION` | `1.0.0` | Server version |

## 🔧 Available Tools

### 1. `vector_search`
Perform semantic similarity search using vector embeddings.

**Parameters:**
- `query` (required): Search query text
- `table` (optional): Table name (default: `document_embeddings`)
- `limit` (optional): Max results (default: 5, max: 50)
- `similarity_threshold` (optional): Min similarity (default: 0.5)

**Example:**
```json
{
  "query": "JavaScript React frontend",
  "table": "document_embeddings",
  "limit": 10,
  "similarity_threshold": 0.7
}
```

### 2. `metadata_search`
Search documents by metadata filters.

**Parameters:**
- `filters` (required): Key-value pairs for metadata filtering
- `table` (optional): Table name (default: `document_embeddings`)
- `limit` (optional): Max results (default: 10, max: 100)

**Example:**
```json
{
  "filters": {
    "repository_name": "up-ai",
    "author_name": "yusufferdogan"
  },
  "limit": 5
}
```

### 3. `get_database_stats`
Get statistics about vector-enabled tables.

**Parameters:** None

**Returns:** Table statistics and document counts

### 4. `insert_document`
Insert a new document with automatic embedding generation.

**Parameters:**
- `content` (required): Document content to embed
- `metadata` (optional): Additional metadata object
- `table` (optional): Table name (default: `document_embeddings`)

**Example:**
```json
{
  "content": "This is a sample document about AI and machine learning.",
  "metadata": {
    "author": "John Doe",
    "category": "AI",
    "tags": ["machine-learning", "artificial-intelligence"]
  }
}
```

## 🗄️ Database Schema

The server works with tables that have vector embedding columns. Default schema:

```sql
CREATE TABLE document_embeddings (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSONB,
  embedding vector(1536),  -- Azure OpenAI ada-002 dimensions
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create vector index for fast similarity search
CREATE INDEX ON document_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);
```

## 🔌 Integration with Claude Code

### MCP Configuration

Add to your Claude Code MCP configuration file:

```json
{
  "mcpServers": {
    "pgvector": {
      "command": "npx",
      "args": ["mcp-pgvector-server"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
        "AZURE_OPENAI_API_KEY": "your-api-key",
        "AZURE_OPENAI_ENDPOINT": "https://your-endpoint.openai.azure.com/"
      }
    }
  }
}
```

### Claude Code Usage Examples

```bash
# Search for similar content
"Search for documents about 'React components' using vector search"

# Filter by metadata
"Find all documents by author 'yusufferdogan' in the up-ai repository"

# Get database overview
"Show me statistics about the vector database"

# Add new content
"Insert this code snippet about authentication with metadata"
```

## 🎯 Use Cases

- **Code Search**: Find similar code patterns and implementations
- **Documentation Retrieval**: Semantic search through technical docs
- **Knowledge Base**: Build intelligent Q&A systems
- **Content Discovery**: Find related articles, commits, or discussions
- **Learning Systems**: Track and retrieve educational content

## 🏗️ Development

### Local Development

```bash
# Clone and install dependencies
git clone <repository>
cd mcp-pgvector-server
npm install

# Set environment variables
export DATABASE_URL="postgresql://localhost:5432/testdb"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

# Start development server
npm run dev
```

### Testing

```bash
# Test the server locally
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | node src/index.js
```

## 📦 Publishing

```bash
# Build and publish to npm
npm run build
npm publish
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check DATABASE_URL format
   - Verify PostgreSQL is running
   - Ensure pgvector extension is installed

2. **Embedding Generation Failed**
   - Verify Azure OpenAI credentials
   - Check API endpoint format
   - Ensure text-embedding-ada-002 deployment exists

3. **Tool Not Found**
   - Verify MCP configuration
   - Check environment variables
   - Restart Claude Code

### Debug Mode

```bash
# Run with debug logging
DEBUG=* npx @mcp/pgvector-server
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 🆘 Support

- Issues: [GitHub Issues](https://github.com/yusuf/mcp-pgvector-server/issues)
- Documentation: [MCP Documentation](https://modelcontextprotocol.io/)
- pgvector: [pgvector GitHub](https://github.com/pgvector/pgvector)