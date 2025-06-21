#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import pg from 'pg';
import dotenv from 'dotenv';
import axios from 'axios';
import { HfInference } from '@huggingface/inference';

dotenv.config();

const { Pool } = pg;

// Configuration
const CONFIG = {
  server: {
    name: process.env.MCP_SERVER_NAME || "pgvector",
    version: process.env.MCP_SERVER_VERSION || "1.0.0",
  },
  db: {
    connectionString: process.env.DATABASE_URL,
  },
  azure: {
    apiKey: process.env.AZURE_OPENAI_API_KEY,
    endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  },
  huggingface: {
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: process.env.HUGGINGFACE_MODEL || "sentence-transformers/all-MiniLM-L6-v2",
  },
  embeddings: {
    provider: process.env.EMBEDDING_PROVIDER || "auto", // auto, azure, huggingface, none
  }
};

// Validate required environment variables
if (!CONFIG.db.connectionString) {
  console.error('ERROR: DATABASE_URL environment variable is required');
  process.exit(1);
}

// Determine embedding provider based on available keys
let embeddingProvider = CONFIG.embeddings.provider;
if (embeddingProvider === "auto") {
  if (CONFIG.azure.apiKey && CONFIG.azure.endpoint) {
    embeddingProvider = "azure";
  } else if (CONFIG.huggingface.apiKey) {
    embeddingProvider = "huggingface";
  } else {
    embeddingProvider = "none";
    console.error('⚠️  No embedding provider configured. Vector search and document insertion will be disabled.');
    console.error('   To enable embeddings, set either:');
    console.error('   - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT (for Azure OpenAI)');
    console.error('   - HUGGINGFACE_API_KEY (for Hugging Face)');
  }
}

// Initialize Hugging Face client if needed
let hfClient = null;
if (embeddingProvider === "huggingface") {
  if (!CONFIG.huggingface.apiKey) {
    console.error('ERROR: HUGGINGFACE_API_KEY is required when using Hugging Face embeddings');
    process.exit(1);
  }
  hfClient = new HfInference(CONFIG.huggingface.apiKey);
}

console.error(`🔧 Embedding provider: ${embeddingProvider}`);

// Initialize PostgreSQL connection
const pool = new Pool({
  connectionString: CONFIG.db.connectionString,
});

// Test database connection
try {
  const client = await pool.connect();
  await client.query('SELECT 1');
  client.release();
  console.error('✅ Database connected successfully');
} catch (error) {
  console.error('❌ Database connection failed:', error.message);
  process.exit(1);
}

/**
 * Generate embeddings using the configured provider
 */
async function generateEmbedding(text) {
  if (embeddingProvider === "none") {
    throw new Error("No embedding provider configured. Please set up Azure OpenAI or Hugging Face credentials.");
  }
  
  try {
    switch (embeddingProvider) {
      case "azure":
        return await generateAzureEmbedding(text);
      case "huggingface":
        return await generateHuggingFaceEmbedding(text);
      default:
        throw new Error(`Unknown embedding provider: ${embeddingProvider}`);
    }
  } catch (error) {
    throw new Error(`Embedding generation failed: ${error.message}`);
  }
}

/**
 * Generate embeddings using Azure OpenAI
 */
async function generateAzureEmbedding(text) {
  const url = `${CONFIG.azure.endpoint}openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15`;
  
  const response = await axios.post(url, {
    input: text,
    model: "text-embedding-ada-002"
  }, {
    headers: {
      'Content-Type': 'application/json',
      'api-key': CONFIG.azure.apiKey
    }
  });
  
  return response.data.data[0].embedding;
}

/**
 * Generate embeddings using Hugging Face
 */
async function generateHuggingFaceEmbedding(text) {
  const response = await hfClient.featureExtraction({
    model: CONFIG.huggingface.model,
    inputs: text
  });
  
  // Convert response to array if needed
  return Array.isArray(response) ? response : Array.from(response);
}

/**
 * Perform vector similarity search
 */
async function vectorSearch(query, table = 'document_embeddings', limit = 5, similarityThreshold = 0.5) {
  const client = await pool.connect();
  try {
    // Generate embedding for the query
    const queryEmbedding = await generateEmbedding(query);
    
    // Perform cosine similarity search
    const sqlQuery = `
      SELECT 
        id,
        content,
        metadata,
        1 - (embedding <=> $1::vector) as similarity,
        created_at
      FROM ${table}
      WHERE 1 - (embedding <=> $1::vector) > $2
      ORDER BY embedding <=> $1::vector
      LIMIT $3
    `;
    
    const result = await client.query(sqlQuery, [
      `[${queryEmbedding.join(',')}]`,
      similarityThreshold,
      limit
    ]);
    
    return result.rows;
  } finally {
    client.release();
  }
}

/**
 * Search by metadata filters
 */
async function metadataSearch(filters = {}, table = 'document_embeddings', limit = 10) {
  const client = await pool.connect();
  try {
    let whereConditions = [];
    let params = [];
    let paramCount = 1;
    
    // Build dynamic WHERE clause based on filters
    Object.entries(filters).forEach(([key, value]) => {
      whereConditions.push(`metadata ->> '${key}' ILIKE $${paramCount}`);
      params.push(`%${value}%`);
      paramCount++;
    });
    
    const whereClause = whereConditions.length > 0 ? 
      `WHERE ${whereConditions.join(' AND ')}` : '';
    
    const sqlQuery = `
      SELECT 
        id,
        content,
        metadata,
        created_at
      FROM ${table}
      ${whereClause}
      ORDER BY created_at DESC
      LIMIT $${paramCount}
    `;
    
    params.push(limit);
    
    const result = await client.query(sqlQuery, params);
    return result.rows;
  } finally {
    client.release();
  }
}

/**
 * Get database statistics
 */
async function getDatabaseStats() {
  const client = await pool.connect();
  try {
    // Get all tables with vector columns
    const tablesQuery = `
      SELECT 
        table_name,
        column_name,
        data_type
      FROM information_schema.columns 
      WHERE table_schema = 'public' 
      AND data_type = 'USER-DEFINED'
      AND column_name LIKE '%embedding%'
      ORDER BY table_name;
    `;
    
    const tables = await client.query(tablesQuery);
    
    const stats = {
      tables: [],
      totalDocuments: 0
    };
    
    for (const table of tables.rows) {
      const countQuery = `SELECT COUNT(*) as count FROM "${table.table_name}"`;
      const count = await client.query(countQuery);
      
      stats.tables.push({
        name: table.table_name,
        embeddingColumn: table.column_name,
        documentCount: parseInt(count.rows[0].count)
      });
      
      stats.totalDocuments += parseInt(count.rows[0].count);
    }
    
    return stats;
  } finally {
    client.release();
  }
}

/**
 * Insert document with embedding
 */
async function insertDocument(content, metadata = {}, table = 'document_embeddings') {
  const client = await pool.connect();
  try {
    // Generate embedding
    const embedding = await generateEmbedding(content);
    
    const query = `
      INSERT INTO ${table} (content, metadata, embedding, created_at)
      VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
      RETURNING id
    `;
    
    const result = await client.query(query, [
      content,
      JSON.stringify(metadata),
      `[${embedding.join(',')}]`
    ]);
    
    return result.rows[0].id;
  } finally {
    client.release();
  }
}

// Create MCP server
const server = new Server(
  {
    name: CONFIG.server.name,
    version: CONFIG.server.version,
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Define available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  const tools = [];
  
  // Add embedding-based tools only if provider is available
  if (embeddingProvider !== "none") {
    tools.push({
      name: "vector_search",
      description: `Perform semantic vector similarity search on PostgreSQL with pgvector (using ${embeddingProvider} embeddings)`,
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query text to find similar content"
            },
            table: {
              type: "string",
              description: "Table name to search in (default: document_embeddings)",
              default: "document_embeddings"
            },
            limit: {
              type: "number",
              description: "Maximum number of results to return (default: 5)",
              default: 5,
              minimum: 1,
              maximum: 50
            },
            similarity_threshold: {
              type: "number",
              description: "Minimum similarity threshold (0-1, default: 0.5)",
              default: 0.5,
              minimum: 0,
              maximum: 1
            }
          },
          required: ["query"]
        }
      });
    
    tools.push({
      name: "insert_document",
      description: `Insert a new document with automatic embedding generation (using ${embeddingProvider} embeddings)`,
      inputSchema: {
        type: "object",
        properties: {
          content: {
            type: "string",
            description: "Document content to embed and store"
          },
          metadata: {
            type: "object",
            description: "Additional metadata to store with the document",
            additionalProperties: true
          },
          table: {
            type: "string",
            description: "Table name to insert into (default: document_embeddings)",
            default: "document_embeddings"
          }
        },
        required: ["content"]
      }
    });
  }
  
  // Always available tools (don't require embeddings)
  tools.push({
    name: "metadata_search",
        description: "Search documents by metadata filters",
        inputSchema: {
          type: "object",
          properties: {
            filters: {
              type: "object",
              description: "Key-value pairs to filter by metadata fields",
              additionalProperties: {
                type: "string"
              }
            },
            table: {
              type: "string",
              description: "Table name to search in (default: document_embeddings)",
              default: "document_embeddings"
            },
            limit: {
              type: "number",
              description: "Maximum number of results to return (default: 10)",
              default: 10,
              minimum: 1,
              maximum: 100
            }
          },
          required: ["filters"]
        }
      });
      
  tools.push({
    name: "get_database_stats",
    description: "Get statistics about vector-enabled tables in the database",
    inputSchema: {
      type: "object",
      properties: {},
      required: []
    }
  });
      
  return { tools };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    switch (name) {
      case "vector_search": {
        if (embeddingProvider === "none") {
          throw new McpError(
            ErrorCode.MethodNotFound,
            "Vector search is disabled. No embedding provider configured. Please set up Azure OpenAI or Hugging Face credentials."
          );
        }
        
        const results = await vectorSearch(
          args.query,
          args.table || 'document_embeddings',
          args.limit || 5,
          args.similarity_threshold || 0.5
        );
        
        return {
          content: [
            {
              type: "text",
              text: `Found ${results.length} similar documents:\n\n` +
                    results.map((result, index) => 
                      `${index + 1}. [Similarity: ${(result.similarity * 100).toFixed(1)}%]\n` +
                      `   Content: ${result.content.slice(0, 200)}...\n` +
                      `   Metadata: ${JSON.stringify(result.metadata, null, 2)}\n`
                    ).join('\n')
            }
          ]
        };
      }
      
      case "metadata_search": {
        const results = await metadataSearch(
          args.filters,
          args.table || 'document_embeddings',
          args.limit || 10
        );
        
        return {
          content: [
            {
              type: "text",
              text: `Found ${results.length} documents matching filters:\n\n` +
                    results.map((result, index) => 
                      `${index + 1}. ID: ${result.id}\n` +
                      `   Content: ${result.content.slice(0, 200)}...\n` +
                      `   Metadata: ${JSON.stringify(result.metadata, null, 2)}\n`
                    ).join('\n')
            }
          ]
        };
      }
      
      case "get_database_stats": {
        const stats = await getDatabaseStats();
        
        return {
          content: [
            {
              type: "text",
              text: `Database Statistics:\n\n` +
                    `Total Documents: ${stats.totalDocuments}\n\n` +
                    `Vector-enabled Tables:\n` +
                    stats.tables.map(table => 
                      `- ${table.name}: ${table.documentCount} documents (${table.embeddingColumn} column)`
                    ).join('\n')
            }
          ]
        };
      }
      
      case "insert_document": {
        if (embeddingProvider === "none") {
          throw new McpError(
            ErrorCode.MethodNotFound,
            "Document insertion with embeddings is disabled. No embedding provider configured. Please set up Azure OpenAI or Hugging Face credentials."
          );
        }
        
        const documentId = await insertDocument(
          args.content,
          args.metadata || {},
          args.table || 'document_embeddings'
        );
        
        return {
          content: [
            {
              type: "text",
              text: `Document inserted successfully with ID: ${documentId}`
            }
          ]
        };
      }
      
      default:
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${name}`
        );
    }
  } catch (error) {
    throw new McpError(
      ErrorCode.InternalError,
      `Tool execution failed: ${error.message}`
    );
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`🚀 MCP PGVector Server started successfully`);
  console.error(`📊 Database: ${CONFIG.db.connectionString.split('@')[1]?.split('/')[0] || 'connected'}`);
  console.error(`🔧 Tools: vector_search, metadata_search, get_database_stats, insert_document`);
}

main().catch((error) => {
  console.error("❌ Failed to start server:", error);
  process.exit(1);
});