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

// Check for health check mode
const isHealthCheck = process.argv.includes('--health') || process.env.HEALTH_CHECK === 'true';

// Validate required environment variables (skip for health check)
if (!CONFIG.db.connectionString && !isHealthCheck) {
  console.error('ERROR: DATABASE_URL environment variable is required');
  console.error('For health checks, use --health flag or set HEALTH_CHECK=true');
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

// Initialize PostgreSQL connection (lazy initialization)
let pool = null;
let dbConnectionStatus = 'not_attempted';

async function initializeDatabase() {
  if (isHealthCheck) {
    console.error('🏥 Health check mode - skipping database connection');
    dbConnectionStatus = 'skipped_health_check';
    return;
  }
  
  if (!CONFIG.db.connectionString) {
    console.error('⚠️  No DATABASE_URL provided - running in connection-less mode');
    dbConnectionStatus = 'no_url';
    return;
  }

  try {
    pool = new Pool({
      connectionString: CONFIG.db.connectionString,
    });
    
    const client = await pool.connect();
    await client.query('SELECT 1');
    client.release();
    console.error('✅ Database connected successfully');
    dbConnectionStatus = 'connected';
  } catch (error) {
    console.error('❌ Database connection failed:', error.message);
    console.error('🔄 Server will continue in limited mode (schema tools disabled)');
    dbConnectionStatus = 'failed';
    pool = null;
  }
}

async function ensurePoolConnection() {
  if (dbConnectionStatus === 'not_attempted') {
    await initializeDatabase();
  }
  
  if (!pool) {
    throw new Error('Database connection not available. Please check DATABASE_URL and ensure PostgreSQL is accessible.');
  }
  
  return pool;
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
 * Get the embedding column name for a table
 */
async function getEmbeddingColumnName(tableName) {
  const currentPool = await ensurePoolConnection();
  const client = await currentPool.connect();
  try {
    const result = await client.query(`
      SELECT column_name 
      FROM information_schema.columns 
      WHERE table_name = $1 
      AND table_schema = 'public'
      AND (column_name LIKE '%embedding%' OR data_type = 'USER-DEFINED')
      ORDER BY 
        CASE 
          WHEN column_name = 'embedding' THEN 1
          WHEN column_name = 'content_embedding' THEN 2
          ELSE 3
        END
      LIMIT 1
    `, [tableName]);
    
    return result.rows[0]?.column_name || 'embedding';
  } finally {
    client.release();
  }
}

/**
 * Perform vector similarity search
 */
async function vectorSearch(query, table = 'document_embeddings', limit = 5, similarityThreshold = 0.5) {
  const currentPool = await ensurePoolConnection();
  const client = await currentPool.connect();
  try {
    // Generate embedding for the query
    const queryEmbedding = await generateEmbedding(query);
    
    // Get the correct embedding column name for this table
    const embeddingColumn = await getEmbeddingColumnName(table);
    
    // Perform cosine similarity search
    const sqlQuery = `
      SELECT 
        id,
        content,
        metadata,
        1 - (${embeddingColumn} <=> $1::vector) as similarity,
        created_at
      FROM ${table}
      WHERE 1 - (${embeddingColumn} <=> $1::vector) > $2
      ORDER BY ${embeddingColumn} <=> $1::vector
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
  const currentPool = await ensurePoolConnection();
  const client = await currentPool.connect();
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
 * Get detailed table schemas and column information
 */
async function getTableSchemas() {
  const currentPool = await ensurePoolConnection();
  const client = await currentPool.connect();
  try {
    // Get all tables with their columns
    const tablesQuery = `
      SELECT 
        t.table_name,
        c.column_name,
        c.data_type,
        c.is_nullable,
        c.column_default,
        c.udt_name,
        c.character_maximum_length
      FROM information_schema.tables t
      LEFT JOIN information_schema.columns c ON t.table_name = c.table_name
      WHERE t.table_schema = 'public' 
      AND t.table_type = 'BASE TABLE'
      ORDER BY t.table_name, c.ordinal_position;
    `;
    
    const result = await client.query(tablesQuery);
    const tables = {};
    
    // Group columns by table
    result.rows.forEach(row => {
      if (!tables[row.table_name]) {
        tables[row.table_name] = {
          name: row.table_name,
          columns: [],
          embeddingColumns: [],
          vectorDimensions: {},
          rowCount: 0
        };
      }
      
      if (row.column_name) {
        const column = {
          name: row.column_name,
          type: row.data_type,
          nullable: row.is_nullable === 'YES',
          default: row.column_default,
          udtName: row.udt_name,
          maxLength: row.character_maximum_length
        };
        
        tables[row.table_name].columns.push(column);
        
        // Detect embedding columns
        if (row.column_name.includes('embedding') && row.udt_name === 'vector') {
          tables[row.table_name].embeddingColumns.push(row.column_name);
        }
      }
    });
    
    // Get row counts and vector dimensions
    for (const tableName of Object.keys(tables)) {
      try {
        // Get row count
        const countResult = await client.query(`SELECT COUNT(*) as count FROM "${tableName}"`);
        tables[tableName].rowCount = parseInt(countResult.rows[0].count);
        
        // Get vector dimensions for embedding columns
        for (const embeddingCol of tables[tableName].embeddingColumns) {
          try {
            const sampleResult = await client.query(
              `SELECT array_length(${embeddingCol}::real[], 1) as dimensions 
               FROM "${tableName}" 
               WHERE ${embeddingCol} IS NOT NULL 
               LIMIT 1`
            );
            if (sampleResult.rows[0]?.dimensions) {
              tables[tableName].vectorDimensions[embeddingCol] = sampleResult.rows[0].dimensions;
            }
          } catch (e) {
            tables[tableName].vectorDimensions[embeddingCol] = 'unknown';
          }
        }
      } catch (e) {
        tables[tableName].rowCount = 0;
      }
    }
    
    return {
      tables: Object.values(tables),
      usage: {
        vectorSearch: "Use vector_search on tables with embedding columns for semantic similarity",
        metadataSearch: "Use metadata_search on tables with metadata/jsonb columns for filtering",
        insertDocument: "Use insert_document to add new content with automatic embeddings",
        tableParameter: "Specify 'table' parameter to target specific tables in queries"
      }
    };
  } finally {
    client.release();
  }
}

/**
 * Get database statistics
 */
async function getDatabaseStats() {
  const currentPool = await ensurePoolConnection();
  const client = await currentPool.connect();
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
  const currentPool = await ensurePoolConnection();
  const client = await currentPool.connect();
  try {
    // Generate embedding
    const embedding = await generateEmbedding(content);
    
    // Get the correct embedding column name for this table
    const embeddingColumn = await getEmbeddingColumnName(table);
    
    const query = `
      INSERT INTO ${table} (content, metadata, ${embeddingColumn}, created_at)
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
  
  tools.push({
    name: "get_table_schemas",
    description: "Get detailed schemas and column information for all tables in the database",
    inputSchema: {
      type: "object",
      properties: {},
      required: []
    }
  });
  
  // Always add status check tool
  tools.push({
    name: "status_check",
    description: "Check server status, database connectivity, and available functionality",
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
      
      case "get_table_schemas": {
        const schemas = await getTableSchemas();
        
        return {
          content: [
            {
              type: "text",
              text: `Database Table Schemas:\n\n` +
                    schemas.tables.map(table => {
                      let tableInfo = `## ${table.name} (${table.rowCount} rows)\n`;
                      
                      // Embedding columns info
                      if (table.embeddingColumns.length > 0) {
                        tableInfo += `📊 Embedding Columns: ${table.embeddingColumns.join(', ')}\n`;
                        table.embeddingColumns.forEach(col => {
                          const dims = table.vectorDimensions[col];
                          tableInfo += `   - ${col}: ${dims} dimensions\n`;
                        });
                      }
                      
                      // All columns
                      tableInfo += `\nColumns:\n`;
                      table.columns.forEach(col => {
                        const nullable = col.nullable ? ' (nullable)' : ' (required)';
                        const defaultVal = col.default ? ` [default: ${col.default}]` : '';
                        tableInfo += `   - ${col.name}: ${col.type}${nullable}${defaultVal}\n`;
                      });
                      
                      return tableInfo;
                    }).join('\n') +
                    `\n${schemas.usage.vectorSearch}\n` +
                    `${schemas.usage.metadataSearch}\n` +
                    `${schemas.usage.insertDocument}\n` +
                    `${schemas.usage.tableParameter}`
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
      
      case "status_check": {
        const status = {
          server: "MCP PGVector Server",
          version: CONFIG.server.version,
          embeddingProvider: embeddingProvider,
          database: {
            status: dbConnectionStatus,
            hasConnection: dbConnectionStatus === 'connected',
            connectionString: CONFIG.db.connectionString ? 
              `${CONFIG.db.connectionString.split('@')[1]?.split('/')[0] || 'configured'}` : 'not configured'
          },
          availableTools: [],
          limitations: []
        };
        
        // Determine available functionality
        if (dbConnectionStatus === 'connected') {
          if (embeddingProvider !== "none") {
            status.availableTools = ["vector_search", "metadata_search", "get_database_stats", "get_table_schemas", "insert_document"];
          } else {
            status.availableTools = ["metadata_search", "get_database_stats", "get_table_schemas"];
            status.limitations.push("Vector search disabled - no embedding provider configured");
          }
        } else {
          status.availableTools = ["status_check"];
          status.limitations.push("Database tools disabled - no database connection");
          
          if (dbConnectionStatus === 'no_url') {
            status.limitations.push("Set DATABASE_URL environment variable to enable database features");
          } else if (dbConnectionStatus === 'failed') {
            status.limitations.push("Database connection failed - check connectivity and credentials");
          } else if (dbConnectionStatus === 'skipped_health_check') {
            status.limitations.push("Health check mode - database connection skipped");
          }
        }
        
        return {
          content: [
            {
              type: "text",
              text: `Server Status Report:

🚀 Server: ${status.server} v${status.version}
🔧 Embedding Provider: ${status.embeddingProvider}

📊 Database Status: ${status.database.status}
🔗 Connection: ${status.database.connectionString}
✅ Connected: ${status.database.hasConnection ? 'Yes' : 'No'}

🛠️  Available Tools: ${status.availableTools.join(', ')}

${status.limitations.length > 0 ? '⚠️  Limitations:\n' + status.limitations.map(l => `   - ${l}`).join('\n') : '✅ All features available'}

${isHealthCheck ? '\n🏥 Running in health check mode' : ''}`
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
  // Initialize database connection (lazy)
  await initializeDatabase();
  
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`🚀 MCP PGVector Server started successfully`);
  
  // Database status reporting
  const dbStatus = {
    'connected': CONFIG.db.connectionString?.split('@')[1]?.split('/')[0] || 'connected',
    'failed': 'connection failed - limited mode',
    'no_url': 'no DATABASE_URL - limited mode', 
    'skipped_health_check': 'health check mode',
    'not_attempted': 'not initialized'
  };
  console.error(`📊 Database: ${dbStatus[dbConnectionStatus] || dbConnectionStatus}`);
  
  // Dynamic tool list based on database availability
  let availableTools = [];
  if (dbConnectionStatus === 'connected') {
    if (embeddingProvider !== "none") {
      availableTools = ["vector_search", "metadata_search", "get_database_stats", "get_table_schemas", "insert_document"];
    } else {
      availableTools = ["metadata_search", "get_database_stats", "get_table_schemas"];
    }
  } else {
    availableTools = ["status_check"];
  }
  
  console.error(`🔧 Tools: ${availableTools.join(', ')}`);
}

main().catch((error) => {
  console.error("❌ Failed to start server:", error);
  process.exit(1);
});