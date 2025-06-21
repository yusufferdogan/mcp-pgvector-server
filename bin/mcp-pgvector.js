#!/usr/bin/env node

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Path to the main server file
const serverPath = join(__dirname, '..', 'src', 'index.js');

// Check if we should show help
const args = process.argv.slice(2);
if (args.includes('--help') || args.includes('-h')) {
  console.log(`
MCP PGVector Server

Usage:
  npx mcp-pgvector-server [options]

Options:
  --help, -h          Show this help message
  --version, -v       Show version
  --config <file>     Path to config file (optional)

Environment Variables:
  DATABASE_URL        PostgreSQL connection string (required)
  
  Embedding Providers (choose one, or none for metadata-only mode):
  AZURE_OPENAI_API_KEY           Azure OpenAI API key (optional)
  AZURE_OPENAI_ENDPOINT          Azure OpenAI endpoint (optional)
  HUGGINGFACE_API_KEY            Hugging Face API key (optional)
  HUGGINGFACE_MODEL              HF model (default: sentence-transformers/all-MiniLM-L6-v2)
  EMBEDDING_PROVIDER             Force provider: auto|azure|huggingface|none (default: auto)
  
  Optional:
  MCP_SERVER_NAME     Server name (default: pgvector)
  
Examples:
  # With Azure OpenAI embeddings
  DATABASE_URL="postgresql://user:pass@localhost:5432/db" \\
  AZURE_OPENAI_API_KEY="your-key" \\
  AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/" \\
  npx mcp-pgvector-server
  
  # With Hugging Face embeddings
  DATABASE_URL="postgresql://user:pass@localhost:5432/db" \\
  HUGGINGFACE_API_KEY="your-hf-token" \\
  npx mcp-pgvector-server
  
  # Metadata-only mode (no embeddings)
  DATABASE_URL="postgresql://user:pass@localhost:5432/db" \\
  npx mcp-pgvector-server

For more information, visit: https://github.com/academic/mcp-pgvector-server
`);
  process.exit(0);
}

if (args.includes('--version') || args.includes('-v')) {
  const pkg = JSON.parse(
    await import('fs').then(fs => 
      fs.readFileSync(join(__dirname, '..', 'package.json'), 'utf8')
    )
  );
  console.log(pkg.version);
  process.exit(0);
}

// Start the MCP server
const child = spawn('node', [serverPath, ...args], {
  stdio: 'inherit',
  env: process.env
});

child.on('error', (error) => {
  console.error('Failed to start MCP PGVector server:', error.message);
  process.exit(1);
});

child.on('exit', (code) => {
  process.exit(code);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  child.kill('SIGINT');
});

process.on('SIGTERM', () => {
  child.kill('SIGTERM');
});