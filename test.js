#!/usr/bin/env node

// Test script for MCP PGVector Server
import dotenv from 'dotenv';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables
dotenv.config();

console.log('🧪 Testing MCP PGVector Server');
console.log('================================');

// Check environment variables
const requiredEnvs = ['DATABASE_URL', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT'];
const missingEnvs = requiredEnvs.filter(env => !process.env[env]);

if (missingEnvs.length > 0) {
  console.error('❌ Missing required environment variables:');
  missingEnvs.forEach(env => console.error(`   - ${env}`));
  console.error('\n💡 Copy .env.example to .env and fill in the values');
  process.exit(1);
}

console.log('✅ Environment variables loaded');
console.log(`📊 Database: ${process.env.DATABASE_URL.split('@')[1]?.split('/')[0] || 'connected'}`);
console.log(`🔑 Azure OpenAI: ${process.env.AZURE_OPENAI_ENDPOINT}`);

// Test tools list
console.log('\n🔧 Testing tools list...');

const serverPath = join(__dirname, 'src', 'index.js');
const child = spawn('node', [serverPath], {
  env: process.env,
  stdio: ['pipe', 'pipe', 'pipe']
});

// Send tools list request
const toolsRequest = JSON.stringify({
  jsonrpc: "2.0",
  id: 1,
  method: "tools/list",
  params: {}
});

child.stdin.write(toolsRequest + '\n');
child.stdin.end();

let output = '';
child.stdout.on('data', (data) => {
  output += data.toString();
});

child.stderr.on('data', (data) => {
  console.log('Server:', data.toString());
});

child.on('close', (code) => {
  if (output.trim()) {
    try {
      const response = JSON.parse(output.trim());
      if (response.result && response.result.tools) {
        console.log('✅ Tools list received:');
        response.result.tools.forEach(tool => {
          console.log(`   - ${tool.name}: ${tool.description}`);
        });
        console.log('\n🎉 MCP PGVector Server is ready!');
        
        console.log('\n📝 To use with Claude Code, add this to your MCP config:');
        console.log(`{
  "mcpServers": {
    "pgvector": {
      "command": "npx",
      "args": ["@mcp/pgvector-server"],
      "env": {
        "DATABASE_URL": "${process.env.DATABASE_URL}",
        "AZURE_OPENAI_API_KEY": "${process.env.AZURE_OPENAI_API_KEY}",
        "AZURE_OPENAI_ENDPOINT": "${process.env.AZURE_OPENAI_ENDPOINT}"
      }
    }
  }
}`);
      } else {
        console.error('❌ Unexpected response format:', output);
      }
    } catch (error) {
      console.error('❌ Failed to parse response:', error.message);
      console.error('Raw output:', output);
    }
  } else {
    console.error('❌ No output received from server');
  }
  
  process.exit(code);
});