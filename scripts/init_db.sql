-- OpenGenAI Database Initialization Script
-- This script sets up the initial database structure and data

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS opengenai;

-- Create test database for testing
CREATE DATABASE IF NOT EXISTS opengenai_test;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create user and grant permissions
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'opengenai') THEN
        CREATE ROLE opengenai WITH LOGIN PASSWORD 'opengenai_password';
    END IF;
END
$$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE opengenai TO opengenai;
GRANT ALL PRIVILEGES ON DATABASE opengenai_test TO opengenai;

-- Connect to opengenai database
\c opengenai;

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO opengenai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO opengenai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO opengenai;

-- Connect to test database
\c opengenai_test;

-- Grant schema permissions for test database
GRANT ALL ON SCHEMA public TO opengenai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO opengenai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO opengenai; 