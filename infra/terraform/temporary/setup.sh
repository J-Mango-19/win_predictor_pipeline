#!/bin/bash
set -e

# Configuration variables
S3_BUCKET_PATH="s3://cr-games-bucket/database/backup.dump"  
DUMP_FILE="/tmp/database.dump"
RESTORED_DB_NAME="cr_games"

# -------------------------------------------------------------------
# Step 1: Install Docker
# -------------------------------------------------------------------
echo "==> Installing Docker..."
dnf update -y
dnf install -y docker 

systemctl enable docker
systemctl start docker

docker --version


# -------------------------------------------------------------------
# Step 2: Pull PostgreSQL & Create Volume
# -------------------------------------------------------------------
echo "==> Setting up Docker volume and images..."
docker pull postgres:17
docker volume create postgres-data

# -------------------------------------------------------------------
# Step 3: Start PostgreSQL Container
# Note: We omit POSTGRES_DB so pg_restore -C can create it cleanly.
# -------------------------------------------------------------------
echo "==> Starting PostgreSQL container..."
docker run -d \
  --name postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=testpassword \
  -v postgres-data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:17

# Wait for PostgreSQL to become ready
echo "==> Waiting for PostgreSQL to start..."
until docker exec postgres pg_isready -U postgres; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done

echo "PostgreSQL is ready!"

# -------------------------------------------------------------------
# Step 4: Download the Database Dump from S3
# -------------------------------------------------------------------

echo "==> Downloading database dump from S3 ($S3_BUCKET_PATH)..."
aws s3 cp "$S3_BUCKET_PATH" "$DUMP_FILE" > /dev/null

# -------------------------------------------------------------------
# Step 5: Execute SQL to Create User/Role
# -------------------------------------------------------------------
echo "==> Creating role pipeline_worker..."
docker exec -i postgres psql -U postgres -c "
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'pipeline_worker') THEN
    CREATE ROLE pipeline_worker WITH LOGIN PASSWORD 'testpassword';
    ALTER ROLE pipeline_worker WITH CREATEDB REPLICATION LOGIN;
    ALTER TABLE games OWNER TO your_pipeline_db_user;
  END IF;
END
\$\$;"

# -------------------------------------------------------------------
# Step 6: Restore Database using pg_restore
# -------------------------------------------------------------------
echo "==> Copying dump file and restoring database..."
# Copy the dump into the container filesystem for fast local restore
docker cp "$DUMP_FILE" postgres:/tmp/database.dump

# Connect to the default 'postgres' database and execute -C (Create DB) restore
docker exec postgres pg_restore \
  -U postgres \
  -d postgres \
  -C \
  --no-owner \
  /tmp/database.dump

# Clean up dump inside container
docker exec postgres rm /tmp/database.dump

# -------------------------------------------------------------------
# Step 7: Grant Permissions to User on the Restored Database
# -------------------------------------------------------------------
echo "==> Granting permissions to pipeline_worker..."
docker exec -i postgres psql -U postgres -d "$RESTORED_DB_NAME" -c "
-- Grant database & schema access
GRANT ALL PRIVILEGES ON DATABASE $RESTORED_DB_NAME TO pipeline_worker;
GRANT ALL ON SCHEMA public TO pipeline_worker;

-- Grant access to existing tables and sequences
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pipeline_worker;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pipeline_worker;

-- Ensure future tables created in public schema are accessible
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO pipeline_worker;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO pipeline_worker;
"

echo "==> Database restore and permissions setup complete!"


# step 8: install uv, clone the repo onto this ec2 machine
echo "==> Installing git and uv..."

dnf install -y git

curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

echo "==> Installing project..."
mkdir -p /opt
git clone https://github.com/J-Mango-19/win_predictor_pipeline.git \
    /opt/classification-pipeline

echo "==> Verifying installation..."
command -v git
command -v uv
test -d /opt/classification-pipeline/services/ingestion

echo "==> Project setup complete!"
touch /opt/classification-pipeline/.setup-complete

# step 9: let prefect run the ingestion command remotely