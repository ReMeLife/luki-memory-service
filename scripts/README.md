# LUKi Memory Service Scripts

This directory contains scripts for managing the LUKi Memory Service data ingestion processes.

## Architecture Overview

The memory service uses **two separate ingestion pipelines** to handle different types of data:

### 1. Project Context Loading (`load_project_context.py`)
- **Purpose**: One-time loading of static project knowledge and system context
- **Data Type**: System documentation, LUKi identity, technical specifications
- **Collection**: `project_context`
- **Frequency**: Run once during system setup, or when context documents are updated
- **Privacy**: No privacy controls needed (public/system knowledge)

### 2. ELR Data Ingestion (`elr_data_ingestion.py`)
- **Purpose**: Continuous processing of personal Electronic Life Records
- **Data Type**: User preferences, life stories, care data, family information
- **Collection**: `user_elr_data`
- **Frequency**: Continuous background process as new user data arrives
- **Privacy**: Full privacy controls, consent management, federated learning preparation

## Usage

### Initial System Setup
```bash
# Load project context (run once)
python scripts/load_project_context.py
```

### User Data Processing
```bash
# Process ELR data (run continuously or on-demand)
python scripts/elr_data_ingestion.py
```

## Key Benefits of Separation

1. **Scalability**: ELR processing can scale independently of system knowledge
2. **Privacy**: User data gets proper privacy controls while system data doesn't need them
3. **Performance**: System context loads once, user data processes continuously
4. **Federated Learning**: ELR pipeline is designed for federated learning preparation
5. **Maintenance**: System updates don't affect user data processing and vice versa

## Collections Structure

- `project_context`: Static system knowledge, LUKi identity, technical documentation
- `user_elr_data`: Personal user data with privacy controls and consent management

## Legacy Scripts

- `load_project_knowledge.py`: Original script (deprecated due to architecture mixing)
- `load_project_knowledge_fixed.py`: Temporary fix (deprecated)
- `load_project_knowledge_working.py`: Working version (deprecated in favor of separated approach)

Use the new separated scripts for all future development.
