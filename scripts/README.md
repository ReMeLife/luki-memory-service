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
# Note: Project context loading scripts have been removed from public repository
# See "Privacy Notice" section below for details
# Implement your own context loading based on the public API patterns
```

### User Data Processing
```bash
# Note: ELR data ingestion scripts contain proprietary logic
# See "Privacy Notice" section below for implementation guidance
# Use the public ingestion pipeline classes as reference
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

## Privacy Notice - Proprietary Scripts Removed

Several scripts mentioned in this documentation have been removed from the public repository as they contain proprietary business logic and sensitive configuration data:

- `load_project_knowledge.py`: Contains proprietary ReMeLife/LUKi domain knowledge
- `load_project_knowledge_fixed.py`: Contains sensitive business logic and tokenomics
- `load_project_knowledge_working.py`: Contains proprietary ecosystem details
- `load_project_context.py`: References internal project documents
- `elr_data_ingestion.py`: Contains proprietary ELR processing algorithms
- `load_test_elr_data.py`: Contains internal test data and configurations

These scripts have been sanitized from the repository to protect proprietary intellectual property while maintaining the core open-source memory service architecture.

For implementation guidance on creating your own data ingestion scripts, refer to the public API documentation and the core ingestion pipeline classes in `luki_memory/ingestion/`.
