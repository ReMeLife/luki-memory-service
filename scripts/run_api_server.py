#!/usr/bin/env python3
"""
Development server script for the Memory Service API
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from luki_memory.api.config import get_settings

def main():
    """Run the development server."""
    settings = get_settings()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting LUKi Memory Service API...")
    logger.info(f"Server will run at: http://{settings.host}:{settings.port}")
    logger.info(f"API documentation: http://{settings.host}:{settings.port}/docs")
    
    # Run the server
    uvicorn.run(
        "luki_memory.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1 if settings.debug else settings.workers
    )

if __name__ == "__main__":
    main()
