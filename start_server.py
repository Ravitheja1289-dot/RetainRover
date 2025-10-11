"""
Startup script for InsuraSense API Backend
This script starts the FastAPI server with proper configuration.
"""

import uvicorn
import os
import sys
from pathlib import Path

def main():
    """Start the FastAPI server"""
    
    # Set environment variables
    os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent))
    
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    reload = os.getenv('RELOAD', 'true').lower() == 'true'
    log_level = os.getenv('LOG_LEVEL', 'info')
    
    print("🚀 Starting InsuraSense API Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📊 Log Level: {log_level}")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "api_backend:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
