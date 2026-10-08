"""
TrafficSentinel AI — Application Entry Point
Imports and launches the main Flask server application.
"""

from main import app

if __name__ == '__main__':
    import config
    import os
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port  = config.PORT
    host  = config.HOST
    print(f"🚀 Starting TrafficSentinel AI Server on http://{host}:{port}")
    app.run(debug=debug, threaded=True, host=host, port=port)
