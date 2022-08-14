import os

from algo_flask import app

if __name__ == '__main__':
    import sys
    sys.path.append('../algo_research')
    app.run(debug=bool(os.getenv("DEBUG_MODE", default=False)))
