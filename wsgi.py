import os
import uvicorn
from api.main import app

# Point d'entrée pour le serveur, pour le développement
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# Pour la production, tu utiliseras Gunicorn avec UvicornWorker :
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker wsgi:app
