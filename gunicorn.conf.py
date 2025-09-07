# Gunicorn configuration file
import multiprocessing

# Bind address
bind = "0.0.0.0:8000"

# Number of worker processes
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class
worker_class = 'sync'

# Timeout
timeout = 120

# Access log
accesslog = "access.log"

# Error log
errorlog = "error.log"

# Log level
loglevel = "info"

# Process name
proc_name = "stock_predictor"

# SSL (uncomment and modify if using HTTPS)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile"

# Maximum requests before worker restart
max_requests = 1000
max_requests_jitter = 50

# Preload application
preload_app = True

# Worker timeout
graceful_timeout = 120
