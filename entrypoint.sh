#!/bin/bash

# Navigate to the Django project directory
# We need to be where manage.py is for the following commands
cd /app/mirror_app

echo "--- Running Migrations ---"
# Apply database migrations to Supabase
python manage.py migrate --noinput

echo "--- Collecting Static Files ---"
# Required for Swagger/Admin to look right on Render
python manage.py collectstatic --noinput

echo "--- Starting Django Q Worker ---"
# Run the worker in the background for persona updates
python manage.py qcluster &

if [ "$DEBUG" = "True" ]; then
    echo "--- Starting Django Development Server (Auto-reload enabled) ---"
    # use runserver for local dev so changes reflect instantly
    exec python manage.py runserver 0.0.0.0:8000
else
    echo "--- Starting Gunicorn ---"
    # Start Gunicorn in the foreground for production (Render)
    exec gunicorn mirror_app.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 3
fi
