#!/bin/sh
# Label Studio on the public port 7860, the model server on localhost:9090 inside the container.
if [ -z "$POSTGRE_HOST" ] && [ "$ALLOW_EPHEMERAL" != "1" ]; then
  echo "REFUSING TO START: no POSTGRE_HOST secret. The Space disk is wiped on every restart," >&2
  echo "so your annotations would be lost. Add the Postgres secrets (README), or ALLOW_EPHEMERAL=1 for a test." >&2
  exit 1
fi
[ -n "$POSTGRE_HOST" ] && export DJANGO_DB=postgresql PGSSLMODE="${PGSSLMODE:-require}"
# Hugging Face sets SPACE_HOST; Django refuses HTTPS form posts from an origin it does not trust
[ -n "$SPACE_HOST" ] && export LABEL_STUDIO_HOST="https://$SPACE_HOST" \
  LABEL_STUDIO_CSRF_TRUSTED_ORIGINS="${LABEL_STUDIO_CSRF_TRUSTED_ORIGINS:-https://$SPACE_HOST}"

/opt/ml/bin/python src/tn.py serve --weights "$(ls runs/*/weights/best.pt | tail -1)" --device cpu &
exec /opt/ls/bin/label-studio start --internal-host 0.0.0.0 --port 7860 --no-browser
