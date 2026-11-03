# swebench-sandbox:repaired = the unmodified organizer image plus /opt/repair, which
# local/swegemma_repaired.py runs in every container after the harness's own setup.
# Nothing is installed at build time; repair_env.py installs per task. Build with build_image.sh.
FROM swebench-sandbox:latest
COPY competition/ /opt/repair/competition/
COPY wheels/ /opt/repair/wheels/
COPY shims/ /opt/repair/shims/
COPY repair_env.py /opt/repair/repair_env.py
