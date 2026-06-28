#!/usr/bin/env bash
set -euo pipefail

# Auxiliary public volumes for encoder pretraining and external context.
# The 24-month curative-resection recurrence endpoint exists only in the private
# EOB-RECUR cohort; see sources.txt for access details.

DEST=${1:-data/public}
mkdir -p "${DEST}"

cat <<'TXT'
Fetch the public pretraining sets listed in sources.txt:
  LiTS          competitions.codalab.org/competitions/17094
  ATLAS         atlas-challenge.u-bourgogne.fr
  LLD-MMRI      github.com/LMMMEng/LLD-MMRI-Dataset
  CHAOS         zenodo.org/records/3431873
  HCC-TACE-Seg  cancerimagingarchive.net/collection/hcc-tace-seg
  TCGA-LIHC     cancerimagingarchive.net/collection/tcga-lihc
Each provider requires acceptance of its own licence before download.
TXT
