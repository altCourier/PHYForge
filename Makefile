MODULATIONS   :=  BPSK QPSK 16-QAM 64-QAM 256-QAM 1024-QAM
MOD_DIR       :=  modulations
OUT_NAME      :=  sweep.h5
CONFIG_NAME   :=  config.json
CLI_MODULE    :=  physys/cli.py

SWEEP_OUTPUTS :=  $(foreach m,$(MODULATIONS),$(MOD_DIR)/$(m)/$(OUT_NAME))

# AMR benchmark dataset (single config.json's amr_dataset block ->
# Umi.h5/Uma.h5/Rma.h5, via `python -m physys.cli dataset`). Separate
# from the per-modulation SWEEP_OUTPUTS above -- this is one config,
# one command, three output files, not one-directory-per-modulation.
AMR_OUT_DIR   :=  data
AMR_OUTPUTS   :=  $(AMR_OUT_DIR)/Umi.h5 $(AMR_OUT_DIR)/Uma.h5 $(AMR_OUT_DIR)/Rma.h5

.PHONY: help dataset amr-dataset clean clean-amr-dataset check-configs $(MODULATIONS)

.DEFAULT_GOAL := help

# Sweeps typically hit a shared GPU (Sionna/TF) — force serial execution by
# default so `make -j4 dataset` doesn't contend for the same device.
# If your sweeps are CPU-only and safe to parallelize, comment this out.
.NOTPARALLEL:

help: ##  Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Per-modulation targets: $(MODULATIONS)"

check-configs: ## Verify every modulation has a config.json before running a sweep
	@status=0; \
	for m in $(MODULATIONS); do \
		f="$(MOD_DIR)/$$m/$(CONFIG_NAME)"; \
		if [ ! -f "$$f" ]; then \
			echo "MISSING: $$f"; \
			status=1; \
		fi; \
	done; \
	exit $$status

dataset: check-configs $(SWEEP_OUTPUTS) ##  Generate sweep.h5 for all modulations (per-modulation directories)

$(MOD_DIR)/%/$(OUT_NAME): $(MOD_DIR)/%/$(CONFIG_NAME) | $(CLI_MODULE)
	@echo "==> Generating $@ from $<"
	python -m physys.cli sweep -c $< -o $@ --iq

# Convenience per-modulation targets, e.g. `make BPSK`, `make QPSK`
$(MODULATIONS): %: $(MOD_DIR)/%/$(OUT_NAME)

amr-dataset: $(AMR_OUT_DIR)/.stamp ##  Generate Umi.h5/Uma.h5/Rma.h5 from config.json's amr_dataset block

# Single stamp file, not per-output rules: `physys.cli dataset` writes
# all three .h5 files in one process (one PHYSys build per modulation,
# reused across that modulation's SNR sweep) -- there's no way to
# regenerate just Uma.h5 without rerunning the whole thing anyway, so
# a three-way pattern rule here would just be misleading.
$(AMR_OUT_DIR)/.stamp: $(CONFIG_NAME) | $(CLI_MODULE)
	@mkdir -p $(AMR_OUT_DIR)
	@echo "==> Generating AMR dataset ($(AMR_OUTPUTS)) from $<"
	python -m physys.cli dataset -c $< -o $(AMR_OUT_DIR)
	@touch $@

clean: ##  Remove all generated sweep.h5 files (asks for confirmation)
	@echo "This will remove:"
	@for f in $(SWEEP_OUTPUTS); do [ -f "$$f" ] && echo "  $$f"; done; true
	@read -p "Proceed? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		rm -f $(SWEEP_OUTPUTS); \
		echo "Removed all generated sweep.h5 files"; \
	else \
		echo "Aborted."; \
	fi

clean-amr-dataset: ##  Remove Umi.h5/Uma.h5/Rma.h5 (asks for confirmation)
	@echo "This will remove:"
	@for f in $(AMR_OUTPUTS) $(AMR_OUT_DIR)/.stamp; do [ -f "$$f" ] && echo "  $$f"; done; true
	@read -p "Proceed? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		rm -f $(AMR_OUTPUTS) $(AMR_OUT_DIR)/.stamp; \
		echo "Removed AMR dataset files"; \
	else \
		echo "Aborted."; \
	fi