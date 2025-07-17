run:
	nextflow run ./nf_workflow.nf -resume -c nextflow.config

run_first_round:
	nextflow run ./nf_workflow.nf -resume -c nextflow.config --input_spectra ./data/round1

run_second_round:
	nextflow run ./nf_workflow.nf -resume -c nextflow.config --input_spectra ./data/round2