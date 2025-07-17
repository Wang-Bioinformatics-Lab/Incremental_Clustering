
run_first_round:
	nextflow run ./nf_workflow.nf -resume -c nextflow.config \
	--input_spectra ./data/round1 \
	--checkpoint_dir "" 


run_second_round:
	nextflow run ./nf_workflow.nf -resume -c nextflow.config \
	--input_spectra ./data/round2 \
	--checkpoint_dir /home/user/SourceCode/GNPS2_Webserver/workflows_user/Incremental_Clustering/nf_output/results \
	--publishdir ./nf_output_second_round