#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_spectra = "/home/user/research/Incremental_Clustering/data"

//This publish dir is mostly  useful when we want to import modules in other workflows, keep it here usually don't change it
params.publishdir = "$baseDir/nf_output"
TOOL_FOLDER = "$baseDir/bin"
params.output_dir = "$baseDir/results"
params.checkpoint_dir  = "$baseDir/checkpoint"




process CLUSTERING {
    publishDir "$params.publishdir", mode: 'copy'
    conda "$baseDir/bin/conda_env.yml"

    input:
    path input

    output:
    file "results/*mzML" optional true
    file "results/*tsv" optional true
    file "results/*csv" optional true
    file "results/*h5" optional true

    script:
    """
    mkdir results
    python3 $TOOL_FOLDER/incremental_clustering.py \
        --folder $input \
        --checkpoint_dir "${params.checkpoint_dir}" \
        --output_dir results \
        --tool_dir $TOOL_FOLDER
    """
}

// TODO: This main will define the workflow that can then be imported, not really used right now, but can be


workflow {

    CLUSTERING(params.input_spectra)

    // Alternatively we can put everyhthing in the main from the above right here
}
